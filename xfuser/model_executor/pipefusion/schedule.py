from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, Sequence, TypeVar

import torch

from .cache import PipeFusionStageOutputCache
from .payload import CombinedTensorPayloadCodec, PipeFusionStagePayload
from .transport import PipeFusionTransport, PipeFusionWorkItem


PreparedPatch = TypeVar("PreparedPatch")
StageOutput = TypeVar("StageOutput")
Result = TypeVar("Result")


def _never_interrupted() -> bool:
    return False


def _ignore_step(step_index: int, timestep):
    return None


def pipefusion_should_update_progress(
    step_index: int,
    *,
    num_async_steps: int,
    pipeline_warmup_steps: int,
    num_warmup_steps: int,
    scheduler_order: int,
) -> bool:
    """Return whether an async PipeFusion step completes scheduler progress."""
    step = step_index + pipeline_warmup_steps + 1
    return (
        step_index == num_async_steps - 1
        or (step > num_warmup_steps and step % scheduler_order == 0)
    )


@dataclass
class PipeFusionAsyncCallbacks(Generic[PreparedPatch, StageOutput, Result]):
    """Functional adapter for model-specific PipeFusion schedule hooks."""

    prepare_patch_fn: Callable[
        [PipeFusionWorkItem, object, Optional[torch.Tensor]],
        PreparedPatch,
    ]
    forward_patch_fn: Callable[
        [PipeFusionWorkItem, object, PreparedPatch],
        StageOutput,
    ]
    commit_patch_fn: Callable[
        [PipeFusionWorkItem, object, StageOutput],
        Optional[torch.Tensor],
    ]
    finalize_fn: Callable[[], Result]
    interrupted_fn: Callable[[], bool] = _never_interrupted
    begin_step_fn: Callable[[int, object], None] = _ignore_step
    end_step_fn: Callable[
        [int, object],
        Optional[Sequence[tuple[int, torch.Tensor]]],
    ] = _ignore_step
    defer_last_prefetch: bool = False
    defer_final_advance: bool = False

    def interrupted(self) -> bool:
        return self.interrupted_fn()

    def begin_step(self, step_index: int, timestep) -> None:
        self.begin_step_fn(step_index, timestep)

    def prepare_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        received: Optional[torch.Tensor],
    ) -> PreparedPatch:
        return self.prepare_patch_fn(work, timestep, received)

    def forward_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        prepared: PreparedPatch,
    ) -> StageOutput:
        return self.forward_patch_fn(work, timestep, prepared)

    def commit_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        output: StageOutput,
    ) -> Optional[torch.Tensor]:
        return self.commit_patch_fn(work, timestep, output)

    def end_step(
        self,
        step_index: int,
        timestep,
    ) -> Optional[Sequence[tuple[int, torch.Tensor]]]:
        return self.end_step_fn(step_index, timestep)

    def finalize(self) -> Result:
        return self.finalize_fn()


class PipeFusionAsyncHooks(
    Protocol,
    Generic[PreparedPatch, StageOutput, Result],
):
    defer_last_prefetch: bool
    defer_final_advance: bool

    def interrupted(self) -> bool:
        ...

    def begin_step(self, step_index: int, timestep) -> None:
        ...

    def prepare_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        received: Optional[torch.Tensor],
    ) -> PreparedPatch:
        ...

    def forward_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        prepared: PreparedPatch,
    ) -> StageOutput:
        ...

    def commit_patch(
        self,
        work: PipeFusionWorkItem,
        timestep,
        output: StageOutput,
    ) -> Optional[torch.Tensor]:
        """Apply scheduler/model state and return an outbound payload."""
        ...

    def end_step(
        self,
        step_index: int,
        timestep,
    ) -> Optional[Sequence[tuple[int, torch.Tensor]]]:
        """Return payloads whose scheduler requires whole-step assembly."""
        ...

    def finalize(self) -> Result:
        ...


class PipeFusionAsyncDriver(Generic[PreparedPatch, StageOutput, Result]):
    """Model-agnostic receive/compute/cache/send PipeFusion state machine."""

    def __init__(
        self,
        *,
        transport: PipeFusionTransport,
        output_cache: PipeFusionStageOutputCache[StageOutput],
        hooks: PipeFusionAsyncHooks[PreparedPatch, StageOutput, Result],
        advance_patch,
    ):
        self.transport = transport
        self.output_cache = output_cache
        self.hooks = hooks
        self.advance_patch = advance_patch

    def run(self, timesteps: Sequence) -> Result:
        num_steps = len(timesteps)
        if num_steps != self.transport.num_steps:
            raise ValueError(
                "PipeFusion transport timestep count does not match the driver: "
                f"{self.transport.num_steps} != {num_steps}."
            )
        if len(self.output_cache.computation_mask) != num_steps:
            raise ValueError(
                "PipeFusion computation mask length does not match the driver: "
                f"{len(self.output_cache.computation_mask)} != {num_steps}."
            )

        self.transport.queue_receives()
        for step_index, timestep in enumerate(timesteps):
            if self.hooks.interrupted():
                continue
            self.hooks.begin_step(step_index, timestep)
            for patch_index in range(self.transport.num_patches):
                work = PipeFusionWorkItem(
                    step_index=step_index,
                    patch_index=patch_index,
                    num_steps=num_steps,
                    num_patches=self.transport.num_patches,
                    first_stage=self.transport.first_stage,
                )
                received = self.transport.receive(work)
                prepared = self.hooks.prepare_patch(
                    work,
                    timestep,
                    received,
                )
                output = self.output_cache.resolve(
                    step_index,
                    patch_index,
                    lambda: self.hooks.forward_patch(
                        work,
                        timestep,
                        prepared,
                    ),
                )
                outbound = self.hooks.commit_patch(work, timestep, output)
                final_patch = (
                    patch_index == self.transport.num_patches - 1
                )
                if final_patch and not self.hooks.defer_last_prefetch:
                    # Prime the circular feedback receive before both stages
                    # enqueue their final-patch sends. Posting both sends first
                    # can deadlock two-rank NCCL process groups.
                    self.transport.prefetch(work)
                if outbound is not None:
                    self.transport.send(outbound, patch_index)
                if not final_patch:
                    self.transport.prefetch(work)
                if not (
                    final_patch and self.hooks.defer_final_advance
                ):
                    self.advance_patch()
            deferred = self.hooks.end_step(step_index, timestep)
            if self.hooks.defer_final_advance:
                self.advance_patch()
            if deferred is not None:
                for patch_index, tensor in deferred:
                    self.transport.send(tensor, patch_index)
            # Whole-step schedulers (for example Wan) must publish their
            # outputs before either stage starts negotiating the next
            # receive shape. Earlier patches can still prefetch normally.
            if self.hooks.defer_last_prefetch:
                self.transport.prefetch(work)
        self.transport.wait_sends()
        return self.hooks.finalize()


class PipeFusionImagePatchSchedule(Generic[Result]):
    """Shared image-plus-condition PipeFusion adapter.

    FLUX-family denoisers return an image prediction and a propagated text
    state. This adapter owns the repeated receive/unpack, condition forwarding,
    scheduler-feedback, and send lifecycle; models provide only their forward,
    last-stage scheduler update, progress, and final assembly policies.
    """

    def __init__(
        self,
        *,
        patch_latents: list[torch.Tensor],
        layout,
        initial_condition: torch.Tensor | Callable[[], torch.Tensor],
        first_stage: bool,
        last_stage: bool,
        condition_reuse: bool,
        forward_patch_fn: Callable[
            [PipeFusionWorkItem, object, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, Optional[torch.Tensor]],
        ],
        update_last_patch_fn: Callable[
            [PipeFusionWorkItem, object, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
        finalize_fn: Callable[[], Result],
        begin_step_fn: Callable[[int, object], None] = _ignore_step,
        end_step_fn: Callable[[int, object], None] = _ignore_step,
        interrupted_fn: Callable[[], bool] = _never_interrupted,
        model_name: str = "PipeFusion",
    ):
        self.patch_latents = patch_latents
        self.layout = layout
        self.initial_condition = initial_condition
        self.first_stage = first_stage
        self.last_stage = last_stage
        self.condition_reuse = condition_reuse
        self.forward_patch_fn = forward_patch_fn
        self.update_last_patch_fn = update_last_patch_fn
        self.finalize_fn = finalize_fn
        self.begin_step_fn = begin_step_fn
        self.end_step_fn = end_step_fn
        self.interrupted_fn = interrupted_fn
        self.model_name = model_name

    def run(
        self,
        *,
        timesteps: Sequence,
        output_cache: PipeFusionStageOutputCache,
        transport: PipeFusionTransport,
        advance_patch: Callable[[], None],
    ) -> Result:
        if len(self.patch_latents) != self.layout.num_patches:
            raise ValueError(
                f"{self.model_name} patch latent count does not match layout."
            )

        payload_codec = CombinedTensorPayloadCodec(
            self.layout,
            model_name=self.model_name,
        )
        previous = (
            [None] * self.layout.num_patches if self.last_stage else None
        )
        condition_states = [None] * self.layout.num_patches
        step_condition = [None]

        def begin_step(step_index, timestep):
            step_condition[0] = None
            self.begin_step_fn(step_index, timestep)

        def prepare_patch(work, _timestep, received):
            patch_index = work.patch_index
            if self.last_stage:
                previous[patch_index] = self.patch_latents[patch_index]
            if received is not None:
                if self.first_stage:
                    self.patch_latents[patch_index] = received
                else:
                    payload = payload_codec.unpack(received, patch_index)
                    self.patch_latents[patch_index] = payload.image_state
                    condition_states[patch_index] = payload.condition_state
            if self.first_stage:
                condition = (
                    self.initial_condition()
                    if callable(self.initial_condition)
                    else self.initial_condition
                )
            elif self.condition_reuse and step_condition[0] is not None:
                condition = step_condition[0]
            else:
                condition = condition_states[patch_index]
            if condition is None:
                raise RuntimeError(
                    f"{self.model_name} missing condition state for patch "
                    f"{patch_index}."
                )
            return PipeFusionStagePayload(
                image_state=self.patch_latents[patch_index],
                condition_state=condition,
            )

        def forward_patch(work, timestep, prepared):
            return self.forward_patch_fn(
                work,
                timestep,
                prepared.image_state,
                prepared.condition_state,
            )

        def commit_patch(work, timestep, output):
            patch_index = work.patch_index
            image_state, next_condition = output
            self.patch_latents[patch_index] = image_state
            if self.last_stage:
                updated = self.update_last_patch_fn(
                    work,
                    timestep,
                    image_state,
                    previous[patch_index],
                )
                self.patch_latents[patch_index] = updated
                return (
                    None
                    if work.step_index == work.num_steps - 1
                    else updated
                )

            if self.condition_reuse:
                if patch_index == 0:
                    step_condition[0] = next_condition
                next_condition = step_condition[0]
            if next_condition is None:
                raise RuntimeError(
                    f"{self.model_name} produced no condition state for patch "
                    f"{patch_index}."
                )
            return payload_codec.pack(
                PipeFusionStagePayload(
                    image_state=image_state,
                    condition_state=next_condition,
                ),
                patch_index,
            )

        callbacks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=self.finalize_fn,
            interrupted_fn=self.interrupted_fn,
            begin_step_fn=begin_step,
            end_step_fn=self.end_step_fn,
        )
        return PipeFusionAsyncDriver(
            transport=transport,
            output_cache=output_cache,
            hooks=callbacks,
            advance_patch=advance_patch,
        ).run(timesteps)
