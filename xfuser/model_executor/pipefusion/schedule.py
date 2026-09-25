from dataclasses import dataclass
from typing import Callable, Generic, Optional, Protocol, Sequence, TypeVar

import torch

from .cache import PipeFusionStageOutputCache
from .transport import PipeFusionTransport, PipeFusionWorkItem


PreparedPatch = TypeVar("PreparedPatch")
StageOutput = TypeVar("StageOutput")
Result = TypeVar("Result")


def _never_interrupted() -> bool:
    return False


def _ignore_step(step_index: int, timestep):
    return None


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
                if outbound is not None:
                    self.transport.send(outbound, patch_index)
                if patch_index < self.transport.num_patches - 1:
                    self.transport.prefetch(work)
                self.advance_patch()
            deferred = self.hooks.end_step(step_index, timestep)
            if deferred is not None:
                for patch_index, tensor in deferred:
                    self.transport.send(tensor, patch_index)
            # Whole-step schedulers (for example Wan) must publish their
            # outputs before either stage starts negotiating the next
            # receive shape. Earlier patches can still prefetch normally.
            self.transport.prefetch(work)
        return self.hooks.finalize()
