"""PipeFusion schedule for joint image/text transformer denoisers."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence, TypeVar

import torch

from ..cache import PipeFusionStageOutputCache
from ..layout import PipeFusionPatchLayout
from ..schedule import (
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    _ignore_step,
    _never_interrupted,
)
from ..transport import PipeFusionTransport, PipeFusionWorkItem


Result = TypeVar("Result")


class ConditionPropagationMode(str, Enum):
    """How a joint transformer propagates text state between PP stages."""

    REUSE_FIRST_PATCH = "reuse_first_patch"
    FORWARD_PER_PATCH = "forward_per_patch"


@dataclass(frozen=True)
class JointImageTextPayload:
    image_state: torch.Tensor
    condition_state: Optional[torch.Tensor] = None


class JointImageTextPayloadCodec:
    """Encode image tokens and propagated text state as one P2P tensor."""

    def __init__(
        self,
        layout: PipeFusionPatchLayout,
        *,
        model_name: str,
    ):
        self.layout = layout
        self.model_name = model_name

    def pack(
        self,
        payload: JointImageTextPayload,
        patch_index: int,
        *,
        include_condition: bool = True,
    ) -> torch.Tensor:
        image_state = payload.image_state
        condition_state = payload.condition_state
        if not include_condition:
            return image_state
        if condition_state is None:
            raise RuntimeError(
                f"{self.model_name} joint schedule requires condition state."
            )
        if image_state.shape[0] != condition_state.shape[0]:
            raise ValueError(
                f"{self.model_name} image and condition batch sizes differ: "
                f"{image_state.shape[0]} != {condition_state.shape[0]}."
            )
        return torch.cat((image_state, condition_state), dim=1)

    def unpack(
        self,
        tensor: torch.Tensor,
        patch_index: int,
        *,
        includes_condition: bool = True,
    ) -> JointImageTextPayload:
        image_tokens = self.layout.image_tokens(patch_index)
        if not includes_condition:
            if tensor.shape[1] != image_tokens:
                raise RuntimeError(
                    f"{self.model_name} image-only payload for patch "
                    f"{patch_index} has {tensor.shape[1]} tokens; expected "
                    f"{image_tokens}."
                )
            return JointImageTextPayload(image_state=tensor)
        if tensor.shape[1] <= image_tokens:
            raise RuntimeError(
                f"{self.model_name} stage payload for patch {patch_index} "
                "contains no condition tokens."
            )
        return JointImageTextPayload(
            image_state=tensor[:, :image_tokens],
            condition_state=tensor[:, image_tokens:],
        )


class JointImageTextPatchSchedule:
    """Adapt the generic async driver to joint image/text transformer semantics."""

    def __init__(
        self,
        *,
        patch_latents: list[torch.Tensor],
        layout: PipeFusionPatchLayout,
        initial_condition: torch.Tensor | Callable[[], torch.Tensor],
        first_stage: bool,
        last_stage: bool,
        condition_mode: ConditionPropagationMode,
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
        self.condition_mode = condition_mode
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
        if self.layout.num_patches != transport.num_patches:
            raise ValueError(
                f"{self.model_name} patch layout does not match transport."
            )

        joint_payload_codec = JointImageTextPayloadCodec(
            self.layout,
            model_name=self.model_name,
        )
        previous_patch_states = (
            [None] * self.layout.num_patches if self.last_stage else None
        )
        received_condition_states = [None] * self.layout.num_patches
        first_patch_condition = [None]

        def begin_step(step_index, timestep):
            first_patch_condition[0] = None
            self.begin_step_fn(step_index, timestep)

        def prepare_patch(work, _timestep, received):
            patch_index = work.patch_index
            if self.last_stage:
                previous_patch_states[patch_index] = self.patch_latents[
                    patch_index
                ]
            if received is not None:
                if self.first_stage:
                    self.patch_latents[patch_index] = received
                else:
                    includes_condition = not (
                        self.condition_mode
                        is ConditionPropagationMode.REUSE_FIRST_PATCH
                        and patch_index > 0
                    )
                    received_payload = joint_payload_codec.unpack(
                        received,
                        patch_index,
                        includes_condition=includes_condition,
                    )
                    self.patch_latents[patch_index] = (
                        received_payload.image_state
                    )
                    if includes_condition:
                        received_condition_states[patch_index] = (
                            received_payload.condition_state
                        )
            if self.first_stage:
                condition = (
                    self.initial_condition()
                    if callable(self.initial_condition)
                    else self.initial_condition
                )
            elif (
                self.condition_mode
                is ConditionPropagationMode.REUSE_FIRST_PATCH
                and first_patch_condition[0] is not None
            ):
                condition = first_patch_condition[0]
            else:
                condition = received_condition_states[patch_index]
            if condition is None:
                raise RuntimeError(
                    f"{self.model_name} missing condition state for patch "
                    f"{patch_index}."
                )
            if (
                self.last_stage
                and self.condition_mode
                is ConditionPropagationMode.REUSE_FIRST_PATCH
                and patch_index == 0
            ):
                first_patch_condition[0] = condition
            return JointImageTextPayload(
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
                    previous_patch_states[patch_index],
                )
                self.patch_latents[patch_index] = updated
                return (
                    None
                    if work.step_index == work.num_steps - 1
                    else updated
                )

            if (
                self.condition_mode
                is ConditionPropagationMode.REUSE_FIRST_PATCH
            ):
                if patch_index == 0:
                    first_patch_condition[0] = next_condition
                next_condition = first_patch_condition[0]
            if next_condition is None:
                raise RuntimeError(
                    f"{self.model_name} produced no condition state for patch "
                    f"{patch_index}."
                )
            return joint_payload_codec.pack(
                JointImageTextPayload(
                    image_state=image_state,
                    condition_state=next_condition,
                ),
                patch_index,
                include_condition=not (
                    self.condition_mode
                    is ConditionPropagationMode.REUSE_FIRST_PATCH
                    and patch_index > 0
                ),
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
