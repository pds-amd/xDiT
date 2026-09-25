from dataclasses import dataclass
from typing import Generic, Optional, Protocol, TypeVar

import torch

from .layout import PipeFusionPatchLayout


Payload = TypeVar("Payload")


@dataclass(frozen=True)
class PipeFusionStagePayload:
    image_state: torch.Tensor
    condition_state: Optional[torch.Tensor] = None


class PipeFusionPayloadCodec(Protocol, Generic[Payload]):
    def pack(self, payload: Payload, patch_index: int) -> torch.Tensor:
        ...

    def unpack(self, tensor: torch.Tensor, patch_index: int) -> Payload:
        ...


class CombinedTensorPayloadCodec:
    """Pack image and condition states into one ordered P2P message."""

    def __init__(
        self,
        layout: PipeFusionPatchLayout,
        *,
        require_condition: bool = True,
        model_name: str = "PipeFusion",
    ):
        self.layout = layout
        self.require_condition = require_condition
        self.model_name = model_name

    def pack(
        self,
        payload: PipeFusionStagePayload,
        patch_index: int,
    ) -> torch.Tensor:
        image_state = payload.image_state
        condition_state = payload.condition_state
        if condition_state is None:
            if self.require_condition:
                raise RuntimeError(
                    f"{self.model_name} stage payload requires condition state."
                )
            return image_state
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
    ) -> PipeFusionStagePayload:
        image_tokens = self.layout.image_tokens(patch_index)
        if tensor.shape[1] < image_tokens:
            raise RuntimeError(
                f"{self.model_name} stage payload for patch {patch_index} "
                f"contains {tensor.shape[1]} tokens; expected at least "
                f"{image_tokens} image tokens."
            )
        image_state = tensor[:, :image_tokens]
        condition_state = tensor[:, image_tokens:]
        if self.require_condition and condition_state.shape[1] == 0:
            raise RuntimeError(
                f"{self.model_name} stage payload for patch {patch_index} "
                "contains no condition tokens."
            )
        return PipeFusionStagePayload(
            image_state=image_state,
            condition_state=(
                condition_state
                if condition_state.shape[1] > 0
                else None
            ),
        )
