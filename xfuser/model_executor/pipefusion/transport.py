from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class PipeFusionWorkItem:
    step_index: int
    patch_index: int
    num_steps: int
    num_patches: int
    first_stage: bool

    @property
    def needs_receive(self) -> bool:
        return not self.first_stage or self.step_index > 0

    @property
    def has_next_receive(self) -> bool:
        return self.needs_receive and not (
            self.step_index == self.num_steps - 1
            and self.patch_index == self.num_patches - 1
        )


class PipeFusionTransport:
    """Own ordered single-message P2P transport for an async schedule."""

    def __init__(
        self,
        group,
        *,
        first_stage: bool,
        num_steps: int,
        num_patches: int,
        payload_name: str = "latent",
    ):
        self.group = group
        self.first_stage = first_stage
        self.num_steps = num_steps
        self.num_patches = num_patches
        self.payload_name = payload_name
        self._recv_primed = False

    def queue_receives(self) -> None:
        receive_steps = self.num_steps - 1 if self.first_stage else self.num_steps
        for _ in range(receive_steps):
            for patch_index in range(self.num_patches):
                self.group.add_pipeline_recv_task(
                    patch_index,
                    self.payload_name,
                )

    def receive(self, work: PipeFusionWorkItem) -> Optional[torch.Tensor]:
        if not work.needs_receive:
            return None
        if not self._recv_primed:
            self.group.recv_next()
        tensor = self.group.get_pipeline_recv_data(
            idx=work.patch_index,
            name=self.payload_name,
        )
        self._recv_primed = False
        return tensor

    def prefetch(self, work: PipeFusionWorkItem) -> None:
        if work.has_next_receive:
            self.group.recv_next()
            self._recv_primed = True

    def send(self, tensor: torch.Tensor, patch_index: int) -> None:
        self.group.pipeline_isend(
            tensor,
            name=self.payload_name,
            segment_idx=patch_index,
        )
