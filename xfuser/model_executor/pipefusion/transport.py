from dataclasses import dataclass
from typing import Optional

import torch
from torch.profiler import record_function

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
        if (
            self.step_index == self.num_steps - 1
            and self.patch_index == self.num_patches - 1
        ):
            return False
        next_step = (
            self.step_index
            if self.patch_index < self.num_patches - 1
            else self.step_index + 1
        )
        return not self.first_stage or next_step > 0


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
        if num_steps < 1:
            raise ValueError("PipeFusion transport requires at least one timestep.")
        if num_patches < 1:
            raise ValueError("PipeFusion transport requires at least one patch.")
        self.group = group
        self.first_stage = first_stage
        self.num_steps = num_steps
        self.num_patches = num_patches
        self.payload_name = payload_name
        self._recv_primed = False
        self._receives_queued = False
        self._send_tasks = {}
        self._send_tensors = {}
        self._fixed_buffer_state = None

    def configure_fixed_payload_shapes(
        self,
        *,
        send_shapes: list[torch.Size],
        recv_shapes: list[torch.Size],
    ) -> None:
        """Avoid first-use shape negotiation for a statically known payload."""
        self._save_payload_buffer_state()
        dtype = getattr(self.group, "dtype", None)
        if dtype is None:
            raise RuntimeError(
                "PipeFusion fixed payload shapes require a configured dtype."
            )
        self.group.set_fixed_payload_buffers(
            self.payload_name,
            send_shapes=send_shapes,
            recv_shapes=recv_shapes,
            dtype=dtype,
        )

    def _save_payload_buffer_state(self) -> None:
        if self._fixed_buffer_state is not None:
            return
        name = self.payload_name
        self._fixed_buffer_state = {
            attribute: (
                name in getattr(self.group, attribute),
                getattr(self.group, attribute).get(name),
            )
            for attribute in ("send_shape", "recv_shape", "recv_buffer")
        }
        self._fixed_buffer_state["was_fixed"] = (
            name in self.group.fixed_payload_names
        )

    def _restore_payload_buffer_state(self) -> None:
        if self._fixed_buffer_state is None:
            return
        name = self.payload_name
        for attribute in ("send_shape", "recv_shape", "recv_buffer"):
            existed, value = self._fixed_buffer_state[attribute]
            buffers = getattr(self.group, attribute)
            if existed:
                buffers[name] = value
            else:
                buffers.pop(name, None)
        if self._fixed_buffer_state["was_fixed"]:
            self.group.fixed_payload_names.add(name)
        else:
            self.group.fixed_payload_names.discard(name)
        self._fixed_buffer_state = None

    def queue_receives(self) -> None:
        if self._receives_queued:
            raise RuntimeError("PipeFusion receives have already been queued.")
        receive_steps = self.num_steps - 1 if self.first_stage else self.num_steps
        for _ in range(receive_steps):
            for patch_index in range(self.num_patches):
                self.group.add_pipeline_recv_task(
                    patch_index,
                    self.payload_name,
                )
        self._receives_queued = True

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
            with record_function("xdit::pipefusion.prefetch_launch"):
                self.group.recv_next()
            self._recv_primed = True

    def send(self, tensor: torch.Tensor, patch_index: int) -> None:
        previous_send_work = self._send_tasks.pop(patch_index, None)
        if previous_send_work is not None:
            with record_function("xdit::pipefusion.send_wait_previous"):
                previous_send_work.wait()
            self._send_tensors.pop(patch_index, None)
        wire_dtype = getattr(self.group, "dtype", None)
        if (
            wire_dtype is not None
            and tensor.is_floating_point()
            and tensor.dtype != wire_dtype
        ):
            tensor = tensor.to(wire_dtype)
        # pipeline_isend() also makes its input contiguous. Do it here so this
        # retained reference is the tensor owned by the asynchronous NCCL work,
        # rather than a pre-contiguous view whose temporary send buffer can die.
        tensor = tensor.contiguous()
        with record_function("xdit::pipefusion.send_launch"):
            work = self.group.pipeline_isend(
                tensor,
                name=self.payload_name,
                segment_idx=patch_index,
            )
        self._send_tasks[patch_index] = work
        self._send_tensors[patch_index] = tensor

    def wait_sends(self) -> None:
        for work in self._send_tasks.values():
            work.wait()
        self._send_tasks.clear()
        self._send_tensors.clear()
        self._restore_payload_buffer_state()
