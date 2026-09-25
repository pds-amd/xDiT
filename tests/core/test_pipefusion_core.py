import pytest
import torch
from types import SimpleNamespace

from xfuser.model_executor.pipefusion import (
    CombinedTensorPayloadCodec,
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
    PipeFusionStageOutputCache,
    PipeFusionStagePayload,
    PipeFusionTransport,
)


def test_patch_layout_is_immutable_and_fingerprints_reference_shape():
    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(2, 3),
        token_counts=(8, 12),
        reference_token_counts=(1, 2),
        name="target+reference",
    )

    assert layout.num_patches == 2
    assert layout.token_ranges == ((0, 8), (8, 20))
    assert layout.image_tokens(1) == 14
    assert layout.fingerprint[-1] == (1, 2)
    with pytest.raises(AttributeError):
        layout.split_dim = 1


def test_combined_payload_codec_round_trips_one_atomic_message():
    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(2,),
        token_counts=(3,),
    )
    codec = CombinedTensorPayloadCodec(layout, model_name="Test")
    image = torch.randn(1, 3, 4)
    condition = torch.randn(1, 2, 4)

    packed = codec.pack(PipeFusionStagePayload(image, condition), 0)
    unpacked = codec.unpack(packed, 0)

    assert torch.equal(unpacked.image_state, image)
    assert torch.equal(unpacked.condition_state, condition)


def test_layout_install_restores_runtime_state_after_exception():
    resets = []
    state = SimpleNamespace(
        pp_patches_token_num=[2, 2],
        pp_patches_token_start_idx_local=[0, 2, 4],
        pp_patches_token_start_end_idx_global=[[0, 2], [2, 4]],
        _reset_recv_buffer=lambda: resets.append(True),
    )
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(3, 2),
        token_counts=(3, 2),
        name="reference",
    )

    with pytest.raises(RuntimeError, match="stop"):
        with layout.install(state):
            assert state.pp_patches_token_num == [3, 2]
            raise RuntimeError("stop")

    assert state.pp_patches_token_num == [2, 2]
    assert not hasattr(state, "_xdit_pipefusion_layout_fingerprint")
    assert resets == [True, True]


class _FakeGroup:
    def __init__(self, incoming):
        self.incoming = iter(incoming)
        self.queued = []
        self.receiving = []
        self.sent = []

    def add_pipeline_recv_task(self, idx, name):
        self.queued.append((name, idx))

    def recv_next(self):
        self.receiving.append(self.queued.pop(0))

    def get_pipeline_recv_data(self, idx, name):
        assert self.receiving.pop(0) == (name, idx)
        return next(self.incoming)

    def pipeline_isend(self, tensor, name, segment_idx):
        self.sent.append((name, segment_idx, tensor.clone()))


class _Hooks:
    def __init__(self):
        self.forwarded = []
        self.steps = []

    def interrupted(self):
        return False

    def begin_step(self, step_index, timestep):
        self.steps.append(("begin", step_index, timestep))

    def prepare_patch(self, work, timestep, received):
        return received

    def forward_patch(self, work, timestep, prepared):
        self.forwarded.append((work.step_index, work.patch_index))
        return prepared + 1

    def commit_patch(self, work, timestep, output):
        return output

    def end_step(self, step_index, timestep):
        self.steps.append(("end", step_index, timestep))

    def finalize(self):
        return "done"


def test_async_driver_owns_receive_cache_send_and_patch_order():
    group = _FakeGroup(
        [torch.tensor(value) for value in (10, 11, 20, 21)]
    )
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=2,
        num_patches=2,
    )
    hooks = _Hooks()
    advances = []
    driver = PipeFusionAsyncDriver(
        transport=transport,
        output_cache=PipeFusionStageOutputCache((1, 0), 2),
        hooks=hooks,
        advance_patch=lambda: advances.append(True),
    )

    assert driver.run((100, 200)) == "done"
    assert hooks.forwarded == [(0, 0), (0, 1)]
    assert hooks.steps == [
        ("begin", 0, 100),
        ("end", 0, 100),
        ("begin", 1, 200),
        ("end", 1, 200),
    ]
    assert [entry[2].item() for entry in group.sent] == [11, 12, 11, 12]
    assert len(advances) == 4
    assert not group.queued
    assert not group.receiving


def test_async_driver_sends_whole_step_scheduler_outputs():
    class DeferredHooks(_Hooks):
        def end_step(self, step_index, timestep):
            super().end_step(step_index, timestep)
            return [(0, torch.tensor(100 + step_index))]

    group = _FakeGroup([torch.tensor(value) for value in (10, 20)])
    driver = PipeFusionAsyncDriver(
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=2,
            num_patches=1,
        ),
        output_cache=PipeFusionStageOutputCache((1, 1), 1),
        hooks=DeferredHooks(),
        advance_patch=lambda: None,
    )

    driver.run((1, 2))

    assert [entry[2].item() for entry in group.sent] == [
        11,
        100,
        21,
        101,
    ]
