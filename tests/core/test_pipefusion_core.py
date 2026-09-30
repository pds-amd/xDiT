import pytest
import torch
from types import SimpleNamespace

from xfuser.model_executor.pipefusion import (
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
    PipeFusionStageOutputCache,
    PipeFusionTransport,
    pipefusion_should_update_progress,
)
from xfuser.model_executor.pipefusion.schedules import (
    ConditionPropagationMode,
    JointImageTextPatchSchedule,
    JointImageTextPayload,
    JointImageTextPayloadCodec,
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


def test_joint_payload_codec_round_trips_one_atomic_message():
    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(2,),
        token_counts=(3,),
    )
    codec = JointImageTextPayloadCodec(layout, model_name="Test")
    image = torch.randn(1, 3, 4)
    condition = torch.randn(1, 2, 4)

    packed = codec.pack(JointImageTextPayload(image, condition), 0)
    unpacked = codec.unpack(packed, 0)

    assert torch.equal(unpacked.image_state, image)
    assert torch.equal(unpacked.condition_state, condition)


def test_joint_payload_codec_round_trips_image_only_message():
    layout = PipeFusionPatchLayout(
        split_dim=2,
        split_sizes=(2,),
        token_counts=(3,),
    )
    codec = JointImageTextPayloadCodec(layout, model_name="Test")
    image = torch.randn(1, 3, 4)

    packed = codec.pack(
        JointImageTextPayload(image),
        0,
        include_condition=False,
    )
    unpacked = codec.unpack(packed, 0, includes_condition=False)

    assert torch.equal(unpacked.image_state, image)
    assert unpacked.condition_state is None


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


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _FakeGroup:
    def __init__(self, incoming):
        self.dtype = torch.float16
        self.incoming = iter(incoming)
        self.queued = []
        self.receiving = []
        self.sent = []
        self.blocking_sent = []
        self.send_work = []
        self.send_shape = {}
        self.recv_shape = {}
        self.recv_buffer = {}
        self.fixed_payload_names = set()

    def set_fixed_payload_buffers(self, name, *, send_shapes, recv_shapes, dtype):
        self.send_shape[name] = {
            index: torch.Size(shape) for index, shape in enumerate(send_shapes)
        }
        self.recv_shape[name] = {
            index: torch.Size(shape) for index, shape in enumerate(recv_shapes)
        }
        self.recv_buffer[name] = {
            index: torch.zeros(shape, dtype=dtype)
            for index, shape in enumerate(recv_shapes)
        }
        self.fixed_payload_names.add(name)

    def add_pipeline_recv_task(self, idx, name):
        self.queued.append((name, idx))

    def recv_next(self):
        self.receiving.append(self.queued.pop(0))

    def get_pipeline_recv_data(self, idx, name):
        assert self.receiving.pop(0) == (name, idx)
        return next(self.incoming)

    def pipeline_isend(self, tensor, name, segment_idx):
        self.sent.append((name, segment_idx, tensor.clone()))
        work = _FakeWork()
        self.send_work.append(work)
        return work

    def pipeline_send(self, tensor, name, segment_idx):
        self.blocking_sent.append((name, segment_idx, tensor.clone()))


class _Hooks:
    defer_last_prefetch = False
    defer_final_advance = False

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
    assert all(work.waited for work in group.send_work)


def test_async_driver_sends_whole_step_scheduler_outputs():
    class DeferredHooks(_Hooks):
        defer_last_prefetch = True
        defer_final_advance = True

        def end_step(self, step_index, timestep):
            super().end_step(step_index, timestep)
            return [(0, torch.tensor(float(100 + step_index)))]

    group = _FakeGroup([torch.tensor(float(value)) for value in (10, 20)])
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
    assert all(entry[2].dtype == torch.float16 for entry in group.sent)


def test_async_driver_rejects_a_mask_that_does_not_match_timesteps():
    driver = PipeFusionAsyncDriver(
        transport=PipeFusionTransport(
            _FakeGroup([torch.tensor(1)]),
            first_stage=False,
            num_steps=1,
            num_patches=1,
        ),
        output_cache=PipeFusionStageOutputCache((1, 1), 1),
        hooks=_Hooks(),
        advance_patch=lambda: None,
    )

    with pytest.raises(ValueError, match="computation mask length"):
        driver.run((1,))


def test_image_patch_schedule_reuses_patch_zero_condition_state():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1, 1),
        token_counts=(1, 1),
    )
    first_payload = torch.tensor([[[10.0], [20.0]]])
    second_payload = torch.tensor([[[11.0]]])
    group = _FakeGroup([first_payload, second_payload])
    seen_conditions = []
    patches = [None, None]
    schedule = JointImageTextPatchSchedule(
        patch_latents=patches,
        layout=layout,
        initial_condition=torch.zeros(1, 1, 1),
        first_stage=False,
        last_stage=False,
        condition_mode=ConditionPropagationMode.REUSE_FIRST_PATCH,
        forward_patch_fn=lambda _work, _time, image, condition: (
            image + 1,
            seen_conditions.append(condition.clone()) or condition + 1,
        ),
        update_last_patch_fn=lambda _work, _time, image, _previous: image,
        finalize_fn=lambda: "done",
        model_name="test",
    )

    assert schedule.run(
        timesteps=(1,),
        output_cache=PipeFusionStageOutputCache((1,), 2),
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=1,
            num_patches=2,
        ),
        advance_patch=lambda: None,
    ) == "done"
    assert torch.equal(seen_conditions[0], torch.tensor([[[20.0]]]))
    assert torch.equal(seen_conditions[1], torch.tensor([[[21.0]]]))
    assert [tensor.shape[1] for _, _, tensor in group.sent] == [2, 1]


def test_image_patch_schedule_forwards_condition_for_every_patch_when_required():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1, 1),
        token_counts=(1, 1),
    )
    group = _FakeGroup(
        [
            torch.tensor([[[10.0], [20.0]]]),
            torch.tensor([[[11.0], [99.0]]]),
        ]
    )
    seen_conditions = []
    schedule = JointImageTextPatchSchedule(
        patch_latents=[None, None],
        layout=layout,
        initial_condition=torch.zeros(1, 1, 1),
        first_stage=False,
        last_stage=False,
        condition_mode=ConditionPropagationMode.FORWARD_PER_PATCH,
        forward_patch_fn=lambda _work, _time, image, condition: (
            image + 1,
            seen_conditions.append(condition.clone()) or condition + 1,
        ),
        update_last_patch_fn=lambda _work, _time, image, _previous: image,
        finalize_fn=lambda: "done",
        model_name="test",
    )

    assert schedule.run(
        timesteps=(1,),
        output_cache=PipeFusionStageOutputCache((1,), 2),
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=1,
            num_patches=2,
        ),
        advance_patch=lambda: None,
    ) == "done"
    assert torch.equal(seen_conditions[0], torch.tensor([[[20.0]]]))
    assert torch.equal(seen_conditions[1], torch.tensor([[[99.0]]]))
    assert [tensor.shape[1] for _, _, tensor in group.sent] == [2, 2]


def test_last_stage_reuses_patch_zero_condition_from_image_only_payloads():
    layout = PipeFusionPatchLayout(
        split_dim=1,
        split_sizes=(1, 1),
        token_counts=(1, 1),
    )
    group = _FakeGroup(
        [
            torch.tensor([[[10.0], [20.0]]]),
            torch.tensor([[[11.0]]]),
        ]
    )
    seen_conditions = []
    schedule = JointImageTextPatchSchedule(
        patch_latents=[None, None],
        layout=layout,
        initial_condition=torch.zeros(1, 1, 1),
        first_stage=False,
        last_stage=True,
        condition_mode=ConditionPropagationMode.REUSE_FIRST_PATCH,
        forward_patch_fn=lambda _work, _time, image, condition: (
            seen_conditions.append(condition.clone()) or image + 1,
            None,
        ),
        update_last_patch_fn=lambda _work, _time, image, _previous: image,
        finalize_fn=lambda: "done",
        model_name="test",
    )

    assert schedule.run(
        timesteps=(1,),
        output_cache=PipeFusionStageOutputCache((1,), 2),
        transport=PipeFusionTransport(
            group,
            first_stage=False,
            num_steps=1,
            num_patches=2,
        ),
        advance_patch=lambda: None,
    ) == "done"
    assert torch.equal(seen_conditions[0], torch.tensor([[[20.0]]]))
    assert torch.equal(seen_conditions[1], torch.tensor([[[20.0]]]))


def test_transport_rejects_duplicate_receive_queuing():
    transport = PipeFusionTransport(
        _FakeGroup([]),
        first_stage=True,
        num_steps=1,
        num_patches=1,
    )

    transport.queue_receives()

    with pytest.raises(RuntimeError, match="already been queued"):
        transport.queue_receives()


def test_transport_retains_async_send_until_drain():
    group = _FakeGroup([])
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=1,
        num_patches=1,
    )

    transport.send(torch.tensor([1.0], dtype=torch.float32), patch_index=0)
    transport.wait_sends()

    assert len(group.sent) == 1
    assert group.sent[0][2].dtype == torch.float16
    assert group.send_work[0].waited
    assert not group.blocking_sent


def test_transport_installs_fixed_payload_shapes():
    group = _FakeGroup([])
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=1,
        num_patches=1,
    )

    transport.configure_fixed_payload_shapes(
        send_shapes=[torch.Size((1, 3))],
        recv_shapes=[torch.Size((1, 2))],
    )

    assert group.send_shape["latent"] == {0: torch.Size((1, 3))}
    assert group.recv_buffer["latent"][0].dtype == torch.float16
    assert "latent" in group.fixed_payload_names


def test_transport_restores_preexisting_buffers_after_fixed_shape_run():
    group = _FakeGroup([])
    group.send_shape["latent"] = {-1: torch.Size((1, 4))}
    group.recv_shape["latent"] = {-1: torch.Size((1, 4))}
    group.recv_buffer["latent"] = {-1: torch.zeros(1, 4)}
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=1,
        num_patches=1,
    )

    transport._save_payload_buffer_state()
    group.send_shape["latent"] = {0: torch.Size((1, 2))}
    group.recv_shape["latent"] = {0: torch.Size((1, 2))}
    group.recv_buffer["latent"] = {0: torch.zeros(1, 2)}
    group.fixed_payload_names.add("latent")
    transport.wait_sends()

    assert group.send_shape["latent"] == {-1: torch.Size((1, 4))}
    assert group.recv_shape["latent"] == {-1: torch.Size((1, 4))}
    assert group.recv_buffer["latent"][-1].shape == (1, 4)
    assert "latent" not in group.fixed_payload_names


def test_transport_retains_contiguous_wire_tensor_for_a_view():
    group = _FakeGroup([])
    transport = PipeFusionTransport(
        group,
        first_stage=False,
        num_steps=1,
        num_patches=1,
    )
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).transpose(0, 1)

    transport.send(source, patch_index=0)

    assert transport._send_tensors[0].is_contiguous()
    assert torch.equal(transport._send_tensors[0], source.to(torch.float16))


@pytest.mark.parametrize(
    ("step_index", "expected"),
    [
        (0, False),
        (1, False),
        (2, True),
        (3, False),
        (4, True),
    ],
)
def test_async_progress_ticks_on_scheduler_boundaries_and_final_step(
    step_index,
    expected,
):
    assert (
        pipefusion_should_update_progress(
            step_index,
            num_async_steps=5,
            pipeline_warmup_steps=1,
            num_warmup_steps=2,
            scheduler_order=2,
        )
        is expected
    )
