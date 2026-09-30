import pytest
import torch

import xfuser.model_executor.pipelines.base_pipeline as base_pipeline
from xfuser.model_executor.pipelines.base_pipeline import (
    xFuserPipelineBaseWrapper,
)
from xfuser.model_executor.pipefusion import (
    PipeFusionStageOutputCache,
    normalize_pipefusion_scm_mask,
    pipefusion_async_computation_mask,
)
from xfuser.model_executor.cache.presets import (
    PipeFusionCacheDecision,
    PipeFusionCachePlan,
    PipeFusionCacheUnit,
    PipeFusionStaticMask,
)

def test_normalize_scm_isolates_cache_steps_and_computes_edges():
    assert normalize_pipefusion_scm_mask(
        (0, 0, 0, 1, 0, 0, 0, 1),
        8,
    ) == (1, 0, 1, 1, 0, 1, 0, 1)


def test_async_schedule_slices_warmup_and_forces_fresh_predecessor():
    pipeline = type(
        "_Pipeline",
        (),
        {"_xdit_pipefusion_cache_plan": PipeFusionCachePlan.full_stage_output()},
    )()

    assert pipefusion_async_computation_mask(
        pipeline, 5, 1, final_stage=True
    ) == (1, 1, 0, 1)


def test_pipefusion_policy_rebuilds_mask_for_runtime_step_count():
    pipeline = type(
        "_Pipeline",
        (),
        {"_xdit_pipefusion_cache_plan": PipeFusionCachePlan.full_stage_output()},
    )()

    mask = pipefusion_async_computation_mask(
        pipeline, 28, 1, final_stage=True
    )

    assert len(mask) == 27
    assert mask[0] == 1
    assert 0 in mask


def test_pipeline_without_scm_uses_all_compute_schedule():
    assert pipefusion_async_computation_mask(
        object(), 5, 2, final_stage=True
    ) == (1, 1, 1)


def test_intermediate_stage_plan_keeps_final_prediction_stage_fresh():
    pipeline = type(
        "_Pipeline",
        (),
        {
            "_xdit_pipefusion_cache_plan": (
                PipeFusionCachePlan.intermediate_stage_output()
            )
        },
    )()

    assert 0 in pipefusion_async_computation_mask(
        pipeline, 25, 1, final_stage=False
    )
    assert pipefusion_async_computation_mask(
        pipeline, 25, 1, final_stage=True
    ) == (1,) * 24


def test_full_stage_output_requires_explicit_final_stage_opt_in():
    with pytest.raises(ValueError, match="explicit final-stage"):
        PipeFusionCachePlan(
            unit=PipeFusionCacheUnit.FULL_STAGE_OUTPUT,
            decision=PipeFusionCacheDecision.STATIC,
            static_mask=PipeFusionStaticMask.ALTERNATING_MIDDLE,
        )


def test_stage_output_cache_reuses_arbitrary_patch_payload():
    cache = PipeFusionStageOutputCache((1, 0), num_patches=2)
    calls = []

    first = cache.resolve(0, 1, lambda: ("stage", calls.append(True)))
    second = cache.resolve(1, 1, lambda: pytest.fail("must not recompute"))

    assert first == second == ("stage", None)
    assert calls == [True]


def test_stage_output_cache_rejects_missing_predecessor():
    cache = PipeFusionStageOutputCache((0,), num_patches=1)

    with pytest.raises(RuntimeError, match="no computed predecessor"):
        cache.resolve(0, 0, lambda: None)


def test_base_async_init_queues_per_patch_side_channels(monkeypatch):
    class _Pipeline(xFuserPipelineBaseWrapper):
        def __call__(self):
            raise NotImplementedError

    class _State:
        num_pipeline_patch = 2
        pp_patches_height = [1, 1]

        def set_patched_mode(self, patch_mode):
            self.patch_mode = patch_mode

    class _Group:
        def __init__(self):
            self.tasks = []

        def add_pipeline_recv_task(self, idx, name="latent"):
            self.tasks.append((name, idx))

    state = _State()
    group = _Group()
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_pipeline, "get_pp_group", lambda: group)
    monkeypatch.setattr(base_pipeline, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(base_pipeline, "is_pipeline_last_stage", lambda: False)
    pipeline = object.__new__(_Pipeline)

    pipeline._init_async_pipeline(
        num_timesteps=1,
        latents=torch.zeros(1),
        num_pipeline_warmup_steps=0,
        per_patch_recv_segments=("encoder_hidden_states",),
    )

    assert group.tasks == [
        ("encoder_hidden_states", 0),
        ("latent", 0),
        ("encoder_hidden_states", 1),
        ("latent", 1),
    ]


def test_base_async_init_can_delegate_receive_queuing_to_transport(monkeypatch):
    class _Pipeline(xFuserPipelineBaseWrapper):
        def __call__(self):
            raise NotImplementedError

    class _State:
        num_pipeline_patch = 1
        pp_patches_height = [1]

        def set_patched_mode(self, patch_mode):
            self.patch_mode = patch_mode

    class _Group:
        def add_pipeline_recv_task(self, idx, name="latent"):
            pytest.fail(f"unexpected receive task: {name}[{idx}]")

    state = _State()
    monkeypatch.setattr(base_pipeline, "get_runtime_state", lambda: state)
    monkeypatch.setattr(base_pipeline, "get_pp_group", _Group)
    monkeypatch.setattr(base_pipeline, "is_pipeline_first_stage", lambda: False)
    monkeypatch.setattr(base_pipeline, "is_pipeline_last_stage", lambda: False)

    pipeline = object.__new__(_Pipeline)
    patch_latents = pipeline._init_async_pipeline(
        num_timesteps=1,
        latents=torch.zeros(1),
        num_pipeline_warmup_steps=0,
        queue_receives=False,
    )

    assert patch_latents == [None]
