import dataclasses
import inspect
import json
import types

import pytest
import torch

import xfuser.core.distributed as distributed
from xfuser.model_executor.cache.adapters.cache_dit import application
from xfuser.model_executor.cache.adapters.cache_dit import config as cache_config
from xfuser.model_executor.cache.adapters.cache_dit import context
from xfuser.model_executor.cache.adapters.cache_dit.application import (
    apply_cache_dit_cache,
)
from xfuser.model_executor.cache.adapters.cache_dit.config import build_config
from xfuser.model_executor.cache.adapters.cache_dit.context import (
    install_cache_decision_sync,
    install_pipefusion_patch_contexts,
)
from xfuser.model_executor.cache.presets import (
    CacheDitAdapterConfig,
    DBCachePreset,
    DBCacheSettings,
    PipeFusionCachePlan,
    PipeFusionCacheUnit,
    PipeFusionStaticMask,
    PipeFusionTopology,
)
from xfuser.model_executor.pipefusion import (
    build_pipefusion_static_mask,
    normalize_pipefusion_scm_mask,
    supports_pipefusion_stage_cache,
)
from xfuser.model_executor.pipelines.pipeline_flux2 import (
    xFuserFlux2PipelineBase,
)
from xfuser.model_executor.models.runner_models.flux import (
    xFuserFlux2Model,
    xFuserFluxModel,
)
from xfuser.model_executor.models.runner_models.stable_diffusion import (
    xFuserStableDiffusionModel,
)


def test_dbcache_automatically_installs_pipefusion_patch_contexts(monkeypatch):
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        application,
        "install_cache_decision_sync",
        lambda _transformer: None,
    )
    installed = []
    monkeypatch.setattr(
        application,
        "install_pipefusion_patch_contexts",
        lambda transformer: installed.append(transformer),
    )
    class _BlockAdapter:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        application,
        "import_cache_dit",
        lambda: (lambda *args, **kwargs: None, _DBCacheConfig, _BlockAdapter, object),
    )
    transformers = []
    for sequence_degree in (1, 2):
        monkeypatch.setattr(
            distributed,
            "get_sequence_parallel_world_size",
            lambda degree=sequence_degree: degree,
        )
        transformer = torch.nn.Linear(2, 2)
        transformers.append(transformer)
        assert apply_cache_dit_cache(
            transformer,
            num_steps=25,
            preset_kwargs=DBCachePreset(
                scm_policy=None,
                enable_taylorseer=False,
            ),
        ) is transformer

    assert installed == transformers


def test_pipefusion_static_mask_alternates_in_middle_window():
    mask = build_pipefusion_static_mask(
        PipeFusionStaticMask.ALTERNATING_MIDDLE,
        25,
    )

    assert [index for index, value in enumerate(mask) if value == 0] == [
        8, 10, 12, 14, 16, 18, 20
    ]
    assert normalize_pipefusion_scm_mask(mask, 25) == tuple(mask)


def test_pipefusion_wide_static_mask_keeps_cache_steps_isolated():
    mask = build_pipefusion_static_mask(
        PipeFusionStaticMask.WIDE_ALTERNATING_MIDDLE,
        25,
    )

    assert [index for index, value in enumerate(mask) if value == 0] == [
        6, 8, 10, 12, 14, 16, 18, 20, 22
    ]
    assert normalize_pipefusion_scm_mask(mask, 25) == tuple(mask)


def test_flux2_pipeline_supports_stage_payload_cache():
    assert supports_pipefusion_stage_cache(
        object.__new__(xFuserFlux2PipelineBase)
    )
    sync_parameters = inspect.signature(
        xFuserFlux2PipelineBase._sync_pipeline
    ).parameters
    async_parameters = inspect.signature(
        xFuserFlux2PipelineBase._async_pipeline
    ).parameters
    assert "computation_mask" not in sync_parameters
    assert "computation_mask" in async_parameters


def test_validated_models_declare_their_pipefusion_cache_plan():
    flux_plan = xFuserFluxModel.settings.step_cache_config[
        "dbcache"
    ].pipefusion_cache_plans[0]
    assert flux_plan == PipeFusionCachePlan.full_stage_output(
        min_inference_steps=25,
        max_inference_steps=25,
        topologies=(
            PipeFusionTopology(
                pp_degree=2,
                num_pipeline_patches=2,
                attn_layer_num_for_pp=(29, 28),
            ),
            PipeFusionTopology(
                pp_degree=4,
                num_pipeline_patches=4,
                attn_layer_num_for_pp=(14, 14, 14, 15),
            ),
        ),
    )
    assert (
        xFuserFlux2Model.settings.step_cache_config["dbcache"]
        .pipefusion_cache_plans
        == (PipeFusionCachePlan.block_local(),)
    )
    assert (
        xFuserStableDiffusionModel.settings.step_cache_config["dbcache"]
        .pipefusion_cache_plans
        == (PipeFusionCachePlan.block_local(),)
    )


def test_balanced_flux_split_stage_plans_remain_explicit_only():
    settings = xFuserFluxModel.settings.step_cache_config["dbcache"]
    kwargs = dict(
        pp_degree=2,
        num_pipeline_patches=2,
        num_inference_steps=25,
        attn_layer_num_for_pp=(28, 29),
    )

    assert (
        settings.resolve_pipefusion_cache_plan(**kwargs).unit
        is PipeFusionCacheUnit.BLOCK_LOCAL
    )
    assert (
        settings.resolve_pipefusion_cache_plan(
            **kwargs,
            requested_unit=PipeFusionCacheUnit.FULL_STAGE_OUTPUT,
        ).unit
        is PipeFusionCacheUnit.FULL_STAGE_OUTPUT
    )
    assert (
        settings.resolve_pipefusion_cache_plan(
            **kwargs,
            requested_unit=PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT,
        ).unit
        is PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT
    )
    kwargs["num_pipeline_patches"] = 4
    assert (
        settings.resolve_pipefusion_cache_plan(
            **kwargs,
            requested_unit=PipeFusionCacheUnit.FULL_STAGE_OUTPUT,
        ).unit
        is PipeFusionCacheUnit.FULL_STAGE_OUTPUT
    )


def test_cache_plan_falls_back_to_block_local_outside_validated_topology():
    settings = DBCacheSettings(
        adapter=CacheDitAdapterConfig(blocks=(("blocks", "P1"),)),
        pipefusion_cache_plans=(
            PipeFusionCachePlan.intermediate_stage_output(
                pp_degrees=(2,),
                pipeline_patch_counts=(2,),
                min_inference_steps=20,
            ),
        ),
    )

    assert (
        settings.resolve_pipefusion_cache_plan(
            pp_degree=2,
            num_pipeline_patches=2,
            num_inference_steps=25,
            attn_layer_num_for_pp=None,
        ).unit
        is PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT
    )
    assert (
        settings.resolve_pipefusion_cache_plan(
            pp_degree=4,
            num_pipeline_patches=4,
            num_inference_steps=25,
            attn_layer_num_for_pp=None,
        ).unit
        is PipeFusionCacheUnit.BLOCK_LOCAL
    )


def test_cache_plan_can_require_a_validated_stage_split():
    settings = DBCacheSettings(
        adapter=CacheDitAdapterConfig(blocks=(("blocks", "P1"),)),
        pipefusion_cache_plans=(
            PipeFusionCachePlan.full_stage_output(
                topologies=(
                    PipeFusionTopology(
                        pp_degree=2,
                        num_pipeline_patches=2,
                        attn_layer_num_for_pp=(29, 28),
                    ),
                ),
            ),
        ),
    )

    assert (
        settings.resolve_pipefusion_cache_plan(
            pp_degree=2,
            num_pipeline_patches=2,
            num_inference_steps=25,
            attn_layer_num_for_pp=(29, 28),
        ).unit
        is PipeFusionCacheUnit.FULL_STAGE_OUTPUT
    )
    assert (
        settings.resolve_pipefusion_cache_plan(
            pp_degree=2,
            num_pipeline_patches=2,
            num_inference_steps=25,
            attn_layer_num_for_pp=(19, 38),
        ).unit
        is PipeFusionCacheUnit.BLOCK_LOCAL
    )


def test_requested_stage_plan_is_model_and_topology_gated():
    topology = PipeFusionTopology(
        pp_degree=2,
        num_pipeline_patches=2,
        attn_layer_num_for_pp=(2, 2),
    )
    settings = DBCacheSettings(
        adapter=CacheDitAdapterConfig(blocks=(("blocks", "P1"),)),
        pipefusion_cache_plans=(
            PipeFusionCachePlan.full_stage_output(topologies=(topology,)),
            PipeFusionCachePlan.intermediate_stage_output(
                    topologies=(topology,),
                    auto_select=False,
            ),
            PipeFusionCachePlan.block_local(),
        ),
    )
    kwargs = {
        "pp_degree": 2,
        "num_pipeline_patches": 2,
        "num_inference_steps": 25,
        "attn_layer_num_for_pp": (2, 2),
    }

    assert (
        settings.resolve_pipefusion_cache_plan(**kwargs).unit
        is PipeFusionCacheUnit.FULL_STAGE_OUTPUT
    )
    assert (
        settings.resolve_pipefusion_cache_plan(
            **kwargs,
            requested_unit=PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT,
        ).unit
        is PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT
    )
    with pytest.raises(ValueError, match="not supported"):
        settings.resolve_pipefusion_cache_plan(
            **{**kwargs, "attn_layer_num_for_pp": (1, 3)},
            requested_unit=PipeFusionCacheUnit.INTERMEDIATE_STAGE_OUTPUT,
        )


def test_pipefusion_patch_contexts_follow_cache_dit_lifecycle(monkeypatch):
    class _Manager:
        def __init__(self):
            self.reset_names = []
            self.selected_name = None

        def reset_context(self, name, *args, **kwargs):
            self.reset_names.append(name)
            return name

        def set_context(self, name, *args, **kwargs):
            self.selected_name = name
            return name

    runtime_state = type(
        "_RuntimeState",
        (),
        {
            "num_pipeline_patch": 2,
            "patch_mode": True,
            "pipeline_patch_idx": 1,
        },
    )()
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: runtime_state)

    manager = _Manager()
    transformer = torch.nn.Linear(2, 2)
    transformer._context_manager = manager
    transformer._context_names = ["blocks", "single_blocks"]

    install_pipefusion_patch_contexts(transformer)
    install_pipefusion_patch_contexts(transformer)
    manager.reset_context("blocks", cache_config=object())
    manager.set_context("blocks")

    assert manager.reset_names == [
        "blocks",
        "blocks:pipefusion_patch_0",
        "blocks:pipefusion_patch_1",
    ]
    assert manager.selected_name == "blocks:pipefusion_patch_1"

    manager.reset_context("single_blocks", cache_config=object())
    assert manager.reset_names[-3:] == [
        "single_blocks",
        "single_blocks:pipefusion_patch_0",
        "single_blocks:pipefusion_patch_1",
    ]

    manager.reset_names.clear()
    runtime_state.num_pipeline_patch = 3
    runtime_state.pipeline_patch_idx = 2
    manager.reset_context("blocks", cache_config=object())
    manager.set_context("blocks")

    assert manager.reset_names == [
        "blocks",
        "blocks:pipefusion_patch_0",
        "blocks:pipefusion_patch_1",
        "blocks:pipefusion_patch_2",
    ]
    assert manager.selected_name == "blocks:pipefusion_patch_2"


def test_pipefusion_can_share_a_leader_cache_decision(monkeypatch):
    class _Manager:
        def __init__(self):
            self.calls = 0
            self.context = types.SimpleNamespace(
                name="blocks:pipefusion_patch_0"
            )

        def can_cache(self, *_args, **_kwargs):
            self.calls += 1
            return True

        def get_context(self):
            return self.context

        def get_current_step(self):
            return 0

    state = types.SimpleNamespace(patch_mode=True, pipeline_patch_idx=0)
    monkeypatch.setattr(
        distributed, "get_pipeline_parallel_world_size", lambda: 2
    )
    monkeypatch.setattr(
        distributed, "get_sequence_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(distributed, "get_runtime_state", lambda: state)
    monkeypatch.setattr(context.dist, "is_available", lambda: True)
    monkeypatch.setattr(context.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(context.dist, "get_world_size", lambda: 2)

    manager = _Manager()
    transformer = torch.nn.Linear(2, 2)
    transformer._context_manager = manager
    transformer._xdit_share_pipefusion_decisions = True
    install_cache_decision_sync(transformer)

    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    state.pipeline_patch_idx = 1
    manager.context.name = "blocks:pipefusion_patch_1"
    assert manager.can_cache(torch.ones(1), prefix="blocks_Fn") is True
    assert manager.calls == 1


@dataclasses.dataclass
class _DBCacheConfig:
    Fn_compute_blocks: int
    Bn_compute_blocks: int
    residual_diff_threshold: float
    max_warmup_steps: int
    max_cached_steps: int
    num_inference_steps: int
    enable_separate_cfg: bool = False
    steps_computation_mask: object = None
    steps_computation_policy: str = "dynamic"


def test_full_stage_cache_bypasses_transformer_cache(monkeypatch):
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 4)
    monkeypatch.setattr(
        application,
        "import_cache_dit",
        lambda: pytest.fail("stage-payload caching must not import cache-dit"),
    )
    transformer = torch.nn.Module()
    transformer.blocks = torch.nn.ModuleList([torch.nn.Identity()])
    pipe = types.SimpleNamespace(
        _pipefusion_async_computation_mask=lambda *args: None,
        _pipefusion_stage_output_cache=lambda *args: None,
    )

    result = apply_cache_dit_cache(
        transformer,
        num_steps=4,
        pipe=pipe,
        preset_kwargs=DBCachePreset(),
        adapter_config=CacheDitAdapterConfig(
            blocks=(("blocks", "P1"),),
        ),
        pipefusion_cache_plan=PipeFusionCachePlan.full_stage_output(),
    )

    assert result is transformer
    assert pipe._xdit_pipefusion_cache_plan == (
        PipeFusionCachePlan.full_stage_output()
    )


def test_pipefusion_stage_cache_does_not_mutate_block_cache_preset(
    monkeypatch,
):
    monkeypatch.setattr(
        distributed,
        "get_pipeline_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        application,
        "import_cache_dit",
        lambda: pytest.fail("stage-payload caching must not import cache-dit"),
    )
    preset = DBCachePreset(scm_policy="ultra")
    pipe = types.SimpleNamespace(
        _pipefusion_async_computation_mask=lambda *args: None,
        _pipefusion_stage_output_cache=lambda *args: None,
    )

    apply_cache_dit_cache(
        torch.nn.Module(),
        num_steps=4,
        pipe=pipe,
        preset_kwargs=preset,
        adapter_config=CacheDitAdapterConfig(
            blocks=(("blocks", "P1"),),
        ),
        pipefusion_cache_plan=PipeFusionCachePlan.full_stage_output(),
    )

    assert preset.scm_policy == "ultra"
    assert preset.steps_computation_policy == "dynamic"


def test_stage_cache_plan_rejects_user_schedule_overrides(monkeypatch):
    monkeypatch.setattr(
        distributed,
        "get_pipeline_parallel_world_size",
        lambda: 2,
    )
    with pytest.raises(ValueError, match="model-owned"):
        apply_cache_dit_cache(
            torch.nn.Module(),
            num_steps=4,
            preset_kwargs=DBCachePreset(),
            cache_config=json.dumps({"scm_policy": "ultra"}),
            adapter_config=CacheDitAdapterConfig(
                blocks=(("blocks", "P1"),),
            ),
            pipefusion_cache_plan=PipeFusionCachePlan.full_stage_output(),
        )


def test_yaml_overrides_select_plain_first_block_cache():
    overrides = json.dumps(
        {
            "Fn_compute_blocks": 1,
            "scm_policy": None,
            "enable_taylorseer": False,
        }
    )
    config, calibrator = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=2,
            residual_diff_threshold=0.16,
            scm_policy="ultra",
        ),
        cache_config_json=overrides,
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config.Fn_compute_blocks == 1
    assert calibrator is None


def test_pipefusion_decision_sharing_is_an_xdit_only_preset_flag():
    config, _ = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            enable_taylorseer=False,
            share_pipefusion_decisions=True,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config._xdit_share_pipefusion_decisions is True


def test_stage_plan_request_is_not_forwarded_to_cache_dit_config():
    config, _ = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(enable_taylorseer=False),
        cache_config_json=json.dumps(
            {"pipefusion_cache_plan": "intermediate_stage_output"}
        ),
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config.Fn_compute_blocks == DBCachePreset().Fn_compute_blocks


def test_stage_tail_plan_uses_static_mn_cache_and_bn_refinement(
    monkeypatch,
):
    monkeypatch.setattr(
        distributed,
        "get_runtime_state",
        lambda: types.SimpleNamespace(
            runtime_config=types.SimpleNamespace(warmup_steps=1)
        ),
    )
    config = _DBCacheConfig(
        Fn_compute_blocks=2,
        Bn_compute_blocks=0,
        residual_diff_threshold=0.16,
        max_warmup_steps=8,
        max_cached_steps=-1,
        num_inference_steps=25,
    )
    plan = PipeFusionCachePlan.intermediate_stage_output(
        tail_compute_blocks=1
    )

    application._configure_pipefusion_stage_tail_cache(config, plan, 25)

    assert config.Fn_compute_blocks == 1
    assert config.Bn_compute_blocks == 1
    assert config.steps_computation_policy == "static"
    assert config.steps_computation_mask[0] == 1
    assert 0 in config.steps_computation_mask


def test_static_scm_can_drive_pipefusion_cache_decisions():
    config, _ = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=1,
            scm_policy="ultra",
            steps_computation_policy="static",
            enable_taylorseer=False,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config.Fn_compute_blocks == 1
    assert config.steps_computation_policy == "static"
    assert config.steps_computation_mask is not None


def test_xdit_alternating_scm_policy_is_shared_with_cache_dit():
    config, _ = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            scm_policy="alternating_wide",
            steps_computation_policy="static",
            enable_taylorseer=False,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config.steps_computation_mask == [
        1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
    ]


def test_context_static_masks_are_kept_out_of_cache_dit_config():
    config, _ = build_config(
        num_steps=4,
        preset_kwargs=DBCachePreset(enable_taylorseer=False),
        cache_config_json=json.dumps(
            {
                "xdit_context_static_masks": {
                    "transformer_blocks": [1, 0, 1, 1],
                },
            }
        ),
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config._xdit_context_static_masks == {
        "transformer_blocks": (1, 0, 1, 1),
    }


def test_context_static_mask_matches_adapter_context_prefix():
    manager = types.SimpleNamespace(
        get_context=lambda: types.SimpleNamespace(
            name="single_transformer_blocks_123"
        ),
        get_current_step=lambda: 1,
    )

    assert context._context_static_reuse_decision(
        manager,
        {"single_transformer_blocks": (1, 0, 1)},
    ) is True


def test_auto_static_scm_freezes_calibration_decisions():
    step = [0]
    context_state = types.SimpleNamespace(
        name="transformer_blocks_123",
        cache_config=types.SimpleNamespace(num_inference_steps=3),
    )
    manager = types.SimpleNamespace(
        get_context=lambda: context_state,
        get_current_step=lambda: step[0],
    )
    state = {
        "decisions": {},
        "expected_contexts": {"transformer_blocks"},
        "masks": None,
        "current_decision": False,
    }

    for index, decision in enumerate((False, True, False)):
        step[0] = index
        state["current_decision"] = decision
        context._derive_auto_static_masks(state, manager)

    assert state["masks"] == {"transformer_blocks": (1, 0, 1)}


def test_auto_static_scm_cannot_override_explicit_context_masks():
    with pytest.raises(ValueError, match="cannot be combined"):
        build_config(
            num_steps=4,
            preset_kwargs=DBCachePreset(enable_taylorseer=False),
            cache_config_json=json.dumps(
                {
                    "xdit_auto_static_scm": True,
                    "xdit_context_static_masks": {"blocks": [1, 0, 1, 1]},
                }
            ),
            enable_separate_cfg=False,
            dbcache_config_type=_DBCacheConfig,
        )


def test_dynamic_scm_retains_taylorseer_calibrator(monkeypatch):
    mask = object()
    calibrator = object()
    monkeypatch.setattr(
        cache_config, "build_scm_mask", lambda *args: mask
    )
    monkeypatch.setattr(
        cache_config,
        "build_calibrator_config",
        lambda *args: calibrator,
    )

    config, selected_calibrator = build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=2,
            scm_policy="ultra",
            steps_computation_policy="dynamic",
            enable_taylorseer=True,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        dbcache_config_type=_DBCacheConfig,
    )

    assert config.steps_computation_policy == "dynamic"
    assert config.steps_computation_mask is mask
    assert selected_calibrator is calibrator
