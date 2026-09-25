import dataclasses
import inspect
import json
import types

import torch

import xfuser.core.distributed as distributed
import xfuser.model_executor.cache.adapters.cache_dit as cache_dit_adapter
from xfuser.model_executor.cache.adapters.cache_dit import (
    _build_config,
    _build_scm_mask,
    _install_pipefusion_patch_contexts,
    apply_cache_dit_cache,
)
from xfuser.model_executor.cache.presets import (
    CacheDitAdapterConfig,
    DBCachePreset,
)
from xfuser.model_executor.pipefusion import supports_pipefusion_stage_cache
from xfuser.model_executor.pipelines.pipeline_flux import (
    _pipefusion_scm_compute_mask,
)
from xfuser.model_executor.pipelines.pipeline_flux2 import (
    xFuserFlux2PipelineBase,
)


def test_dbcache_automatically_installs_pipefusion_patch_contexts(monkeypatch):
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 2)
    monkeypatch.setattr(cache_dit_adapter, "_install_sp_can_cache_sync", lambda: None)
    installed = []
    monkeypatch.setattr(
        cache_dit_adapter,
        "_install_pipefusion_patch_contexts",
        lambda transformer: installed.append(transformer),
    )
    class _BlockAdapter:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        cache_dit_adapter,
        "_import_cache_dit",
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


def test_pipefusion_global_scm_mask_isolates_cache_steps():
    assert _pipefusion_scm_compute_mask(
        (0, 0, 0, 1, 0, 0, 0, 1),
        8,
    ) == (1, 0, 1, 1, 0, 1, 0, 1)


def test_pipefusion_scm_policy_alternates_in_middle_window():
    mask = _build_scm_mask("pipefusion", 25)

    assert [index for index, value in enumerate(mask) if value == 0] == [
        8, 10, 12, 14, 16, 18, 20
    ]
    assert _pipefusion_scm_compute_mask(mask, 25) == tuple(mask)


def test_flux2_pipeline_supports_global_scm_output_cache():
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

    _install_pipefusion_patch_contexts(transformer)
    _install_pipefusion_patch_contexts(transformer)
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


def test_pipefusion_global_scm_cache_bypasses_transformer_cache(monkeypatch):
    monkeypatch.setattr(distributed, "get_pipeline_parallel_world_size", lambda: 4)
    monkeypatch.setattr(distributed, "get_sequence_parallel_world_size", lambda: 1)
    sync_install_calls = []
    monkeypatch.setattr(
        cache_dit_adapter,
        "_install_sp_can_cache_sync",
        lambda: sync_install_calls.append(True),
    )
    enable_calls = []

    class _BlockAdapter:
        def __init__(self, **kwargs):
            pass

    monkeypatch.setattr(
        cache_dit_adapter,
        "_import_cache_dit",
        lambda: (
            lambda *args, **kwargs: enable_calls.append((args, kwargs)),
            _DBCacheConfig,
            _BlockAdapter,
            type("_Patterns", (), {"P1": object()}),
        ),
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
        preset_kwargs={
            "Fn_compute_blocks": 1,
            "Bn_compute_blocks": 0,
            "residual_diff_threshold": 0.1,
            "max_warmup_steps": 1,
            "max_cached_steps": -1,
            "steps_computation_mask": [1, 0, 1, 1],
        },
        adapter_config=CacheDitAdapterConfig(
            blocks=(("blocks", "P1"),),
            pipefusion_global_scm_cache=True,
        ),
    )

    assert result is transformer
    assert pipe._xdit_pipefusion_scm_mask == (1, 0, 1, 1)
    assert enable_calls == []
    assert sync_install_calls == []


def test_pipefusion_global_scm_uses_static_policy_without_mutating_preset(
    monkeypatch,
):
    monkeypatch.setattr(
        distributed,
        "get_pipeline_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(
        cache_dit_adapter,
        "_import_cache_dit",
        lambda: (object(), object(), object(), object()),
    )
    captured = {}

    def build_config(**kwargs):
        captured["preset"] = kwargs["preset_kwargs"]
        return types.SimpleNamespace(
            steps_computation_mask=[1, 0, 1, 1],
        ), None

    monkeypatch.setattr(cache_dit_adapter, "_build_config", build_config)
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
            pipefusion_global_scm_cache=True,
        ),
    )

    effective = captured["preset"]
    assert effective.scm_policy == "pipefusion"
    assert effective.steps_computation_policy == "static"
    assert effective.enable_taylorseer is False
    assert preset.scm_policy == "ultra"
    assert preset.steps_computation_policy == "dynamic"


def test_yaml_overrides_select_plain_first_block_cache():
    overrides = json.dumps(
        {
            "Fn_compute_blocks": 1,
            "scm_policy": None,
            "enable_taylorseer": False,
        }
    )
    config, calibrator = _build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=2,
            residual_diff_threshold=0.16,
            scm_policy="ultra",
        ),
        cache_config_json=overrides,
        enable_separate_cfg=False,
        DBCacheConfig=_DBCacheConfig,
    )

    assert config.Fn_compute_blocks == 1
    assert calibrator is None


def test_static_scm_can_drive_pipefusion_cache_decisions():
    config, _ = _build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=1,
            scm_policy="ultra",
            steps_computation_policy="static",
            enable_taylorseer=False,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        DBCacheConfig=_DBCacheConfig,
    )

    assert config.Fn_compute_blocks == 1
    assert config.steps_computation_policy == "static"
    assert config.steps_computation_mask is not None


def test_dynamic_scm_retains_taylorseer_calibrator(monkeypatch):
    mask = object()
    calibrator = object()
    monkeypatch.setattr(
        cache_dit_adapter, "_build_scm_mask", lambda *args: mask
    )
    monkeypatch.setattr(
        cache_dit_adapter,
        "_build_calibrator_config",
        lambda *args: calibrator,
    )

    config, selected_calibrator = _build_config(
        num_steps=25,
        preset_kwargs=DBCachePreset(
            Fn_compute_blocks=2,
            scm_policy="ultra",
            steps_computation_policy="dynamic",
            enable_taylorseer=True,
        ),
        cache_config_json=None,
        enable_separate_cfg=False,
        DBCacheConfig=_DBCacheConfig,
    )

    assert config.steps_computation_policy == "dynamic"
    assert config.steps_computation_mask is mask
    assert selected_calibrator is calibrator
