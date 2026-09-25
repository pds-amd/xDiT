"""Focused, dependency-free checks for FLUX.1-Kontext PipeFusion wiring."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

import xfuser.model_executor.models.transformers.transformer_flux as transformer_flux

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "xfuser/model_executor/models/runner_models/flux.py"
PIPELINE = ROOT / "xfuser/model_executor/pipelines/pipeline_flux_kontext.py"
PIPELINE_EXPORTS = ROOT / "xfuser/model_executor/pipelines/__init__.py"
PACKAGE_EXPORTS = ROOT / "xfuser/__init__.py"


def _class(path: Path, name: str) -> ast.ClassDef:
    module = ast.parse(path.read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _method(class_node: ast.ClassDef, name: str):
    return next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    )


def _calls(node, attribute: str):
    return [
        call
        for call in ast.walk(node)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == attribute
    ]


def test_kontext_runner_enables_pipefusion_and_stage_local_loading():
    runner = _class(RUNNER, "xFuserFluxKontextModel")
    capabilities = next(
        node.value
        for node in runner.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "capabilities"
            for target in node.targets
        )
    )
    capability_values = {
        keyword.arg: keyword.value.value
        for keyword in capabilities.keywords
        if isinstance(keyword.value, ast.Constant)
    }
    assert capability_values["pipefusion_parallel_degree"] is True
    assert capability_values["ulysses_degree"] is True
    assert capability_values["supports_step_caching"] is True

    load_model = _method(runner, "_load_model")
    assert _calls(load_model, "plan_pipefusion_components")
    assert _calls(load_model, "mark_pipeline_stage_blockwise")
    imported_names = {
        alias.name
        for node in ast.walk(load_model)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "FluxTransformer2DModel" in imported_names
    assert "xFuserFluxKontextPipeline" in imported_names


def test_kontext_pipeline_reuses_flux_schedule_and_partitions_conditioning():
    pipeline = _class(PIPELINE, "xFuserFluxKontextPipeline")
    assert any(
        isinstance(base, ast.Name) and base.id == "xFuserFluxPipeline"
        for base in pipeline.bases
    )

    call = _method(pipeline, "__call__")
    assert _calls(call, "_sync_pipeline")
    async_calls = _calls(call, "_async_pipeline")
    assert async_calls
    assert all(
        any(
            keyword.arg == "computation_mask"
            for keyword in async_call.keywords
        )
        for async_call in async_calls
    )

    set_conditioning = _method(pipeline, "_set_image_conditioning")
    split_dims = {
        keyword.value.value
        for call_node in _calls(set_conditioning, "tensor_split")
        for keyword in call_node.keywords
        if keyword.arg == "dim" and isinstance(keyword.value, ast.Constant)
    }
    assert split_dims == {0, 1}


def test_kontext_backbone_keeps_reference_tokens_between_stages():
    pipeline = _class(PIPELINE, "xFuserFluxKontextPipeline")
    backbone = _method(pipeline, "_backbone_forward")

    source = ast.unparse(backbone)
    assert "is_pipeline_first_stage()" in source
    assert "torch.cat([hidden_states, image_latents], dim=1)" in source
    assert "torch.cat([latent_image_ids, image_ids], dim=0)" in source
    assert "is_pipeline_last_stage()" in source
    assert "noise_pred[:, :target_tokens]" in source


def test_kontext_reference_cache_supports_hybrid_sequence_layout(monkeypatch):
    state = SimpleNamespace(patch_mode=False)
    monkeypatch.setattr(transformer_flux, "get_runtime_state", lambda: state)
    attn = SimpleNamespace()
    full_key = torch.arange(24).reshape(1, 2, 3, 4)
    full_value = full_key + 100

    transformer_flux._update_flux_reference_kv_cache(
        attn,
        full_key,
        full_value,
        sequence_dim=2,
        reference_start=0,
        reference_end=3,
    )

    state.patch_mode = True
    patch_key = torch.full((1, 2, 1, 4), -1)
    patch_value = torch.full((1, 2, 1, 4), -2)
    cached_key, cached_value = transformer_flux._update_flux_reference_kv_cache(
        attn,
        patch_key,
        patch_value,
        sequence_dim=2,
        reference_start=1,
        reference_end=2,
    )

    assert torch.equal(cached_key[:, :, 1:2], patch_key)
    assert torch.equal(cached_value[:, :, 1:2], patch_value)
    assert torch.equal(cached_key[:, :, :1], full_key[:, :, :1])


def test_kontext_wrapper_is_optionally_exported():
    for path in (PIPELINE_EXPORTS, PACKAGE_EXPORTS):
        module = ast.parse(path.read_text())
        optional_symbols = {
            arg.value
            for node in ast.walk(module)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_optional"
            for arg in node.args[1:]
            if isinstance(arg, ast.Constant)
        }
        assert "xFuserFluxKontextPipeline" in optional_symbols
