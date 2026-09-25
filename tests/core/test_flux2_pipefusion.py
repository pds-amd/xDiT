import ast
from pathlib import Path
from types import SimpleNamespace

import torch

import xfuser.model_executor.pipelines.pipeline_flux2 as pipeline_flux2
import xfuser.model_executor.models.runner_models.flux as runner_flux
import xfuser.model_executor.models.transformers.transformer_flux as transformer_flux
import xfuser.model_executor.models.transformers.transformer_flux2 as transformer_flux2


def test_flux2_uses_atomic_image_and_text_payloads():
    path = (
        Path(__file__).resolve().parents[2]
        / "xfuser/model_executor/pipelines/pipeline_flux2.py"
    )
    module = ast.parse(path.read_text())
    pipeline = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "xFuserFlux2PipelineBase"
    )
    async_method = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "_async_pipeline"
    )
    source = ast.unparse(async_method)
    assert "CombinedTensorPayloadCodec" in source
    assert "PipeFusionTransport" in source
    assert "payload_codec.pack" in source
    assert "payload_codec.unpack" in source
    assert 'name="encoder_hidden_states"' not in source


def test_flux2_reference_only_patch_still_advances_scheduler():
    path = (
        Path(__file__).resolve().parents[2]
        / "xfuser/model_executor/pipelines/pipeline_flux2.py"
    )
    module = ast.parse(path.read_text())
    pipeline = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef)
        and node.name == "xFuserFlux2PipelineBase"
    )
    async_method = next(
        node
        for node in pipeline.body
        if isinstance(node, ast.FunctionDef) and node.name == "_async_pipeline"
    )
    generated_token_conditionals = [
        node
        for node in ast.walk(async_method)
        if isinstance(node, ast.If)
        and any(
            isinstance(part, ast.Name)
            and part.id == "generated_patch_tokens"
            for part in ast.walk(node.test)
        )
    ]

    assert not any(
        isinstance(part, ast.Call)
        and isinstance(part.func, ast.Attribute)
        and part.func.attr == "_scheduler_step"
        for conditional in generated_token_conditionals
        for part in ast.walk(conditional)
    )


def test_flux2_reference_tokens_resize_pipefusion_buffers(monkeypatch):
    resets = []
    state = SimpleNamespace(
        num_pipeline_patch=4,
        pp_patches_token_num=[1, 1, 1, 1],
        pp_patches_token_start_idx_local=[0, 1, 2, 3, 4],
        pp_patches_token_start_end_idx_global=[
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
        ],
        _reset_recv_buffer=lambda: resets.append(True),
    )
    monkeypatch.setattr(pipeline_flux2, "get_runtime_state", lambda: state)
    pipeline = object.__new__(pipeline_flux2.xFuserFlux2PipelineBase)

    layout = pipeline._reference_patch_layout(total_sequence_length=10)
    with layout.install(state):
        assert state.pp_patches_token_num == [3, 3, 2, 2]
        assert state.pp_patches_token_start_idx_local == [0, 3, 6, 8, 10]
        assert state.pp_patches_token_start_end_idx_global == [
            [0, 3],
            [3, 6],
            [6, 8],
            [8, 10],
        ]

    assert state.pp_patches_token_num == [1, 1, 1, 1]
    assert resets == [True, True]


def test_flux2_fp8_attention_fallback_only_applies_to_pure_pipefusion(
    monkeypatch,
):
    runtime = SimpleNamespace(attention_backend="AITER_FLYDSL_FP8")
    parallel = SimpleNamespace(sp_degree=1)
    engine = SimpleNamespace(
        runtime_config=runtime,
        parallel_config=parallel,
    )
    monkeypatch.setattr(runner_flux, "log", lambda *_args, **_kwargs: None)

    runner_flux._ensure_flux2_pipefusion_attention_quality(engine)
    assert runtime.attention_backend == "AITER_FLYDSL"

    runtime.attention_backend = "AITER_FLYDSL_FP8"
    parallel.sp_degree = 2
    runner_flux._ensure_flux2_pipefusion_attention_quality(engine)
    assert runtime.attention_backend == "AITER_FLYDSL_FP8"


def test_flux_hybrid_uses_sequence_parallel_kv_cache(monkeypatch):
    monkeypatch.setattr(transformer_flux.parallel_state, "_SP", object())
    for module in (transformer_flux, transformer_flux2):
        monkeypatch.setattr(module, "HAS_LONG_CTX_ATTN", True)
        monkeypatch.setattr(
            module, "get_sequence_parallel_world_size", lambda: 2
        )

    assert transformer_flux.xFuserFluxAttnProcessor().use_long_ctx_attn_kvcache
    assert transformer_flux2.xFuserFlux2AttnProcessor().use_long_ctx_attn_kvcache
    assert (
        transformer_flux2.xFuserFlux2ParallelSelfAttnProcessor()
        .use_long_ctx_attn_kvcache
    )


def test_flux_processor_can_initialize_before_parallel_groups(monkeypatch):
    monkeypatch.setattr(transformer_flux.parallel_state, "_SP", None)
    for module in (transformer_flux, transformer_flux2):
        monkeypatch.setattr(module, "HAS_LONG_CTX_ATTN", True)
        monkeypatch.setattr(
            module,
            "get_sequence_parallel_world_size",
            lambda: (_ for _ in ()).throw(AssertionError("SP is not initialized")),
        )

    assert not transformer_flux.xFuserFluxAttnProcessor().use_long_ctx_attn_kvcache
    assert not transformer_flux2.xFuserFlux2AttnProcessor().use_long_ctx_attn_kvcache
    assert not (
        transformer_flux2.xFuserFlux2ParallelSelfAttnProcessor()
        .use_long_ctx_attn_kvcache
    )
