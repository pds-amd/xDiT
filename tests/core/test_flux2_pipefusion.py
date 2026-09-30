from types import SimpleNamespace

import torch

import xfuser.model_executor.pipelines.pipeline_flux2 as pipeline_flux2
import xfuser.model_executor.models.runner_models.flux as runner_flux
import xfuser.model_executor.models.transformers.transformer_flux as transformer_flux
import xfuser.model_executor.models.transformers.transformer_flux2 as transformer_flux2


def test_flux2_reference_pipefusion_uses_dynamic_compile_only_when_needed():
    model = object.__new__(runner_flux.xFuserFlux2Model)
    model.config = SimpleNamespace(pipefusion_parallel_degree=2)

    assert model._get_compile_dynamic({"input_images": ["reference.png"]}) is True
    assert model._get_compile_dynamic({"input_images": []}) is False

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


def test_flux2_fp8_attention_override_is_limited_to_sync_pipefusion_warmup(
    monkeypatch,
):
    state = SimpleNamespace(
        attention_backend=transformer_flux2.AttentionBackendType.AITER_FLYDSL_FP8,
        num_pipeline_patch=2,
        patch_mode=False,
    )
    monkeypatch.setattr(transformer_flux2, "get_runtime_state", lambda: state)
    monkeypatch.setattr(
        transformer_flux2,
        "get_sequence_parallel_world_size",
        lambda: 1,
    )

    assert (
        transformer_flux2._flux2_pipefusion_attention_backend()
        is transformer_flux2.AttentionBackendType.AITER_FLYDSL
    )

    state.patch_mode = True
    assert transformer_flux2._flux2_pipefusion_attention_backend() is None


def test_flux2_marks_only_pipefusion_patch_kv_as_rectangular_fp8(monkeypatch):
    state = SimpleNamespace(num_pipeline_patch=2, patch_mode=True)
    monkeypatch.setattr(transformer_flux2, "get_runtime_state", lambda: state)
    monkeypatch.setattr(
        transformer_flux2,
        "get_sequence_parallel_world_size",
        lambda: 1,
    )

    assert transformer_flux2._flux2_pipefusion_attention_kwargs() == {
        "pipefusion_rectangular_kv": True
    }

    state.patch_mode = False
    assert transformer_flux2._flux2_pipefusion_attention_kwargs() is None


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
