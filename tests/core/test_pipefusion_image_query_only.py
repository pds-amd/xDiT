import copy

import pytest
import torch
import torch.nn.functional as F

diffusers = pytest.importorskip("diffusers")

import xfuser.model_executor.layers.attention_processor as attention_processor
import xfuser.model_executor.models.transformers.transformer_flux as flux_module
import xfuser.model_executor.models.transformers.transformer_sd3 as sd3_module
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)
from xfuser.model_executor.layers.attention_processor import (
    xFuserJointAttnProcessor2_0,
)
from xfuser.model_executor.models.transformers.transformer_flux import (
    xFuserFluxAttnProcessor,
)


class _RuntimeState:
    patch_mode = True
    pipeline_patch_idx = 1
    num_pipeline_patch = 2
    max_condition_sequence_length = 3
    split_text_embed_in_sp = False


class _RecordingCache:
    def __init__(self):
        self.token_counts = []

    def update_and_get_kv_cache(
        self, *, new_kv, layer, slice_dim, layer_type
    ):
        self.token_counts.append((new_kv[0].shape[slice_dim], slice_dim))
        return new_kv


def _usp(
    query,
    key,
    value,
    joint_query=None,
    joint_key=None,
    joint_value=None,
    joint_strategy=None,
    **kwargs,
):
    if joint_strategy == "front":
        if joint_query is not None:
            query = torch.cat([joint_query, query], dim=2)
        if joint_key is not None:
            key = torch.cat([joint_key, key], dim=2)
            value = torch.cat([joint_value, value], dim=2)
    elif joint_strategy == "rear":
        if joint_query is not None:
            query = torch.cat([query, joint_query], dim=2)
        if joint_key is not None:
            key = torch.cat([key, joint_key], dim=2)
            value = torch.cat([value, joint_value], dim=2)
    return F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )


@pytest.fixture
def pipefusion_runtime(monkeypatch):
    state = _RuntimeState()
    cache = _RecordingCache()
    for module in (flux_module, sd3_module):
        monkeypatch.setattr(module, "get_runtime_state", lambda: state)
        monkeypatch.setattr(
            module, "get_pipeline_parallel_world_size", lambda: 2
        )
        monkeypatch.setattr(
            module, "get_sequence_parallel_world_size", lambda: 1
        )
    monkeypatch.setattr(
        attention_processor, "get_runtime_state", lambda: state
    )
    monkeypatch.setattr(
        attention_processor, "get_sequence_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(attention_processor, "HAS_LONG_CTX_ATTN", False)
    monkeypatch.setattr(attention_processor, "USP", _usp)
    monkeypatch.setattr(attention_processor, "get_cache_manager", lambda: cache)
    monkeypatch.setattr(flux_module, "USP", _usp)
    monkeypatch.setattr(flux_module, "get_cache_manager", lambda: cache)
    monkeypatch.setattr(flux_module, "_HAS_FLYDSL", False)
    return state, cache


def _flux_rope(text_tokens, image_tokens, head_dim):
    shape = (text_tokens + image_tokens, head_dim)
    return torch.ones(shape), torch.zeros(shape)


@pytest.mark.parametrize(
    "block_factory",
    [
        lambda: FluxTransformerBlock(
            dim=8, num_attention_heads=2, attention_head_dim=4
        ),
        lambda: FluxSingleTransformerBlock(
            dim=8, num_attention_heads=2, attention_head_dim=4
        ),
    ],
    ids=["double-stream", "single-stream"],
)
def test_flux_image_query_only_matches_discarded_text_reference(
    monkeypatch, pipefusion_runtime, block_factory
):
    torch.manual_seed(0)
    reference = block_factory().eval()
    optimized = copy.deepcopy(reference).eval()
    reference.attn.set_processor(xFuserFluxAttnProcessor())
    optimized.attn.set_processor(xFuserFluxAttnProcessor())

    text = torch.randn(1, 3, 8)
    image = torch.randn(1, 2, 8)
    temb = torch.randn(1, 8)
    rope = _flux_rope(3, 2, 4)

    reference_text, reference_image = reference(
        hidden_states=image,
        encoder_hidden_states=text,
        temb=temb,
        image_rotary_emb=rope,
    )
    optimized_text, optimized_image = flux_module._flux_block_forward(
        optimized, image, text, temb, rope, None
    )

    assert optimized_text is text
    assert optimized_text.shape == reference_text.shape
    assert optimized_image.shape == reference_image.shape
    torch.testing.assert_close(optimized_image, reference_image)


def test_flux_image_query_only_caches_image_kv_and_skips_text_work(
    monkeypatch, pipefusion_runtime
):
    _, cache = pipefusion_runtime
    block = FluxTransformerBlock(
        dim=8, num_attention_heads=2, attention_head_dim=4
    ).eval()
    block.attn.set_processor(xFuserFluxAttnProcessor())

    calls = {"text_q": 0, "text_out": 0, "text_ff": 0}
    block.attn.add_q_proj.register_forward_hook(
        lambda *args: calls.__setitem__("text_q", calls["text_q"] + 1)
    )
    block.attn.to_add_out.register_forward_hook(
        lambda *args: calls.__setitem__("text_out", calls["text_out"] + 1)
    )
    block.ff_context.register_forward_hook(
        lambda *args: calls.__setitem__("text_ff", calls["text_ff"] + 1)
    )

    text = torch.randn(1, 3, 8)
    image = torch.randn(1, 2, 8)
    flux_module._flux_block_forward(
        block,
        image,
        text,
        torch.randn(1, 8),
        _flux_rope(3, 2, 4),
        None,
    )

    assert calls == {"text_q": 0, "text_out": 0, "text_ff": 0}
    assert cache.token_counts == [(2, 1)]


def test_flux_single_stream_keeps_wrapped_projection_calls_image_scoped(
    pipefusion_runtime,
):
    block = FluxSingleTransformerBlock(
        dim=8, num_attention_heads=2, attention_head_dim=4
    ).eval()
    block.attn.set_processor(xFuserFluxAttnProcessor())
    token_counts = {
        "q": [],
        "k": [],
        "v": [],
        "mlp": [],
        "out": [],
    }
    for name, module in (
        ("q", block.attn.to_q),
        ("k", block.attn.to_k),
        ("v", block.attn.to_v),
        ("mlp", block.proj_mlp),
        ("out", block.proj_out),
    ):
        module.register_forward_pre_hook(
            lambda module, args, name=name: token_counts[name].append(
                args[0].shape[1]
            )
        )

    flux_module._flux_block_forward(
        block,
        torch.randn(1, 2, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 8),
        _flux_rope(3, 2, 4),
        None,
    )

    assert token_counts == {
        "q": [2],
        "k": [2, 3],
        "v": [2, 3],
        "mlp": [2],
        "out": [2],
    }


def test_flux_safe_fallbacks_use_original_block(monkeypatch, pipefusion_runtime):
    state, _ = pipefusion_runtime
    block = FluxTransformerBlock(
        dim=8, num_attention_heads=2, attention_head_dim=4
    ).eval()
    image = torch.randn(1, 2, 8)
    text = torch.randn(1, 3, 8)
    rope = _flux_rope(3, 2, 4)

    state.pipeline_patch_idx = 0
    assert not flux_module._can_use_flux_image_query_only(
        block, image, text, rope, None
    )
    state.pipeline_patch_idx = 1
    block.train()
    assert not flux_module._can_use_flux_image_query_only(
        block, image, text, rope, None
    )
    block.eval()
    block.attn.fused_projections = True
    assert not flux_module._can_use_flux_image_query_only(
        block, image, text, rope, None
    )
    block.attn.fused_projections = False
    state.runtime_config = type(
        "RuntimeConfig",
        (),
        {"disable_pipefusion_image_query_only": True},
    )()
    assert not flux_module._can_use_flux_image_query_only(
        block, image, text, rope, None
    )


@pytest.mark.parametrize("context_pre_only", [False, True])
def test_sd3_image_query_only_matches_discarded_text_reference(
    pipefusion_runtime, context_pre_only
):
    torch.manual_seed(1)
    reference = JointTransformerBlock(
        dim=8,
        num_attention_heads=2,
        attention_head_dim=4,
        context_pre_only=context_pre_only,
    ).eval()
    optimized = copy.deepcopy(reference).eval()
    reference.attn.set_processor(xFuserJointAttnProcessor2_0())
    optimized.attn.set_processor(xFuserJointAttnProcessor2_0())

    text = torch.randn(1, 3, 8)
    image = torch.randn(1, 2, 8)
    temb = torch.randn(1, 8)
    reference_text, reference_image = reference(
        hidden_states=image,
        encoder_hidden_states=text,
        temb=temb,
    )
    optimized_text, optimized_image = sd3_module._sd3_block_forward(
        optimized, image, text, temb, None
    )

    assert optimized_text is (None if context_pre_only else text)
    if reference_text is not None:
        assert optimized_text.shape == reference_text.shape
    assert optimized_image.shape == reference_image.shape
    torch.testing.assert_close(optimized_image, reference_image)


def test_sd3_image_query_only_preserves_text_kv_and_skips_text_outputs(
    pipefusion_runtime,
):
    _, cache = pipefusion_runtime
    block = JointTransformerBlock(
        dim=8, num_attention_heads=2, attention_head_dim=4
    ).eval()
    block.attn.set_processor(xFuserJointAttnProcessor2_0())
    calls = {"text_q": 0, "text_k": 0, "text_v": 0, "text_ff": 0}
    for name, module in (
        ("text_q", block.attn.add_q_proj),
        ("text_k", block.attn.add_k_proj),
        ("text_v", block.attn.add_v_proj),
        ("text_ff", block.ff_context),
    ):
        module.register_forward_hook(
            lambda *args, name=name: calls.__setitem__(
                name, calls[name] + 1
            )
        )

    sd3_module._sd3_block_forward(
        block,
        torch.randn(1, 2, 8),
        torch.randn(1, 3, 8),
        torch.randn(1, 8),
        None,
    )

    assert calls == {"text_q": 0, "text_k": 1, "text_v": 1, "text_ff": 0}
    assert cache.token_counts == [(2, 1)]


def test_attention_processor_registry_prefers_exact_subclass_match():
    registry = attention_processor.xFuserAttentionProcessorRegister

    class BaseProcessor:
        pass

    class SpecializedProcessor(BaseProcessor):
        pass

    original = registry._XFUSER_ATTENTION_PROCESSOR_MAPPING
    try:
        registry._XFUSER_ATTENTION_PROCESSOR_MAPPING = {
            BaseProcessor: "base",
            SpecializedProcessor: "specialized",
        }
        assert (
            registry.get_processor(SpecializedProcessor())
            == "specialized"
        )
    finally:
        registry._XFUSER_ATTENTION_PROCESSOR_MAPPING = original
