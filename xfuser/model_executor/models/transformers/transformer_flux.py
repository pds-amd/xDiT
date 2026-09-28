import inspect
import torch
import torch.distributed
import torch.nn as nn
from typing import Optional, Dict, Any, Union, Tuple

#from diffusers.models.embeddings import PatchEmbed
from diffusers.models.transformers.transformer_flux import (
    _get_qkv_projections,
    FluxTransformer2DModel,
    FluxAttnProcessor,
    FluxAttention,
    FluxTransformerBlock,
    FluxSingleTransformerBlock,
)
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import apply_rotary_emb

from xfuser.core.distributed.parallel_state import (
    get_tensor_model_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.core.distributed import (
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_pipeline_parallel_world_size,
    get_cfg_group,
    get_sp_group,
)
from xfuser.core.distributed import parallel_state
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state

from xfuser.logger import init_logger
from xfuser.envs import PACKAGES_CHECKER
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionBaseWrapper,
    xFuserAttentionProcessorRegister
)
from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    flydsl_fused_qk_norm_rope,
    _HAS_FLYDSL,
)

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]


@xFuserLayerWrappersRegister.register(FluxAttention)
class xFuserFluxAttentionWrapper(xFuserAttentionBaseWrapper):
    def __init__(
        self,
        attention: FluxAttention,
    ):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks", "ip_hidden_states"}
        unused_kwargs = [k for k, _ in kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"joint_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        kwargs = {k: w for k, w in kwargs.items() if k in attn_parameters}
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, image_rotary_emb, **kwargs)

def _split_rotary_emb(image_rotary_emb, num_text_tokens: int, num_image_tokens: int):
    """Split a diffusers ``(cos, sin)`` pair at the text/image boundary.

    The joint FLUX stream is ``cat([text, image], dim=seq)`` and RoPE is a
    strictly per-token map, so slicing the tables and applying RoPE to each
    stream *before* the concat is bit-identical to applying it after.  Doing it
    before is what lets QK-RMSNorm and RoPE collapse into one kernel per stream.

    Returns ``(None, None)`` -- meaning "caller must apply RoPE post-concat" --
    unless the table row count is exactly ``num_text_tokens + num_image_tokens``,
    so any sequence-parallel slicing we do not understand stays on the safe path.
    """
    if image_rotary_emb is None:
        return None, None
    if not isinstance(image_rotary_emb, (tuple, list)) or len(image_rotary_emb) != 2:
        # unsupported (e.g. complex) layout -- caller applies RoPE post-concat
        return None, None
    cos, sin = image_rotary_emb
    if not isinstance(cos, torch.Tensor) or not isinstance(sin, torch.Tensor):
        return None, None
    if cos.dim() != 2 or cos.shape != sin.shape:
        return None, None
    if cos.shape[0] != num_text_tokens + num_image_tokens:
        return None, None
    txt = (cos[:num_text_tokens, :], sin[:num_text_tokens, :])
    img = (cos[num_text_tokens:, :], sin[num_text_tokens:, :])
    return txt, img


def _update_flux_reference_kv_cache(
    attn,
    reference_key,
    reference_value,
    *,
    sequence_dim,
    reference_start,
    reference_end,
):
    reference_kv = torch.cat([reference_key, reference_value], dim=-1)
    state = get_runtime_state()
    if not state.patch_mode:
        attn._xdit_pipefusion_reference_kv = reference_kv
    else:
        cached_reference_kv = getattr(
            attn, "_xdit_pipefusion_reference_kv", None
        )
        if cached_reference_kv is None:
            raise RuntimeError("Kontext reference KV cache was not initialized")
        reference_end = (
            reference_end
            if reference_end is not None
            else reference_start + reference_key.shape[sequence_dim]
        )
        cached_reference_kv.narrow(
            sequence_dim,
            reference_start,
            reference_end - reference_start,
        ).copy_(reference_kv)
        attn._xdit_pipefusion_reference_kv = cached_reference_kv
    return torch.chunk(attn._xdit_pipefusion_reference_kv, 2, dim=-1)


@xFuserAttentionProcessorRegister.register(FluxAttnProcessor)
class xFuserFluxAttnProcessor(FluxAttnProcessor):

    def __init__(self):
        super().__init__()

    @property
    def use_long_ctx_attn_kvcache(self):
        return (
            HAS_LONG_CTX_ATTN
            and parallel_state._SP is not None
            and get_sequence_parallel_world_size() > 1
        )

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        image_query_only: bool = False,
        num_txt_tokens: Optional[int] = None,
        pipefusion_reference_tokens: int = 0,
        pipefusion_reference_start: int = 0,
        pipefusion_reference_end: Optional[int] = None,
    ) -> torch.Tensor:
        if image_query_only:
            if attn.fused_projections:
                raise RuntimeError(
                    "FLUX image-query-only attention requires unfused projections."
                )
            if get_sequence_parallel_world_size() != 1:
                raise RuntimeError(
                    "FLUX image-query-only attention does not support sequence parallelism."
                )
            if encoder_hidden_states is None:
                if not num_txt_tokens:
                    raise ValueError(
                        "num_txt_tokens is required for single-stream image-query-only attention."
                    )
                text_hidden_states = hidden_states[:, :num_txt_tokens]
                image_hidden_states = hidden_states[:, num_txt_tokens:]
                query = attn.to_q(image_hidden_states)
                key = attn.to_k(image_hidden_states)
                value = attn.to_v(image_hidden_states)
                encoder_query = None
                encoder_key = attn.to_k(text_hidden_states)
                encoder_value = attn.to_v(text_hidden_states)
            else:
                query = attn.to_q(hidden_states)
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                encoder_query = None
                encoder_key = attn.add_k_proj(encoder_hidden_states)
                encoder_value = attn.add_v_proj(encoder_hidden_states)
        else:
            query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
                attn, hidden_states, encoder_hidden_states
            )
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        if image_query_only and attn.added_kv_proj_dim is None:
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

        if attn.added_kv_proj_dim is not None:
            if encoder_query is not None:
                encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            num_encoder_hidden_states_tokens = encoder_key.shape[1]
            num_query_tokens = query.shape[1]

            if _HAS_FLYDSL:
                # AITER/FlyDSL present: fuse QK-RMSNorm and RoPE into one kernel
                # per stream. RoPE is per-token, so applying it before the joint
                # concat over dim=1 is mathematically identical to applying it
                # after.
                txt_rope, img_rope = _split_rotary_emb(
                    image_rotary_emb, num_encoder_hidden_states_tokens, num_query_tokens
                )
                if image_query_only:
                    _, encoder_key = flydsl_fused_qk_norm_rope(
                        encoder_key,
                        encoder_key,
                        None,
                        attn.norm_added_k,
                        txt_rope,
                    )
                else:
                    encoder_query, encoder_key = flydsl_fused_qk_norm_rope(
                        encoder_query,
                        encoder_key,
                        attn.norm_added_q,
                        attn.norm_added_k,
                        txt_rope,
                    )
                query, key = flydsl_fused_qk_norm_rope(
                    query, key, attn.norm_q, attn.norm_k, img_rope
                )

                if not image_query_only:
                    query = torch.cat([encoder_query, query], dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)

                if txt_rope is None and image_rotary_emb is not None:
                    # tables could not be split -> apply RoPE on the joint stream
                    query, key = flydsl_fused_qk_norm_rope(
                        query, key, None, None, image_rotary_emb
                    )
            else:
                # No AITER: the original unfused diffusers path (norm then rope
                # on the joint stream), unchanged from before the fused kernel.
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                encoder_key = attn.norm_added_k(encoder_key)
                if not image_query_only:
                    encoder_query = attn.norm_added_q(encoder_query)

                if image_query_only:
                    txt_rope, img_rope = _split_rotary_emb(
                        image_rotary_emb,
                        num_encoder_hidden_states_tokens,
                        num_query_tokens,
                    )
                    query = apply_rotary_emb(query, img_rope, sequence_dim=1)
                    key = apply_rotary_emb(key, img_rope, sequence_dim=1)
                    encoder_key = apply_rotary_emb(
                        encoder_key, txt_rope, sequence_dim=1
                    )
                else:
                    query = torch.cat([encoder_query, query], dim=1)
                    if image_rotary_emb is not None:
                        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                key = torch.cat([encoder_key, key], dim=1)
                value = torch.cat([encoder_value, value], dim=1)
                if image_rotary_emb is not None and not image_query_only:
                    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        else:
            num_encoder_hidden_states_tokens = (
                num_txt_tokens
                if image_query_only
                else get_runtime_state().max_condition_sequence_length
            )
            num_query_tokens = query.shape[1]
            if not image_query_only:
                num_query_tokens -= num_encoder_hidden_states_tokens
            if _HAS_FLYDSL:
                if image_query_only:
                    txt_rope, img_rope = _split_rotary_emb(
                        image_rotary_emb,
                        num_encoder_hidden_states_tokens,
                        num_query_tokens,
                    )
                    query, key = flydsl_fused_qk_norm_rope(
                        query, key, attn.norm_q, attn.norm_k, img_rope
                    )
                    _, encoder_key = flydsl_fused_qk_norm_rope(
                        encoder_key,
                        encoder_key,
                        None,
                        attn.norm_k,
                        txt_rope,
                    )
                    key = torch.cat([encoder_key, key], dim=1)
                    value = torch.cat([encoder_value, value], dim=1)
                else:
                    query, key = flydsl_fused_qk_norm_rope(
                        query, key, attn.norm_q, attn.norm_k, image_rotary_emb
                    )
            else:
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                if image_query_only:
                    encoder_key = attn.norm_k(encoder_key)
                    txt_rope, img_rope = _split_rotary_emb(
                        image_rotary_emb,
                        num_encoder_hidden_states_tokens,
                        num_query_tokens,
                    )
                    query = apply_rotary_emb(query, img_rope, sequence_dim=1)
                    key = apply_rotary_emb(key, img_rope, sequence_dim=1)
                    encoder_key = apply_rotary_emb(
                        encoder_key, txt_rope, sequence_dim=1
                    )
                    key = torch.cat([encoder_key, key], dim=1)
                    value = torch.cat([encoder_value, value], dim=1)
                elif image_rotary_emb is not None:
                    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
                    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        distri_cache_updated = False
        if (
            get_runtime_state().num_pipeline_patch > 1
            and not self.use_long_ctx_attn_kvcache
        ):
            encoder_hidden_states_key_proj, key = key.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            encoder_hidden_states_value_proj, value = value.split(
                [num_encoder_hidden_states_tokens, num_query_tokens], dim=1
            )
            reference_key = reference_value = None
            if pipefusion_reference_tokens:
                target_tokens = key.shape[1] - pipefusion_reference_tokens
                key, reference_key = key.split(
                    [target_tokens, pipefusion_reference_tokens], dim=1
                )
                value, reference_value = value.split(
                    [target_tokens, pipefusion_reference_tokens], dim=1
                )
            key, value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[key, value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
            if reference_key is not None:
                reference_key, reference_value = (
                    _update_flux_reference_kv_cache(
                        attn,
                        reference_key,
                        reference_value,
                        sequence_dim=1,
                        reference_start=pipefusion_reference_start,
                        reference_end=pipefusion_reference_end,
                    )
                )
                key = torch.cat([key, reference_key], dim=1)
                value = torch.cat([value, reference_value], dim=1)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)
            distri_cache_updated = True

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        uses_pipeline_parallelism = get_runtime_state().num_pipeline_patch > 1
        if not uses_pipeline_parallelism:
            hidden_states = USP(
                query, key, value, combine_qkv_a2a=True, attn_layer=attn
            )
            hidden_states = hidden_states.transpose(1, 2)
        else:
            if image_query_only:
                hidden_states = USP(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                    combine_qkv_a2a=True,
                    attn_layer=None if distri_cache_updated else attn,
                    head_balance_layer=attn,
                )
                hidden_states = hidden_states.transpose(1, 2)
            elif get_runtime_state().split_text_embed_in_sp:
                encoder_hidden_states_query_proj = None
                encoder_hidden_states_key_proj = None
                encoder_hidden_states_value_proj = None
            else:
                num_query_tokens_q = query.shape[2] - num_encoder_hidden_states_tokens
                num_query_tokens_kv = key.shape[2] - num_encoder_hidden_states_tokens
                encoder_hidden_states_query_proj, query = query.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens_q], dim=2
                )
                encoder_hidden_states_key_proj, key = key.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens_kv], dim=2
                )
                encoder_hidden_states_value_proj, value = value.split(
                    [num_encoder_hidden_states_tokens, num_query_tokens_kv], dim=2
                )
                if (
                    self.use_long_ctx_attn_kvcache
                    and pipefusion_reference_tokens
                ):
                    target_tokens = key.shape[2] - pipefusion_reference_tokens
                    key, reference_key = key.split(
                        [target_tokens, pipefusion_reference_tokens], dim=2
                    )
                    value, reference_value = value.split(
                        [target_tokens, pipefusion_reference_tokens], dim=2
                    )
                    reference_key, reference_value = (
                        _update_flux_reference_kv_cache(
                            attn,
                            reference_key,
                            reference_value,
                            sequence_dim=2,
                            reference_start=pipefusion_reference_start,
                            reference_end=pipefusion_reference_end,
                        )
                    )
                    encoder_hidden_states_key_proj = torch.cat(
                        [encoder_hidden_states_key_proj, reference_key], dim=2
                    )
                    encoder_hidden_states_value_proj = torch.cat(
                        [encoder_hidden_states_value_proj, reference_value],
                        dim=2,
                    )
            if not image_query_only:
                hidden_states = USP(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=False,
                    combine_qkv_a2a=True,
                    joint_query=encoder_hidden_states_query_proj,
                    joint_key=encoder_hidden_states_key_proj,
                    joint_value=encoder_hidden_states_value_proj,
                    joint_strategy="front",
                    attn_layer=None if distri_cache_updated else attn,
                    head_balance_layer=attn,
                )
                hidden_states = hidden_states.transpose(1, 2)


        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None and not image_query_only:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if image_query_only and encoder_hidden_states is not None:
                hidden_states = attn.to_out[0](hidden_states)
                hidden_states = attn.to_out[1](hidden_states)
            return hidden_states




def flux_attn_modules(transformer) -> list[torch.nn.Module]:
    """Self-attention modules of a Flux1/Flux2 transformer, in block order.

    `single_transformer_blocks` is empty for pipefusion stages that hold none and for
    Flux2 configs without them.
    """
    return [
        block.attn
        for block in (
            *transformer.transformer_blocks,
            *transformer.single_transformer_blocks,
        )
    ]


def _can_use_flux_image_query_only(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor],
    image_rotary_emb,
    joint_attention_kwargs: Optional[Dict[str, Any]],
) -> bool:
    """Whether a later PipeFusion patch can safely omit FLUX text outputs."""
    state = get_runtime_state()
    image_only_disabled = getattr(
        getattr(state, "runtime_config", None),
        "disable_pipefusion_image_query_only",
        False,
    )
    if (
        image_only_disabled
        or block.training
        or not state.patch_mode
        or state.pipeline_patch_idx == 0
        or state.num_pipeline_patch <= 1
        or get_pipeline_parallel_world_size() <= 1
        or get_sequence_parallel_world_size() != 1
        or encoder_hidden_states is None
        or joint_attention_kwargs
    ):
        return False

    attn = block.attn
    if getattr(attn, "fused_projections", False):
        return False
    if not isinstance(image_rotary_emb, (tuple, list)) or len(image_rotary_emb) != 2:
        return False
    cos, sin = image_rotary_emb
    return (
        isinstance(cos, torch.Tensor)
        and isinstance(sin, torch.Tensor)
        and cos.dim() == 2
        and cos.shape == sin.shape
        and cos.shape[0]
        == encoder_hidden_states.shape[1] + hidden_states.shape[1]
    )


def _register_pipefusion_reference_buffers(module: nn.Module) -> None:
    for attention in flux_attn_modules(module):
        if "_xdit_pipefusion_reference_kv" not in attention._buffers:
            attention.register_buffer(
                "_xdit_pipefusion_reference_kv",
                None,
                persistent=False,
            )


def _flux_double_block_image_query_only(
    block: FluxTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.norm1(
        hidden_states, emb=temb
    )
    norm_encoder_hidden_states = block.norm1_context(
        encoder_hidden_states, emb=temb
    )[0]
    attn_output = block.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        image_query_only=True,
    )
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    hidden_states = (
        hidden_states
        + gate_mlp.unsqueeze(1) * block.ff(norm_hidden_states)
    )
    return encoder_hidden_states, hidden_states


def _flux_single_block_image_query_only(
    block: FluxSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_txt_tokens = encoder_hidden_states.shape[1]
    joint_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    norm_hidden_states, gate = block.norm(joint_hidden_states, emb=temb)
    image_norm_hidden_states = norm_hidden_states[:, num_txt_tokens:]
    mlp_hidden_states = block.act_mlp(block.proj_mlp(image_norm_hidden_states))
    attn_output = block.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        image_query_only=True,
        num_txt_tokens=num_txt_tokens,
    )
    update = block.proj_out(torch.cat([attn_output, mlp_hidden_states], dim=2))
    hidden_states = hidden_states + gate.unsqueeze(1) * update
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _flux_block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
    joint_attention_kwargs: Optional[Dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _can_use_flux_image_query_only(
        block,
        hidden_states,
        encoder_hidden_states,
        image_rotary_emb,
        joint_attention_kwargs,
    ):
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
    if isinstance(block, FluxTransformerBlock):
        return _flux_double_block_image_query_only(
            block, hidden_states, encoder_hidden_states, temb, image_rotary_emb
        )
    if isinstance(block, FluxSingleTransformerBlock):
        return _flux_single_block_image_query_only(
            block, hidden_states, encoder_hidden_states, temb, image_rotary_emb
        )
    return block(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        temb=temb,
        image_rotary_emb=image_rotary_emb,
        joint_attention_kwargs=joint_attention_kwargs,
    )


class xFuserFlux1Transformer2DWrapper(FluxTransformer2DModel):

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
        )

        for block in self.transformer_blocks + self.single_transformer_blocks:
            block.attn.processor = xFuserFluxAttnProcessor()
        _register_pipefusion_reference_buffers(self)
        register_fp8_comms_eligible_modules(self, flux_attn_modules(self))

    def pad_to_sp_divisible(self, tensor: torch.Tensor, padding_length: int, dim: int) -> torch.Tensor:
        padding =  torch.zeros(
            *tensor.shape[:dim], padding_length, *tensor.shape[dim + 1 :], dtype=tensor.dtype, device=tensor.device
        )
        tensor = torch.cat([tensor, padding], dim=dim)
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        **kwargs,
    ):

        sp_world_size = get_sequence_parallel_world_size()
        sequence_length = hidden_states.shape[1]
        padding_length = (sp_world_size - (sequence_length % sp_world_size)) % sp_world_size
        if padding_length > 0:
            hidden_states = self._pad_to_sp_divisible(hidden_states, padding_length, dim=1)
            img_ids = self._pad_to_sp_divisible(img_ids, padding_length, dim=0)
        assert (
            hidden_states.shape[0] % get_classifier_free_guidance_world_size() == 0
        ), f"Cannot split dim 0 of hidden_states ({hidden_states.shape[0]}) into {get_classifier_free_guidance_world_size()} parts."
        if encoder_hidden_states.shape[-2] % get_sequence_parallel_world_size() != 0:
            get_runtime_state().split_text_embed_in_sp = False
        else:
            get_runtime_state().split_text_embed_in_sp = True

        if (
            isinstance(timestep, torch.Tensor)
            and timestep.ndim != 0
            and timestep.shape[0] == hidden_states.shape[0]
        ):
            timestep = torch.chunk(
                timestep, get_classifier_free_guidance_world_size(), dim=0
            )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        hidden_states = torch.chunk(
            hidden_states, get_sequence_parallel_world_size(), dim=-2
        )[get_sequence_parallel_rank()]
        encoder_hidden_states = torch.chunk(
            encoder_hidden_states, get_classifier_free_guidance_world_size(), dim=0
        )[get_classifier_free_guidance_rank()]
        if get_runtime_state().split_text_embed_in_sp:
            encoder_hidden_states = torch.chunk(
                encoder_hidden_states, get_sequence_parallel_world_size(), dim=-2
            )[get_sequence_parallel_rank()]
        img_ids = torch.chunk(img_ids, get_sequence_parallel_world_size(), dim=-2)[
            get_sequence_parallel_rank()
        ]
        if get_runtime_state().split_text_embed_in_sp:
            txt_ids = torch.chunk(txt_ids, get_sequence_parallel_world_size(), dim=-2)[
                get_sequence_parallel_rank()
            ]

        output = super().forward(
            hidden_states,
            encoder_hidden_states,
            *args,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            **kwargs,
        )

        return_dict = not isinstance(output, tuple)
        sample = output[0]
        sample = get_sp_group().all_gather(sample, dim=-2)
        sample = get_cfg_group().all_gather(sample, dim=0)
        if padding_length > 0:
            sample = sample[:, :-padding_length, :]
        if return_dict:
            return output.__class__(sample, *output[1:])
        return (sample, *output[1:])


@xFuserTransformerWrappersRegister.register(FluxTransformer2DModel)
class xFuserFluxTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=(
                [FeedForward] if get_tensor_model_parallel_world_size() > 1 else []
            ),
            submodule_name_to_wrap=["attn"],
            transformer_blocks_name=["transformer_blocks", "single_transformer_blocks"],
        )
        self.encoder_hidden_states_cache = [
            None
            for _ in range(
                len(self.transformer_blocks) + len(self.single_transformer_blocks)
            )
        ]
        _register_pipefusion_reference_buffers(self)
        register_fp8_comms_eligible_modules(self, flux_attn_modules(self))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                joint_attention_kwargs is not None
                and joint_attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        if is_pipeline_first_stage():
            hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None
        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        if is_pipeline_first_stage():
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated."
                "Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        **ckpt_kwargs,
                    )
                )

            else:
                state = get_runtime_state()
                if (
                    not block.training
                    and state.patch_mode
                    and state.pipeline_patch_idx == 0
                ):
                    self.encoder_hidden_states_cache[index_block] = (
                        encoder_hidden_states
                    )
                cached_encoder_hidden_states = self.encoder_hidden_states_cache[
                    index_block
                ]
                block_encoder_hidden_states = (
                    cached_encoder_hidden_states
                    if (
                        not block.training
                        and state.patch_mode
                        and state.pipeline_patch_idx > 0
                        and cached_encoder_hidden_states is not None
                    )
                    else encoder_hidden_states
                )
                if (
                    state.patch_mode
                    and state.pipeline_patch_idx > 0
                    and cached_encoder_hidden_states is None
                ):
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=block_encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = _flux_block_forward(
                        block,
                        hidden_states,
                        block_encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                    )

            # controlnet residual
            # if controlnet_block_samples is not None:
            #     interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            #     interval_control = int(np.ceil(interval_control))
            #     hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # if self.stage_info.after_flags["transformer_blocks"]:

        for index_block, block in enumerate(self.single_transformer_blocks):
            cache_index = len(self.transformer_blocks) + index_block
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                state = get_runtime_state()
                if (
                    not block.training
                    and state.patch_mode
                    and state.pipeline_patch_idx == 0
                ):
                    self.encoder_hidden_states_cache[cache_index] = (
                        encoder_hidden_states
                    )
                cached_encoder_hidden_states = self.encoder_hidden_states_cache[
                    cache_index
                ]
                block_encoder_hidden_states = (
                    cached_encoder_hidden_states
                    if (
                        not block.training
                        and state.patch_mode
                        and state.pipeline_patch_idx > 0
                        and cached_encoder_hidden_states is not None
                    )
                    else encoder_hidden_states
                )
                if (
                    state.patch_mode
                    and state.pipeline_patch_idx > 0
                    and cached_encoder_hidden_states is None
                ):
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=block_encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                else:
                    encoder_hidden_states, hidden_states = _flux_block_forward(
                        block,
                        hidden_states,
                        block_encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        joint_attention_kwargs,
                    )

            # controlnet residual
            # if controlnet_single_block_samples is not None:
            #     interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            #     interval_control = int(np.ceil(interval_control))
            #     hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
            #         hidden_states[:, encoder_hidden_states.shape[1] :, ...]
            #         + controlnet_single_block_samples[index_block // interval_control]
            #     )


        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        encoder_hidden_states = hidden_states[:, : encoder_hidden_states.shape[1], ...]
        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        if self.stage_info.after_flags["single_transformer_blocks"]:
            hidden_states = self.norm_out(hidden_states, temb)
            output = self.proj_out(hidden_states), None
        else:
            output = hidden_states, encoder_hidden_states

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


if __name__ == "__main__":
    # print module in FluxTransformer2DModel
    model = FluxTransformer2DModel()
