import torch
import math
from torch.nn.utils.rnn import pad_sequence
from typing import List, Optional

from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput


from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.model_executor.layers.fused_qk_rope_zimage_flydsl import (
    flydsl_fused_qk_norm_rope,
)

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_ulysses_parallel_world_size,
    get_sp_group,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32


def _scatter_pad_token(x: torch.Tensor, mask: torch.Tensor, pad_token: torch.Tensor) -> torch.Tensor:
    """``x[mask] = pad_token`` without the device->host sync.

    Boolean-mask ``index_put_`` lowers to ``nonzero()`` on CUDA, which must read the
    match count back to the host to size its output -- a full blocking sync on the
    hot path, twice per denoise step (x tokens and caption tokens).

    ``torch.where`` computes exactly the same result as a data-independent
    elementwise select: no ``nonzero``, no host round-trip, no dynamic shape (which
    also removes a torch.compile graph break).  Rows selected by ``mask`` get
    ``pad_token`` broadcast across the feature dim, all other rows are copied
    verbatim, so the produced tensor is bit-identical to the in-place form.

    Ungated and selected purely from tensor properties; anything that does not
    match the expected 2-D (tokens, dim) / 1-D bool-mask layout falls through to
    the original in-place reference path.
    """
    if (
        x.dim() == 2
        and mask.dim() == 1
        and mask.dtype == torch.bool
        and mask.shape[0] == x.shape[0]
        and pad_token.dim() == 2
        and pad_token.shape[0] == 1
        and pad_token.shape[1] == x.shape[1]
    ):
        return torch.where(mask.unsqueeze(-1), pad_token.to(x.dtype), x)
    # _reference fallback: unchanged upstream behaviour.
    x[mask] = pad_token
    return x

class xFuserZSingleStreamAttnProcessor:
    """
    Processor for Z-Image single stream attention that adapts the existing Attention class to match the behavior of the
    original Z-ImageAttention module.
    """

    @staticmethod
    def _pad_heads(x: torch.Tensor, pad_heads: int) -> torch.Tensor:
        if pad_heads <= 0:
            return x
        pad_shape = list(x.shape)
        pad_shape[1] = pad_heads
        return torch.cat([x, x.new_zeros(pad_shape)], dim=1)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        latte_temporal_attention: bool = False,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # Apply Norms + RoPE.
        #
        # The wrapper selects from tensor properties alone and falls back to the
        # unfused diffusers path (norm then complex rope) whenever the FlyDSL
        # stack is absent or the shape is out of envelope, so this call is
        # numerically interchangeable with the code it replaced.
        query, key = flydsl_fused_qk_norm_rope(
            query, key, attn.norm_q, attn.norm_k, freqs_cis
        )

        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Transpose for attention
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        ulysses_world_size = get_ulysses_parallel_world_size()
        pad_heads = 0
        if ulysses_world_size > 1:
            pad_heads = (ulysses_world_size - (query.shape[1] % ulysses_world_size)) % ulysses_world_size
            if pad_heads:
                query = self._pad_heads(query, pad_heads)
                key = self._pad_heads(key, pad_heads)
                value = self._pad_heads(value, pad_heads)

        hidden_states = USP(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attn_layer=attn,
        )

        if pad_heads:
            hidden_states = hidden_states[:, :-pad_heads, :, :]

        # Transpose back to original shape
        hidden_states = hidden_states.transpose(1, 2)

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)

        return output


class xFuserZImagePipeFusionAttnProcessor(xFuserZSingleStreamAttnProcessor):
    """Z-Image attention with a full-image stale-KV cache.

    Z-Image's unified stream is ordered ``[image, caption]``. Only image tokens
    are spatially patched; caption KV is recomputed for every patch.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        latte_temporal_attention: bool = False,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        query, key = flydsl_fused_qk_norm_rope(
            query, key, attn.norm_q, attn.norm_k, freqs_cis
        )
        dtype = query.dtype

        runtime_state = get_runtime_state()
        image_tokens = int(getattr(attn, "_xfuser_image_tokens", 0))
        if runtime_state.num_pipeline_patch > 1 and image_tokens > 0:
            image_key, caption_key = key.split(
                [image_tokens, key.shape[1] - image_tokens], dim=1
            )
            image_value, caption_value = value.split(
                [image_tokens, value.shape[1] - image_tokens], dim=1
            )
            image_key, image_value = get_cache_manager().update_and_get_kv_cache(
                new_kv=[image_key, image_value],
                layer=attn,
                slice_dim=1,
                layer_type="attn",
            )
            key = torch.cat([image_key, caption_key], dim=1)
            value = torch.cat([image_value, caption_value], dim=1)

            if (
                attention_mask is not None
                and attention_mask.ndim == 2
                and runtime_state.patch_mode
            ):
                full_image_tokens = int(
                    getattr(attn, "_xfuser_full_image_tokens", image_key.shape[1])
                )
                # PipeFusion deliberately exposes the whole spatial cache:
                # earlier stripes contain this step's KV and later stripes
                # retain the previous step's KV. Masking later stripes would
                # remove the cross-patch context the algorithm approximates.
                image_mask = torch.ones(
                    attention_mask.shape[0],
                    full_image_tokens,
                    dtype=torch.bool,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    [image_mask, attention_mask[:, image_tokens:]], dim=1
                )

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        hidden_states = USP(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
            attn_layer=None if runtime_state.num_pipeline_patch > 1 else attn,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


def z_image_attn_modules(transformer) -> list[torch.nn.Module]:
    """Return every Z-Image attention module that executes USP."""
    return [
        layer.attention
        for layer in (
            *transformer.noise_refiner,
            *transformer.context_refiner,
            *transformer.layers,
        )
    ]


class xFuserZImageTransformer2DWrapper(ZImageTransformer2DModel):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        for layer in self.layers + self.context_refiner + self.noise_refiner:
            layer.attention.processor = xFuserZSingleStreamAttnProcessor()
        register_fp8_comms_eligible_modules(self, z_image_attn_modules(self))


    def _chunk_and_pad_sequence(self, x: torch.Tensor, sp_world_rank: int, sp_world_size: int, pad_amount: int, dim: int) -> torch.Tensor:
        if pad_amount > 0:
            if dim < 0:
                dim = x.ndim + dim
            pad_shape = list(x.shape)
            pad_shape[dim] = pad_amount
            x = torch.cat([x,
                        torch.zeros(
                            pad_shape,
                            dtype=x.dtype,
                            device=x.device,
                        )], dim=dim)
        x = torch.chunk(x,
                        sp_world_size,
                        dim=dim)[sp_world_rank]
        return x

    def _gather_and_unpad(self, x: torch.Tensor, pad_amount: int, dim: int) -> torch.Tensor:
        x = get_sp_group().all_gather(x, dim=dim)
        size = x.size(dim)
        return x.narrow(dim=dim, start=0, length=size - pad_amount)

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        cfg_world_size = get_classifier_free_guidance_world_size()
        cfg_rank = get_classifier_free_guidance_rank()
        do_cfg_parallel = cfg_world_size > 1

        if do_cfg_parallel:
            B = len(x) // cfg_world_size
            x = x[cfg_rank * B : (cfg_rank + 1) * B]
            cap_feats = cap_feats[cfg_rank * B : (cfg_rank + 1) * B]
            t = t.chunk(cfg_world_size, dim=0)[cfg_rank]

        bsz = len(x)
        device = x[0].device
        # The pipeline keeps the timestep schedule host-resident so that its
        # per-step ``t_norm = timestep[0].item()`` is not a blocking D2H sync
        # (see xfuser/model_executor/models/runner_models/z_image.py::
        # _keep_timesteps_host_resident).  Pay for that with one cheap
        # non-blocking H2D copy of the (B,) timestep vector instead of a full
        # queue drain per denoise step.  Selected purely from tensor properties;
        # if ``t`` is already on-device this is a no-op and the original
        # behaviour is preserved byte for byte.
        if isinstance(t, torch.Tensor) and t.device != device:
            t = t.to(device, non_blocking=True)
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        # assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x = _scatter_pad_token(x, torch.cat(x_inner_pad_mask), self.x_pad_token)
        x = list(x.split(x_item_seqlens, dim=0))
        x_freqs_cis = list(self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        # SP support
        pad_amount = (sp_world_size - (x.shape[1] % sp_world_size)) % sp_world_size
        x = self._chunk_and_pad_sequence(x, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        x_attn_mask = self._chunk_and_pad_sequence(x_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        x_freqs_cis_chunked = self._chunk_and_pad_sequence(x_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.noise_refiner:
                x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis_chunked, adaln_input)
        else:
            for layer in self.noise_refiner:
                x = layer(x, x_attn_mask, x_freqs_cis_chunked, adaln_input)

        # Gather SP outputs and remove padding
        x = self._gather_and_unpad(x, pad_amount, dim=-2)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        # assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats = _scatter_pad_token(cap_feats, torch.cat(cap_inner_pad_mask), self.cap_pad_token)
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_freqs_cis = list(self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split([len(_) for _ in cap_pos_ids], dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
        # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
        cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1


        # SP support
        pad_amount = (sp_world_size - (cap_feats.shape[1] % sp_world_size)) % sp_world_size
        cap_feats = self._chunk_and_pad_sequence(cap_feats, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        cap_attn_mask = self._chunk_and_pad_sequence(cap_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        cap_freqs_cis_chunked = self._chunk_and_pad_sequence(cap_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.context_refiner:
                cap_feats = self._gradient_checkpointing_func(layer, cap_feats, cap_attn_mask, cap_freqs_cis_chunked)
        else:
            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis_chunked)

        # Gather SP outputs and remove padding
        cap_feats = self._gather_and_unpad(cap_feats, pad_amount, dim=-2)


        # unified
        unified = []
        unified_freqs_cis = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        assert unified_item_seqlens == [len(_) for _ in unified]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        # SP support
        pad_amount = (sp_world_size - (unified.shape[1] % sp_world_size)) % sp_world_size
        unified = self._chunk_and_pad_sequence(unified, sp_world_rank, sp_world_size, pad_amount, dim=-2)
        unified_attn_mask = self._chunk_and_pad_sequence(unified_attn_mask, sp_world_rank, sp_world_size, pad_amount, dim=-1)
        unified_freqs_cis = self._chunk_and_pad_sequence(unified_freqs_cis, sp_world_rank, sp_world_size, pad_amount, dim=-2)

        if torch.is_grad_enabled() and self.gradient_checkpointing:
            for layer in self.layers:
                unified = self._gradient_checkpointing_func(
                    layer, unified, unified_attn_mask, unified_freqs_cis, adaln_input
                )
        else:
            for layer in self.layers:
                unified = layer(unified, unified_attn_mask, unified_freqs_cis, adaln_input)

        # Gather SP outputs and remove padding
        unified = self._gather_and_unpad(unified, pad_amount, dim=-2)

        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        if do_cfg_parallel:
            x_stacked = torch.stack(x, dim=0)
            x_stacked = get_cfg_group().all_gather(x_stacked, dim=0)
            x = list(x_stacked.unbind(dim=0))

        if not return_dict:
            return (x,)

        return Transformer2DModelOutput(sample=x)


@xFuserTransformerWrappersRegister.register(ZImageTransformer2DModel)
class xFuserZImagePipeFusionTransformerWrapper(xFuserTransformerBaseWrapper):
    """Stage-sliced Z-Image transformer for PipeFusion.

    Refiner and main block lists form one contiguous pipeline. At context-only
    boundaries the image and caption states travel as one unified tensor, so
    the pipeline transport remains a single tensor.
    """

    transformer_blocks_name = ["noise_refiner", "context_refiner", "layers"]

    def __init__(self, transformer: ZImageTransformer2DModel):
        for layer in (
            *transformer.noise_refiner,
            *transformer.context_refiner,
            *transformer.layers,
        ):
            layer.attention.processor = (
                xFuserZImagePipeFusionAttnProcessor()
            )
        super().__init__(
            transformer=transformer,
            submodule_name_to_wrap=[],
            transformer_blocks_name=self.transformer_blocks_name,
        )
        cache_manager = get_cache_manager()
        for attention in z_image_attn_modules(self):
            if not cache_manager.has_cache_entry(attention):
                cache_manager.register_cache_entry(attention, "attn")
        register_fp8_comms_eligible_modules(self, z_image_attn_modules(self))

    @staticmethod
    def _starts_at_zero(blocks: torch.nn.ModuleList) -> bool:
        if not blocks:
            return False
        fqn = getattr(blocks[0], "_xfuser_checkpoint_fqn", "")
        return fqn.endswith(".0")

    @staticmethod
    def _unify(
        x,
        cap_feats,
        x_freqs,
        cap_freqs,
        x_lengths,
        cap_lengths,
        device,
    ):
        unified = [
            torch.cat([x[i][:x_lengths[i]], cap_feats[i][:cap_lengths[i]]])
            for i in range(len(x_lengths))
        ]
        unified_freqs = [
            torch.cat(
                [x_freqs[i][:x_lengths[i]], cap_freqs[i][:cap_lengths[i]]]
            )
            for i in range(len(x_lengths))
        ]
        lengths = [a + b for a, b in zip(x_lengths, cap_lengths)]
        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_freqs = pad_sequence(
            unified_freqs, batch_first=True, padding_value=0.0
        )
        mask = torch.zeros(
            (len(lengths), max(lengths)), dtype=torch.bool, device=device
        )
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        return unified, unified_freqs, mask

    @staticmethod
    def _set_image_token_metadata(
        blocks, image_tokens: int, full_image_tokens: int
    ) -> None:
        for layer in blocks:
            attention = getattr(layer, "attention", None)
            if attention is not None:
                attention._xfuser_image_tokens = image_tokens
                attention._xfuser_full_image_tokens = full_image_tokens
                continue
            cached_blocks = getattr(layer, "transformer_blocks", None)
            if cached_blocks is not None:
                xFuserZImagePipeFusionTransformerWrapper._set_image_token_metadata(
                    cached_blocks, image_tokens, full_image_tokens
                )

    def forward(
        self,
        x: List[torch.Tensor],
        t,
        cap_feats: List[torch.Tensor],
        hidden_states: Optional[torch.Tensor] = None,
        patch_start_height: int = 0,
        full_image_tokens: Optional[int] = None,
        patch_size: int = 2,
        f_patch_size: int = 1,
        return_dict: bool = True,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size
        device = x[0].device
        if hidden_states is not None:
            hidden_states = hidden_states.to(
                dtype=next(self.parameters()).dtype
            )
        t = t.to(device=device, non_blocking=True) * self.t_scale
        adaln_input = self.t_embedder(t)

        (
            x_source,
            cap_source,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_pad_mask,
            cap_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # patchify_and_embed numbers every spatial patch from row zero. Restore
        # global image-row coordinates so each PipeFusion patch gets the same
        # RoPE positions as the unsliced image.
        row_offset = patch_start_height // patch_size
        if row_offset:
            for ids, pad_mask in zip(x_pos_ids, x_pad_mask):
                ids[~pad_mask, 1] += row_offset

        x_lengths = [len(value) for value in x_source]
        cap_lengths = [len(value) for value in cap_source]
        x_source = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](
            torch.cat(x_source, dim=0)
        )
        adaln_input = adaln_input.type_as(x_source)
        x_source = _scatter_pad_token(
            x_source, torch.cat(x_pad_mask), self.x_pad_token
        )
        x_source = pad_sequence(
            list(x_source.split(x_lengths, dim=0)),
            batch_first=True,
            padding_value=0.0,
        )
        x_freqs = pad_sequence(
            list(
                self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split(
                    [len(ids) for ids in x_pos_ids], dim=0
                )
            ),
            batch_first=True,
            padding_value=0.0,
        )
        x_mask = torch.zeros(
            (len(x_lengths), max(x_lengths)), dtype=torch.bool, device=device
        )
        for i, length in enumerate(x_lengths):
            x_mask[i, :length] = True

        cap_source = self.cap_embedder(torch.cat(cap_source, dim=0))
        cap_source = _scatter_pad_token(
            cap_source, torch.cat(cap_pad_mask), self.cap_pad_token
        )
        cap_source = pad_sequence(
            list(cap_source.split(cap_lengths, dim=0)),
            batch_first=True,
            padding_value=0.0,
        )
        cap_freq_chunks = self.rope_embedder(
            torch.cat(cap_pos_ids, dim=0)
        ).split([len(ids) for ids in cap_pos_ids], dim=0)
        cap_freqs = pad_sequence(
            [
                frequencies[:length]
                for frequencies, length in zip(
                    cap_freq_chunks, cap_lengths
                )
            ],
            batch_first=True,
            padding_value=0.0,
        )
        cap_mask = torch.zeros(
            (len(cap_lengths), max(cap_lengths)),
            dtype=torch.bool,
            device=device,
        )
        for i, length in enumerate(cap_lengths):
            cap_mask[i, :length] = True

        local_image_tokens = x_source.shape[1]
        full_image_tokens = int(full_image_tokens or local_image_tokens)
        has_noise = len(self.noise_refiner) > 0
        has_context = len(self.context_refiner) > 0
        has_main = len(self.layers) > 0

        unified = unified_freqs = unified_mask = None
        if is_pipeline_first_stage():
            x_state = x_source
            cap_state = cap_source
        elif has_noise:
            x_state = hidden_states
            cap_state = cap_source
        elif has_context and self._starts_at_zero(self.context_refiner):
            x_state = hidden_states
            cap_state = cap_source
        elif has_context:
            x_state = hidden_states[:, :local_image_tokens]
            cap_state = hidden_states[
                :, local_image_tokens : local_image_tokens + cap_source.shape[1]
            ]
        else:
            unified = hidden_states
            x_state = cap_state = None

        if has_noise:
            self._set_image_token_metadata(
                self.noise_refiner, local_image_tokens, full_image_tokens
            )
            for layer in self.noise_refiner:
                x_state = layer(x_state, x_mask, x_freqs, adaln_input)

        if has_context:
            self._set_image_token_metadata(self.context_refiner, 0, 0)
            for layer in self.context_refiner:
                cap_state = layer(cap_state, cap_mask, cap_freqs)

        if has_context or (has_main and unified is None):
            unified, unified_freqs, unified_mask = self._unify(
                x_state,
                cap_state,
                x_freqs,
                cap_freqs,
                x_lengths,
                cap_lengths,
                device,
            )
        elif unified is not None:
            _, unified_freqs, unified_mask = self._unify(
                x_source,
                cap_source,
                x_freqs,
                cap_freqs,
                x_lengths,
                cap_lengths,
                device,
            )

        if has_main:
            self._set_image_token_metadata(
                self.layers, local_image_tokens, full_image_tokens
            )
            for layer in self.layers:
                unified = layer(
                    unified, unified_mask, unified_freqs, adaln_input
                )

        if is_pipeline_last_stage():
            unified = self.all_final_layer[
                f"{patch_size}-{f_patch_size}"
            ](unified, adaln_input)
            image_states = [
                value[:x_lengths[i]]
                for i, value in enumerate(unified.unbind(dim=0))
            ]
            output = self.unpatchify(
                image_states, x_size, patch_size, f_patch_size
            )
        elif has_noise and not has_context and not has_main:
            output = x_state
        else:
            output = unified

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

