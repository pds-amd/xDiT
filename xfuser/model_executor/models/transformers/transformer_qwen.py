import torch
from math import prod
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple, List
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel, apply_rotary_emb_qwen, compute_text_seq_len_from_mask
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from xfuser.model_executor.layers.usp import USP
from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers.attention_processor import (
    xFuserAttentionProcessorRegister,
)
from xfuser.model_executor.models.transformers.base_transformer import (
    xFuserTransformerBaseWrapper,
)
from xfuser.model_executor.models.transformers.register import (
    xFuserTransformerWrappersRegister,
)
from xfuser.model_executor.models.transformers.transformers_utils import chunk_and_pad_sequence, gather_and_unpad

from xfuser.model_executor.layers.fused_qk_rope_flydsl import (
    flydsl_fused_qk_norm_rope,
    _HAS_FLYDSL,
)


def _qwen_cos_sin(freqs: torch.Tensor):
    """Real ``(cos, sin)`` tables equivalent to the complex Qwen ``freqs_cis``.

    ``apply_rotary_emb_qwen(x, freqs, use_real=False)`` treats each interleaved
    channel pair ``(2k, 2k+1)`` of ``x`` as a complex number and multiplies by
    ``freqs`` -- the interleaved (GPT-J) rotation the fused kernel implements
    with ``cos = repeat_interleave(Re(freqs), 2)`` and
    ``sin = repeat_interleave(Im(freqs), 2)``.
    """
    fr = torch.view_as_real(freqs.contiguous())  # [S, D/2, 2] fp32
    cos = fr[..., 0].repeat_interleave(2, dim=-1)
    sin = fr[..., 1].repeat_interleave(2, dim=-1)
    return cos, sin


class xFuserQwenDoubleStreamAttnProcessor:

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Fuse QK-RMSNorm and RoPE into one kernel per stream when AITER/FlyDSL
        # is present. RoPE is per-token, so fusing it per stream before the
        # joint concat is identical to the unfused norm-then-rope path below.
        # Qwen's rotary emb is a complex freqs pair, so convert it to the real
        # (cos, sin) tables the kernel expects.
        if _HAS_FLYDSL and image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query, img_key = flydsl_fused_qk_norm_rope(
                img_query, img_key, attn.norm_q, attn.norm_k, _qwen_cos_sin(img_freqs)
            )
            txt_query, txt_key = flydsl_fused_qk_norm_rope(
                txt_query, txt_key, attn.norm_added_q, attn.norm_added_k, _qwen_cos_sin(txt_freqs)
            )
        else:
            # Apply QK normalization
            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)

            # Apply RoPE
            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)




        joint_hidden_states = USP(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            attn_layer=attn,
        ).transpose(1, 2)

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

        return img_attn_output, txt_attn_output


@xFuserAttentionProcessorRegister.register(xFuserQwenDoubleStreamAttnProcessor)
class xFuserQwenPipeFusionAttnProcessor(xFuserQwenDoubleStreamAttnProcessor):
    """Qwen joint attention with PipeFusion's spatial KV cache."""

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        target_image_tokens: Optional[int] = None,
        reference_patch_start: Optional[int] = None,
        reference_patch_end: Optional[int] = None,
        latte_temporal_attention: bool = False,
    ) -> torch.FloatTensor:
        seq_txt = encoder_hidden_states.shape[1]

        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))
        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if _HAS_FLYDSL and image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query, img_key = flydsl_fused_qk_norm_rope(
                img_query, img_key, attn.norm_q, attn.norm_k, _qwen_cos_sin(img_freqs)
            )
            txt_query, txt_key = flydsl_fused_qk_norm_rope(
                txt_query, txt_key, attn.norm_added_q, attn.norm_added_k, _qwen_cos_sin(txt_freqs)
            )
        else:
            if attn.norm_q is not None:
                img_query = attn.norm_q(img_query)
            if attn.norm_k is not None:
                img_key = attn.norm_k(img_key)
            if attn.norm_added_q is not None:
                txt_query = attn.norm_added_q(txt_query)
            if attn.norm_added_k is not None:
                txt_key = attn.norm_added_k(txt_key)
            if image_rotary_emb is not None:
                img_freqs, txt_freqs = image_rotary_emb
                img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
                img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
                txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
                txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        runtime_state = get_runtime_state()
        if runtime_state.num_pipeline_patch > 1:
            manager = get_cache_manager()
            if (
                runtime_state.patch_mode
                and target_image_tokens is not None
                and reference_patch_start is not None
                and reference_patch_end is not None
            ):
                # Edit inputs are laid out [target, reference]. Update the two
                # disjoint spatial ranges in the full-image cache independently.
                combined = torch.cat([img_key, img_value], dim=-1)
                cache = getattr(attn, "_xdit_kv_cache", None)
                if cache is None:
                    cache = combined
                else:
                    patch_idx = runtime_state.pipeline_patch_idx
                    target_start = runtime_state.pp_patches_token_start_idx_local[patch_idx]
                    target_end = runtime_state.pp_patches_token_start_idx_local[patch_idx + 1]
                    total_target = runtime_state.pp_patches_token_start_idx_local[-1]
                    cache[:, target_start:target_end, ...] = combined[:, :target_image_tokens, ...]
                    cache[
                        :,
                        total_target + reference_patch_start : total_target + reference_patch_end,
                        ...,
                    ] = combined[:, target_image_tokens:, ...]
                attn._xdit_kv_cache = cache
                img_key, img_value = torch.chunk(cache, 2, dim=-1)
            else:
                img_key, img_value = manager.update_and_get_kv_cache(
                    new_kv=[img_key, img_value],
                    layer=attn,
                    slice_dim=1,
                    layer_type="attn",
                )

        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)
        joint_hidden_states = USP(
            joint_query.transpose(1, 2),
            joint_key.transpose(1, 2),
            joint_value.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            attn_layer=None if runtime_state.num_pipeline_patch > 1 else attn,
        ).transpose(1, 2)
        joint_hidden_states = joint_hidden_states.flatten(2, 3).to(joint_query.dtype)
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]
        img_attn_output = joint_hidden_states[:, seq_txt:, :]
        img_attn_output = attn.to_out[0](img_attn_output.contiguous())
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)
        txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())
        return img_attn_output, txt_attn_output


@xFuserTransformerWrappersRegister.register(QwenImageTransformer2DModel)
class xFuserQwenImagePipeFusionTransformerWrapper(xFuserTransformerBaseWrapper):
    """Stage-sliced Qwen-Image transformer used by pipeline wrappers."""

    def __init__(self, transformer: QwenImageTransformer2DModel):
        for block in transformer.transformer_blocks:
            block.attn.processor = xFuserQwenDoubleStreamAttnProcessor()
        super().__init__(
            transformer=transformer,
            submodule_name_to_wrap=["attn"],
            transformer_blocks_name=["transformer_blocks"],
        )
        register_fp8_comms_eligible_modules(
            self, [block.attn for block in self.transformer_blocks]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ):
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
            pipefusion_full_img_shapes = attention_kwargs.pop(
                "_xdit_pipefusion_full_img_shapes",
                None,
            )
        else:
            attention_kwargs = {}
            lora_scale = 1.0
            pipefusion_full_img_shapes = None
        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        transformer_dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(dtype=transformer_dtype)
        encoder_hidden_states = encoder_hidden_states.to(
            dtype=transformer_dtype
        )
        if additional_t_cond is not None:
            additional_t_cond = additional_t_cond.to(
                dtype=transformer_dtype
            )
        if is_pipeline_first_stage():
            hidden_states = self.img_in(hidden_states)
            encoder_hidden_states = self.txt_in(self.txt_norm(encoder_hidden_states))

        timestep = timestep.to(hidden_states.dtype)
        if self.zero_cond_t:
            raise ValueError("Qwen zero_cond_t is not supported with PipeFusion")
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        text_seq_len, _, encoder_hidden_states_mask = (
            compute_text_seq_len_from_mask(
                encoder_hidden_states, encoder_hidden_states_mask
            )
        )
        image_rotary_emb = self.pos_embed(
            pipefusion_full_img_shapes or img_shapes,
            max_txt_seq_len=text_seq_len,
            device=hidden_states.device,
        )
        if pipefusion_full_img_shapes is not None:
            image_freqs, text_freqs = image_rotary_emb
            runtime_state = get_runtime_state()
            patch_idx = runtime_state.pipeline_patch_idx
            target_start = (
                runtime_state.pp_patches_token_start_idx_local[patch_idx]
            )
            target_end = (
                runtime_state.pp_patches_token_start_idx_local[patch_idx + 1]
            )
            patch_image_freqs = [image_freqs[target_start:target_end]]
            reference_start = attention_kwargs.get("reference_patch_start")
            reference_end = attention_kwargs.get("reference_patch_end")
            if reference_start is not None and reference_end is not None:
                target_total = (
                    runtime_state.pp_patches_token_start_idx_local[-1]
                )
                patch_image_freqs.append(
                    image_freqs[
                        target_total + reference_start :
                        target_total + reference_end
                    ]
                )
            image_rotary_emb = (
                torch.cat(patch_image_freqs, dim=0),
                text_freqs,
            )
        block_attention_kwargs = dict(attention_kwargs)
        if encoder_hidden_states_mask is not None:
            image_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
            block_attention_kwargs["attention_mask"] = torch.cat(
                [encoder_hidden_states_mask, image_mask], dim=1
            )

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=block_attention_kwargs,
                modulate_index=None,
            )
            if controlnet_block_samples is not None:
                interval = int(np.ceil(len(self.transformer_blocks) / len(controlnet_block_samples)))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval]

        if is_pipeline_last_stage():
            hidden_states = self.proj_out(self.norm_out(hidden_states, temb))
            output = (hidden_states, None)
        else:
            output = (hidden_states, encoder_hidden_states)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)
        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


class xFuserQwenImageTransformerWrapper(QwenImageTransformer2DModel):

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
        zero_cond_t: bool = False,
        use_additional_t_cond: bool = False,
        use_layer3d_rope: bool = False,
    ):
        super().__init__(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            guidance_embeds=guidance_embeds,
            axes_dims_rope=axes_dims_rope,
            zero_cond_t=zero_cond_t,
            use_additional_t_cond=use_additional_t_cond,
            use_layer3d_rope=use_layer3d_rope,
        )

        for block in self.transformer_blocks:
            block.attn.processor = xFuserQwenDoubleStreamAttnProcessor()
        register_fp8_comms_eligible_modules(
            self, [block.attn for block in self.transformer_blocks]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        additional_t_cond=None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:


        sp_world_rank = get_sequence_parallel_rank()
        sp_world_size = get_sequence_parallel_world_size()

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)

        if self.zero_cond_t:
            timestep = torch.cat([timestep, timestep * 0], dim=0)
            modulate_index = torch.tensor(
                [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
                device=timestep.device,
                dtype=torch.int,
            )
        else:
            modulate_index = None

        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)



        # Use the encoder_hidden_states sequence length for RoPE computation and normalize mask
        text_seq_len, _, encoder_hidden_states_mask = compute_text_seq_len_from_mask(
            encoder_hidden_states, encoder_hidden_states_mask
        )

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states, additional_t_cond)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states, additional_t_cond)
        )

        image_rotary_emb = self.pos_embed(img_shapes, max_txt_seq_len=text_seq_len, device=hidden_states.device)

        pad_amount = (sp_world_size - (hidden_states.shape[1] % sp_world_size)) % sp_world_size
        encoder_pad_amount = (sp_world_size - (encoder_hidden_states.shape[1] % sp_world_size)) % sp_world_size
        hidden_states = chunk_and_pad_sequence(hidden_states, sp_world_rank, sp_world_size, pad_amount, dim=1)
        encoder_hidden_states = chunk_and_pad_sequence(encoder_hidden_states, sp_world_rank, sp_world_size, encoder_pad_amount, dim=1)

        image_rotary_emb = [
            chunk_and_pad_sequence(image_rotary_emb[0], sp_world_rank, sp_world_size, pad_amount, dim=0),
            chunk_and_pad_sequence(image_rotary_emb[1], sp_world_rank, sp_world_size, encoder_pad_amount, dim=0),
        ]

        # Construct joint attention mask once to avoid reconstructing in every block
        # This eliminates 60 GPU syncs during training while maintaining torch.compile compatibility
        block_attention_kwargs = attention_kwargs.copy() if attention_kwargs is not None else {}
        if encoder_hidden_states_mask is not None:
            # Build joint mask: [text_mask, all_ones_for_image]
            batch_size, image_seq_len = hidden_states.shape[:2]
            image_mask = torch.ones((batch_size, image_seq_len), dtype=torch.bool, device=hidden_states.device)
            joint_attention_mask = torch.cat([encoder_hidden_states_mask, image_mask], dim=1)
            block_attention_kwargs["attention_mask"] = joint_attention_mask

        for index_block, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    None,  # Don't pass encoder_hidden_states_mask (using attention_mask instead)
                    temb,
                    image_rotary_emb,
                    block_attention_kwargs,
                    modulate_index,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,  # Don't pass (using attention_mask instead)
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=block_attention_kwargs,
                    modulate_index=modulate_index,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        if self.zero_cond_t:
            temb = temb.chunk(2, dim=0)[0]
        # Use only the image part (hidden_states) from the dual-stream blocks
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        output = gather_and_unpad(output, pad_amount, dim=1)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
