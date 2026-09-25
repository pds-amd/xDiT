from typing import Optional, Dict, Any, Union
import torch
import torch.distributed
import torch.nn as nn

from diffusers.models.embeddings import PatchEmbed
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import (
    is_torch_version,
    scale_lora_layers,
    USE_PEFT_BACKEND,
    unscale_lora_layers,
)

from xfuser.core.distributed.fp8_comms import register_fp8_comms_eligible_modules
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.logger import init_logger
from xfuser.model_executor.base_wrapper import xFuserBaseWrapper
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_sequence_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from .register import xFuserTransformerWrappersRegister
from .base_transformer import xFuserTransformerBaseWrapper

logger = init_logger(__name__)


def sd3_attn_modules(transformer) -> list[nn.Module]:
    """Return every SD3 self-attention module that executes USP."""
    modules = []
    for block in transformer.transformer_blocks:
        modules.append(block.attn)
        attn2 = getattr(block, "attn2", None)
        if attn2 is not None:
            modules.append(attn2)
    return modules


def _can_use_sd3_image_query_only(
    block: nn.Module,
    encoder_hidden_states: Optional[torch.Tensor],
    joint_attention_kwargs: Optional[Dict[str, Any]],
) -> bool:
    state = get_runtime_state()
    image_only_disabled = getattr(
        getattr(state, "runtime_config", None),
        "disable_pipefusion_image_query_only",
        False,
    )
    return (
        not image_only_disabled
        and isinstance(block, JointTransformerBlock)
        and not block.training
        and state.patch_mode
        and state.pipeline_patch_idx > 0
        and state.num_pipeline_patch > 1
        and get_pipeline_parallel_world_size() > 1
        and get_sequence_parallel_world_size() == 1
        and encoder_hidden_states is not None
        and not joint_attention_kwargs
        and getattr(block, "_chunk_size", None) is None
    )


def _sd3_block_forward(
    block: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    joint_attention_kwargs: Optional[Dict[str, Any]],
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    if not _can_use_sd3_image_query_only(
        block, encoder_hidden_states, joint_attention_kwargs
    ):
        return block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    if block.use_dual_attention:
        (
            norm_hidden_states,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            norm_hidden_states2,
            gate_msa2,
        ) = block.norm1(hidden_states, emb=temb)
    else:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            block.norm1(hidden_states, emb=temb)
        )

    if block.context_pre_only:
        norm_encoder_hidden_states = block.norm1_context(
            encoder_hidden_states, temb
        )
    else:
        norm_encoder_hidden_states = block.norm1_context(
            encoder_hidden_states, emb=temb
        )[0]

    attn_output = block.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_query_only=True,
    )
    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output

    if block.use_dual_attention:
        attn_output2 = block.attn2(hidden_states=norm_hidden_states2)
        hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

    norm_hidden_states = block.norm2(hidden_states)
    norm_hidden_states = (
        norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    )
    hidden_states = (
        hidden_states
        + gate_mlp.unsqueeze(1) * block.ff(norm_hidden_states)
    )
    return (
        None if block.context_pre_only else encoder_hidden_states,
        hidden_states,
    )


@xFuserTransformerWrappersRegister.register(SD3Transformer2DModel)
class xFuserSD3Transformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: SD3Transformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn", "attn2"],
        )
        self.encoder_hidden_states_cache = [
            None for _ in range(len(self.transformer_blocks))
        ]
        register_fp8_comms_eligible_modules(self, sd3_attn_modules(self))

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
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
        elif joint_attention_kwargs and "scale" in joint_attention_kwargs:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        #! ---------------------------------------- MODIFIED BELOW ----------------------------------------
        # * get height & width from runtime state
        height, width = self._get_patch_height_width()

        # * only pp rank 0 needs pos_embed (patchify)
        if is_pipeline_first_stage():
            hidden_states = self.pos_embed(
                hidden_states
            )  # takes care of adding positional embeddings too.

        #! ORIGIN:
        # height, width = hidden_states.shape[-2:]
        # hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        #! ---------------------------------------- MODIFIED ABOVE ----------------------------------------
        temb = self.time_text_embed(timestep, pooled_projections)
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if is_pipeline_first_stage():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for i, block in enumerate(self.transformer_blocks):
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
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )

            else:
                if (
                    get_runtime_state().patch_mode
                    and get_runtime_state().pipeline_patch_idx == 0
                ):
                    self.encoder_hidden_states_cache[i] = encoder_hidden_states
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )
                elif get_runtime_state().patch_mode:
                    cached_encoder_hidden_states = self.encoder_hidden_states_cache[i]
                    block_encoder_hidden_states = (
                        cached_encoder_hidden_states
                        if cached_encoder_hidden_states is not None
                        else encoder_hidden_states
                    )
                    if cached_encoder_hidden_states is None:
                        _, hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=block_encoder_hidden_states,
                            temb=temb,
                            joint_attention_kwargs=joint_attention_kwargs,
                        )
                    else:
                        _, hidden_states = _sd3_block_forward(
                            block,
                            hidden_states,
                            block_encoder_hidden_states,
                            temb,
                            joint_attention_kwargs,
                        )
                else:
                    encoder_hidden_states, hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        joint_attention_kwargs=joint_attention_kwargs,
                    )

        # * only the last pp rank needs unpatchify
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        if is_pipeline_last_stage():
            #! ---------------------------------------- ADD ABOVE ----------------------------------------
            hidden_states = self.norm_out(hidden_states, temb)
            hidden_states = self.proj_out(hidden_states)

            # unpatchify
            patch_size = self.config.patch_size

            hidden_states = hidden_states.reshape(
                shape=(
                    hidden_states.shape[0],
                    height,
                    width,
                    patch_size,
                    patch_size,
                    self.out_channels,
                )
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = (
                hidden_states.reshape(
                    shape=(
                        hidden_states.shape[0],
                        self.out_channels,
                        height * patch_size,
                        width * patch_size,
                    )
                ),
                None,
            )

            if USE_PEFT_BACKEND:
                # remove `lora_scale` from each PEFT layer
                unscale_lora_layers(self, lora_scale)
        #! ---------------------------------------- ADD BELOW ----------------------------------------
        else:
            output = hidden_states, encoder_hidden_states
        #! ---------------------------------------- ADD ABOVE ----------------------------------------

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
