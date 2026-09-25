"""PipeFusion wrappers for Qwen-Image text-to-image and image editing."""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import QwenImageEditPipeline, QwenImagePipeline
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
    calculate_shift,
    retrieve_timesteps,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import calculate_dimensions

from xfuser.config import EngineConfig
from xfuser.model_executor.pipefusion import (
    CombinedTensorPayloadCodec,
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
    PipeFusionStagePayload,
    PipeFusionTransport,
)
from xfuser.core.distributed import (
    get_cfg_group,
    get_classifier_free_guidance_rank,
    get_classifier_free_guidance_world_size,
    get_pipeline_parallel_world_size,
    get_pp_group,
    get_runtime_state,
    is_dp_last_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)
from xfuser.model_executor.models.transformers.transformer_qwen import (  # noqa: F401
    xFuserQwenImagePipeFusionTransformerWrapper,
    xFuserQwenImageTransformerWrapper,
)

from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister


def _pad_condition_sequence(embeds, mask, target_length):
    current_length = embeds.shape[1]
    if current_length == target_length:
        return embeds, mask
    if mask is None:
        mask = torch.ones(
            embeds.shape[0],
            current_length,
            dtype=torch.bool,
            device=embeds.device,
        )
    embeds = torch.cat(
        [
            embeds,
            embeds.new_zeros(
                embeds.shape[0],
                target_length - current_length,
                embeds.shape[2],
            ),
        ],
        dim=1,
    )
    mask = torch.cat(
        [
            mask,
            mask.new_zeros(
                mask.shape[0],
                target_length - current_length,
            ),
        ],
        dim=1,
    )
    return embeds, mask


class xFuserQwenImagePipelineBase(xFuserPipelineBaseWrapper):
    """Shared Qwen preparation, CFG, PipeFusion scheduling, and decoding."""

    _diffusers_cls = QwenImagePipeline
    _is_edit = False

    def _convert_transformer_backbone(self, transformer, **kwargs):
        # The runner's non-PipeFusion load path already constructs the
        # Ulysses-capable subclass and the loader still needs that concrete
        # component type for blockwise materialization. Do not wrap it in the
        # composition-style PipeFusion wrapper a second time.
        if isinstance(transformer, xFuserQwenImageTransformerWrapper):
            return transformer
        return super()._convert_transformer_backbone(transformer, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = cls._diffusers_cls.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def _prepare_image(
        self, image, height: Optional[int], width: Optional[int]
    ):
        return image, height, width, None, None

    def _prepare_model_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        return (
            self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                dtype,
                device,
                generator,
                latents,
            ),
            None,
        )

    def _encode_condition(
        self,
        prompt,
        prompt_image,
        prompt_embeds,
        prompt_embeds_mask,
        device,
        num_images_per_prompt,
        max_sequence_length,
    ):
        kwargs = dict(
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if self._is_edit:
            kwargs["image"] = prompt_image
        return self.encode_prompt(**kwargs)

    def _place_text_encoder(self, device) -> None:
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None or hasattr(text_encoder, "_hf_hook"):
            return
        text_encoder.to(device)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        image=None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_embeds_mask=None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        **kwargs,
    ):
        image, height, width, prompt_image, image_shape = self._prepare_image(
            image, height, width
        )
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs or {}
        self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device
        self._place_text_encoder(device)
        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=False,
        )

        has_negative = negative_prompt is not None or (
            negative_prompt_embeds is not None
            and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_negative
        prompt_embeds, prompt_embeds_mask = self._encode_condition(
            prompt,
            prompt_image,
            prompt_embeds,
            prompt_embeds_mask,
            device,
            num_images_per_prompt,
            max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = (
                self._encode_condition(
                    negative_prompt,
                    prompt_image,
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    device,
                    num_images_per_prompt,
                    max_sequence_length,
                )
            )

        cfg_size = get_classifier_free_guidance_world_size()
        if do_true_cfg and cfg_size == 2:
            if get_classifier_free_guidance_rank() == 0:
                prompt_embeds = negative_prompt_embeds
                prompt_embeds_mask = negative_prompt_embeds_mask
        elif do_true_cfg and cfg_size == 1:
            target_length = max(
                prompt_embeds.shape[1],
                negative_prompt_embeds.shape[1],
            )
            prompt_embeds, prompt_embeds_mask = _pad_condition_sequence(
                prompt_embeds,
                prompt_embeds_mask,
                target_length,
            )
            negative_prompt_embeds, negative_prompt_embeds_mask = (
                _pad_condition_sequence(
                    negative_prompt_embeds,
                    negative_prompt_embeds_mask,
                    target_length,
                )
            )
            prompt_embeds = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            if (
                prompt_embeds_mask is not None
                or negative_prompt_embeds_mask is not None
            ):
                if prompt_embeds_mask is None:
                    prompt_embeds_mask = torch.ones(
                        prompt_embeds.shape[0] // 2,
                        prompt_embeds.shape[1],
                        dtype=torch.bool,
                        device=prompt_embeds.device,
                    )
                if negative_prompt_embeds_mask is None:
                    negative_prompt_embeds_mask = torch.ones_like(
                        prompt_embeds_mask
                    )
                prompt_embeds_mask = torch.cat(
                    [negative_prompt_embeds_mask, prompt_embeds_mask], dim=0
                )
        elif do_true_cfg:
            raise ValueError("Qwen true CFG supports CFG degree 1 or 2")

        # The 2K VAE has a multi-GiB decode peak. Prompt encoding is complete,
        # so release the replicated language encoder before denoising/decoding.
        self._place_text_encoder("cpu")
        torch.cuda.empty_cache()

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self._prepare_model_latents(
            image,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        target_shape = (
            1,
            height // self.vae_scale_factor // 2,
            width // self.vae_scale_factor // 2,
        )
        full_img_shapes = [[target_shape]]
        if image_shape is not None:
            full_img_shapes[0].append(image_shape)
        full_img_shapes *= batch_size
        if do_true_cfg and cfg_size == 1:
            full_img_shapes *= 2

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        mu = calculate_shift(
            latents.shape[1],
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)
        if self.transformer.config.guidance_embeds:
            if guidance_scale is None:
                raise ValueError(
                    "guidance_scale is required for guidance-distilled Qwen models"
                )
            guidance = torch.full(
                [latents.shape[0]], guidance_scale, device=device, dtype=torch.float32
            )
        else:
            guidance = None

        self.scheduler.set_begin_index(0)
        pipeline_warmup = get_runtime_state().runtime_config.warmup_steps
        common = dict(
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            image_latents=image_latents,
            full_img_shapes=full_img_shapes,
            guidance=guidance,
            do_true_cfg=do_true_cfg,
            true_cfg_scale=true_cfg_scale,
            num_warmup_steps=num_warmup_steps,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            common["progress_bar"] = progress_bar
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > pipeline_warmup
            ):
                latents = self._sync_pipeline(
                    latents=latents,
                    timesteps=timesteps[:pipeline_warmup],
                    **common,
                )
                latents = self._async_pipeline(
                    latents=latents,
                    timesteps=timesteps[pipeline_warmup:],
                    **common,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    timesteps=timesteps,
                    sync_only=True,
                    **common,
                )

        def process_latents(value):
            value = self._unpack_latents(
                value, height, width, self.vae_scale_factor
            ).to(self.vae.dtype)
            latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(value.device, value.dtype)
            latents_std = (
                1.0
                / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                )
            ).to(value.device, value.dtype)
            return value / latents_std + latents_mean

        image_out = None
        runtime_state = get_runtime_state()
        if output_type == "latent":
            image_out = latents
        elif (
            runtime_state.runtime_config.use_parallel_vae
            and runtime_state.parallel_config.vae_parallel_size > 0
        ):
            latents = self.gather_latents_for_vae(latents)
            if latents is not None:
                latents = process_latents(latents)
            self.send_to_vae_decode(latents)
        elif runtime_state.runtime_config.use_parallel_vae:
            latents = self.gather_broadcast_latents(latents)
            latents = process_latents(latents)
            image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
        elif is_dp_last_group() and latents is not None:
            latents = process_latents(latents)
            image_out = self.vae.decode(latents, return_dict=False)[0][:, :, 0]

        if output_type != "latent" and image_out is not None:
            image_out = self.image_processor.postprocess(
                image_out, output_type=output_type
            )

        if not is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image_out,)
        return QwenImagePipelineOutput(images=image_out)

    def _patch_shapes(self, full_img_shapes, patch_idx, reference_counts=None):
        target_width = full_img_shapes[0][0][2]
        target_tokens = get_runtime_state().pp_patches_token_num[patch_idx]
        target_shape = (1, target_tokens // target_width, target_width)
        shapes = [target_shape]
        if reference_counts is not None:
            ref_width = full_img_shapes[0][1][2]
            shapes.append((1, reference_counts[patch_idx] // ref_width, ref_width))
        return [shapes for _ in full_img_shapes]

    def _backbone_forward(
        self,
        latents,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        image_latents,
        img_shapes,
        guidance,
        t,
        do_true_cfg,
        true_cfg_scale,
        target_tokens,
        reference_range=None,
        full_rope_img_shapes=None,
    ):
        hidden_states = latents
        local_batched_cfg = (
            do_true_cfg and get_classifier_free_guidance_world_size() == 1
        )
        if is_pipeline_first_stage():
            if local_batched_cfg:
                hidden_states = torch.cat([hidden_states, hidden_states], dim=0)
                if image_latents is not None:
                    image_latents = torch.cat([image_latents, image_latents], dim=0)
            if image_latents is not None:
                hidden_states = torch.cat([hidden_states, image_latents], dim=1)
        if local_batched_cfg and guidance is not None:
            guidance = torch.cat([guidance, guidance], dim=0)

        call_attention_kwargs = dict(self.attention_kwargs)
        if full_rope_img_shapes is not None:
            call_attention_kwargs["_xdit_pipefusion_full_img_shapes"] = (
                full_rope_img_shapes
            )
        if reference_range is not None:
            call_attention_kwargs.update(
                target_image_tokens=target_tokens,
                reference_patch_start=reference_range[0],
                reference_patch_end=reference_range[1],
            )
        timestep = t.expand(hidden_states.shape[0]).to(hidden_states.dtype)
        ret = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=guidance,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            encoder_hidden_states=encoder_hidden_states,
            img_shapes=img_shapes,
            attention_kwargs=call_attention_kwargs,
            return_dict=False,
        )[0]
        if get_pipeline_parallel_world_size() > 1:
            noise_pred, encoder_hidden_states = ret
        else:
            noise_pred, encoder_hidden_states = ret, None

        if is_pipeline_last_stage():
            noise_pred = noise_pred[:, :target_tokens]
            if do_true_cfg:
                if get_classifier_free_guidance_world_size() == 2:
                    neg, cond = get_cfg_group().all_gather(
                        noise_pred, separate_tensors=True
                    )
                else:
                    neg, cond = noise_pred.chunk(2, dim=0)
                combined = neg + true_cfg_scale * (cond - neg)
                cond_norm = torch.norm(cond, dim=-1, keepdim=True)
                noise_norm = torch.norm(combined, dim=-1, keepdim=True)
                noise_pred = combined * (cond_norm / noise_norm.clamp_min(1e-12))
        return noise_pred, encoder_hidden_states

    def _sync_pipeline(
        self,
        latents,
        prompt_embeds,
        prompt_embeds_mask,
        image_latents,
        full_img_shapes,
        guidance,
        timesteps,
        do_true_cfg,
        true_cfg_scale,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
        sync_only=False,
    ):
        get_runtime_state().set_patched_mode(patch_mode=False)
        target_tokens = latents.shape[1]
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            if is_pipeline_last_stage():
                previous_latents = latents
            if get_pipeline_parallel_world_size() > 1 and not (
                is_pipeline_first_stage() and i == 0
            ):
                latents = get_pp_group().pipeline_recv()
                if not is_pipeline_first_stage():
                    prompt_states = get_pp_group().pipeline_recv(
                        0, "encoder_hidden_states"
                    )
            latents, prompt_states = self._backbone_forward(
                latents,
                prompt_embeds if is_pipeline_first_stage() else prompt_states,
                prompt_embeds_mask,
                image_latents,
                full_img_shapes,
                guidance,
                t,
                do_true_cfg,
                true_cfg_scale,
                target_tokens,
            )
            if is_pipeline_last_stage():
                latents = self.scheduler.step(
                    latents, t, previous_latents, return_dict=False
                )[0]
                if callback_on_step_end is not None:
                    callback_outputs = callback_on_step_end(
                        self, i, t, {"latents": latents}
                    )
                    latents = callback_outputs.pop("latents", latents)
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps
                and (i + 1) % self.scheduler.order == 0
            ):
                progress_bar.update()
            if not (
                sync_only
                and is_pipeline_last_stage()
                and i == len(timesteps) - 1
            ) and get_pipeline_parallel_world_size() > 1:
                get_pp_group().pipeline_send(latents)
                if not is_pipeline_last_stage():
                    get_pp_group().pipeline_send(
                        prompt_states, name="encoder_hidden_states"
                    )
        return latents

    def _async_pipeline(
        self,
        latents,
        prompt_embeds,
        prompt_embeds_mask,
        image_latents,
        full_img_shapes,
        guidance,
        timesteps,
        do_true_cfg,
        true_cfg_scale,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
    ):
        if len(timesteps) == 0:
            return latents
        state = get_runtime_state()
        num_patches = state.num_pipeline_patch
        pipeline_warmup = state.runtime_config.warmup_steps
        patch_latents = self._init_async_pipeline(
            len(timesteps),
            latents,
            pipeline_warmup,
            split_sizes=state.pp_patches_token_num,
            split_dim=1,
            queue_receives=False,
        )

        image_patches = None
        reference_ranges = None
        reference_counts = None
        if image_latents is not None:
            reference_height = full_img_shapes[0][1][1]
            reference_width = full_img_shapes[0][1][2]
            reference_rows = image_latents.reshape(
                image_latents.shape[0],
                reference_height,
                reference_width,
                image_latents.shape[-1],
            )
            image_patches = [
                rows.flatten(1, 2)
                for rows in torch.tensor_split(
                    reference_rows, num_patches, dim=1
                )
            ]
            reference_counts = [value.shape[1] for value in image_patches]
            starts = np.cumsum([0] + reference_counts).tolist()
            reference_ranges = list(zip(starts[:-1], starts[1:]))

        layout = PipeFusionPatchLayout.from_runtime_state(
            state,
            split_dim=1,
            split_sizes=state.pp_patches_token_num,
            reference_token_counts=reference_counts,
            name="qwen-target+reference",
        )
        payload_codec = CombinedTensorPayloadCodec(
            layout,
            model_name="Qwen PipeFusion",
        )
        transport = PipeFusionTransport(
            get_pp_group(),
            first_stage=is_pipeline_first_stage(),
            num_steps=len(timesteps),
            num_patches=num_patches,
        )
        previous = [None] * num_patches if is_pipeline_last_stage() else None
        condition_states = [None] * num_patches
        computation_mask = self._pipefusion_async_computation_mask(
            len(timesteps) + pipeline_warmup,
            pipeline_warmup,
        )
        stage_output_cache = self._pipefusion_stage_output_cache(
            computation_mask,
            num_patches,
        )

        def prepare_patch(work, _timestep, received):
            patch_idx = work.patch_index
            if is_pipeline_last_stage():
                previous[patch_idx] = patch_latents[patch_idx]
            if received is not None:
                if is_pipeline_first_stage():
                    patch_latents[patch_idx] = received
                else:
                    payload = payload_codec.unpack(received, patch_idx)
                    patch_latents[patch_idx] = payload.image_state
                    condition_states[patch_idx] = payload.condition_state
            return PipeFusionStagePayload(
                image_state=patch_latents[patch_idx],
                condition_state=(
                    prompt_embeds
                    if is_pipeline_first_stage()
                    else condition_states[patch_idx]
                ),
            )

        def forward_patch(work, timestep, prepared):
            patch_idx = work.patch_index
            return self._backbone_forward(
                prepared.image_state,
                prepared.condition_state,
                prompt_embeds_mask,
                image_patches[patch_idx] if image_patches is not None else None,
                self._patch_shapes(
                    full_img_shapes, patch_idx, reference_counts
                ),
                guidance,
                timestep,
                do_true_cfg,
                true_cfg_scale,
                state.pp_patches_token_num[patch_idx],
                reference_ranges[patch_idx]
                if reference_ranges is not None
                else None,
                full_img_shapes,
            )

        def commit_patch(work, timestep, output):
            patch_idx = work.patch_index
            noise_pred, next_prompt_states = output
            patch_latents[patch_idx] = noise_pred
            if is_pipeline_last_stage():
                patch_latents[patch_idx] = self.scheduler.step(
                    noise_pred,
                    timestep,
                    previous[patch_idx],
                    return_dict=False,
                )[0]
                if work.step_index == work.num_steps - 1:
                    return None
                return patch_latents[patch_idx]
            return payload_codec.pack(
                PipeFusionStagePayload(
                    image_state=noise_pred,
                    condition_state=next_prompt_states,
                ),
                patch_idx,
            )

        def end_step(step_index, _timestep):
            if (
                step_index == len(timesteps) - 1
                or (
                    (step_index + pipeline_warmup + 1) > num_warmup_steps
                    and (step_index + pipeline_warmup + 1)
                    % self.scheduler.order
                    == 0
                )
            ):
                progress_bar.update()

        hooks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=lambda: (
                torch.cat(patch_latents, dim=1)
                if is_pipeline_last_stage()
                else None
            ),
            interrupted_fn=lambda: self.interrupt,
            end_step_fn=end_step,
        )
        return PipeFusionAsyncDriver(
            transport=transport,
            output_cache=stage_output_cache,
            hooks=hooks,
            advance_patch=state.next_patch,
        ).run(timesteps)


@xFuserPipelineWrapperRegister.register(QwenImagePipeline)
class xFuserQwenImagePipeline(xFuserQwenImagePipelineBase):
    _diffusers_cls = QwenImagePipeline


@xFuserPipelineWrapperRegister.register(QwenImageEditPipeline)
class xFuserQwenImageEditPipeline(xFuserQwenImagePipelineBase):
    _diffusers_cls = QwenImageEditPipeline
    _is_edit = True

    def _prepare_image(self, image, height, width):
        first_image = image[0] if isinstance(image, list) else image
        if isinstance(first_image, torch.Tensor):
            image_size = (first_image.shape[-1], first_image.shape[-2])
        else:
            image_size = first_image.size
        calculated_width, calculated_height, _ = calculate_dimensions(
            1024 * 1024, image_size[0] / image_size[1]
        )
        height = height or calculated_height
        width = width or calculated_width
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of
        prompt_image = image
        if not (
            isinstance(image, torch.Tensor)
            and image.ndim > 1
            and image.size(1) == self.latent_channels
        ):
            image = self.image_processor.resize(
                image, calculated_height, calculated_width
            )
            prompt_image = image
            image = self.image_processor.preprocess(
                image, calculated_height, calculated_width
            ).unsqueeze(2)
        image_shape = (
            1,
            calculated_height // self.vae_scale_factor // 2,
            calculated_width // self.vae_scale_factor // 2,
        )
        return image, height, width, prompt_image, image_shape

    def _prepare_model_latents(
        self,
        image,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        return self.prepare_latents(
            image,
            batch_size,
            num_channels_latents,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
        )
