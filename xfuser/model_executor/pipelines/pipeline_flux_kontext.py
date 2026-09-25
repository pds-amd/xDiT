"""PipeFusion pipeline wrapper for FLUX.1-Kontext.

The denoising schedule is shared with :mod:`pipeline_flux`; this wrapper only
adds Kontext's reference-image tokens to each corresponding PipeFusion patch.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from diffusers import FluxKontextPipeline
from diffusers.pipelines.flux.pipeline_flux_kontext import (
    PREFERRED_KONTEXT_RESOLUTIONS,
)
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from xfuser.config import EngineConfig
from xfuser.core.distributed import (
    get_pipeline_parallel_world_size,
    get_runtime_state,
    is_dp_last_group,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
)

from .base_pipeline import xFuserPipelineBaseWrapper
from .pipeline_flux import xFuserFluxPipeline
from .register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(FluxKontextPipeline)
class xFuserFluxKontextPipeline(xFuserFluxPipeline):
    """FLUX.1 PipeFusion with stage-local Kontext image conditioning."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = FluxKontextPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    def _set_image_conditioning(
        self,
        image_latents: Optional[torch.Tensor],
        image_ids: Optional[torch.Tensor],
    ) -> None:
        self._kontext_image_latents = image_latents
        self._kontext_image_ids = image_ids
        if image_latents is None:
            self._kontext_patch_image_latents = None
            self._kontext_patch_image_ids = None
            self._kontext_reference_offsets = None
            return

        num_patches = get_runtime_state().num_pipeline_patch
        self._kontext_patch_image_latents = tuple(
            torch.tensor_split(image_latents, num_patches, dim=1)
        )
        self._kontext_patch_image_ids = tuple(
            torch.tensor_split(image_ids, num_patches, dim=0)
        )
        lengths = [
            patch.shape[1] for patch in self._kontext_patch_image_latents
        ]
        self._kontext_reference_offsets = [
            sum(lengths[:index]) for index in range(len(lengths) + 1)
        ]

    def _current_image_conditioning(self):
        if self._kontext_image_latents is None:
            return None, None
        runtime_state = get_runtime_state()
        if runtime_state.patch_mode:
            patch_idx = runtime_state.pipeline_patch_idx
            return (
                self._kontext_patch_image_latents[patch_idx],
                self._kontext_patch_image_ids[patch_idx],
            )
        return self._kontext_image_latents, self._kontext_image_ids

    def _backbone_forward(
        self,
        latents: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids,
        latent_image_ids,
        guidance,
        t: Union[float, torch.Tensor],
    ):
        image_latents, image_ids = self._current_image_conditioning()
        target_tokens = latent_image_ids.shape[0]

        # The first stage embeds target and reference tokens together. Later
        # stages receive that combined stream from their predecessor.
        hidden_states = latents
        if image_latents is not None and is_pipeline_first_stage():
            hidden_states = torch.cat([hidden_states, image_latents], dim=1)
        if image_ids is not None:
            latent_image_ids = torch.cat([latent_image_ids, image_ids], dim=0)

        timestep = t.expand(hidden_states.shape[0]).to(hidden_states.dtype)
        attention_kwargs = dict(self.joint_attention_kwargs or {})
        if image_latents is not None:
            if get_runtime_state().patch_mode:
                patch_idx = get_runtime_state().pipeline_patch_idx
                reference_start = self._kontext_reference_offsets[patch_idx]
                reference_end = self._kontext_reference_offsets[patch_idx + 1]
            else:
                reference_start = 0
                reference_end = self._kontext_image_latents.shape[1]
            attention_kwargs.update(
                pipefusion_reference_tokens=image_latents.shape[1],
                pipefusion_reference_start=reference_start,
                pipefusion_reference_end=reference_end,
            )

        ret = self.transformer(
            hidden_states=hidden_states,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        if self.engine_config.parallel_config.dit_parallel_size > 1:
            noise_pred, encoder_hidden_states = ret
        else:
            noise_pred, encoder_hidden_states = ret, None

        # Reference-token predictions are conditioning-only. Cropping on the
        # last stage preserves the generated-latent scheduler state and keeps
        # intermediate stages' complete image stream intact.
        if image_latents is not None and is_pipeline_last_stage():
            noise_pred = noise_pred[:, :target_tokens]
        return noise_pred, encoder_hidden_states

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.check_model_parallel_state(
        cfg_parallel_available=False
    )
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        image=None,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt=None,
        negative_prompt_2=None,
        true_cfg_scale: float = 1.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        ip_adapter_image=None,
        ip_adapter_image_embeds=None,
        negative_ip_adapter_image=None,
        negative_ip_adapter_image_embeds=None,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        max_area: int = 1024**2,
        _auto_resize: bool = True,
        **kwargs,
    ):
        has_negative_condition = (
            negative_prompt is not None
            or negative_prompt_embeds is not None
            or negative_pooled_prompt_embeds is not None
            or negative_ip_adapter_image is not None
            or negative_ip_adapter_image_embeds is not None
        )
        if true_cfg_scale > 1 and has_negative_condition:
            raise ValueError("True CFG is not supported with Kontext PipeFusion.")

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)
        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        self.check_inputs(
            prompt,
            prompt_2,
            height,
            width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = self._execution_device

        get_runtime_state().set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=get_pipeline_parallel_world_size() == 1,
        )

        lora_scale = (
            self.joint_attention_kwargs.get("scale")
            if self.joint_attention_kwargs is not None
            else None
        )
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        if image is not None and not (
            isinstance(image, torch.Tensor)
            and image.ndim > 1
            and image.size(1) == self.latent_channels
        ):
            first_image = image[0] if isinstance(image, list) else image
            image_height, image_width = self.image_processor.get_default_height_width(
                first_image
            )
            if _auto_resize:
                image_aspect_ratio = image_width / image_height
                _, image_width, image_height = min(
                    (abs(image_aspect_ratio - w / h), w, h)
                    for w, h in PREFERRED_KONTEXT_RESOLUTIONS
                )
            image_width = image_width // multiple_of * multiple_of
            image_height = image_height // multiple_of * multiple_of
            image = self.image_processor.resize(
                image, image_height, image_width
            )
            image = self.image_processor.preprocess(
                image, image_height, image_width
            )

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents, latent_ids, image_ids = self.prepare_latents(
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
        self._set_image_conditioning(image_latents, image_ids)

        sigmas = (
            np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
            if sigmas is None
            else sigmas
        )
        image_seq_len = latents.shape[1]
        from diffusers.pipelines.flux.pipeline_flux import (
            calculate_shift,
            retrieve_timesteps,
        )

        mu = calculate_shift(
            image_seq_len,
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
        guidance = (
            torch.full(
                [latents.shape[0]],
                guidance_scale,
                device=device,
                dtype=torch.float32,
            )
            if self.transformer.config.guidance_embeds
            else None
        )

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
            if self._joint_attention_kwargs is None:
                self._joint_attention_kwargs = {}
            self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(0)
        pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
        async_computation_mask = self._pipefusion_async_computation_mask(
            len(timesteps),
            pipeline_warmup_steps,
        )
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            if (
                get_pipeline_parallel_world_size() > 1
                and len(timesteps) > pipeline_warmup_steps
            ):
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps[:pipeline_warmup_steps],
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
                latents = self._async_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps[pipeline_warmup_steps:],
                    computation_mask=async_computation_mask,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                )
            else:
                latents = self._sync_pipeline(
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    latent_image_ids=latent_ids,
                    guidance=guidance,
                    timesteps=timesteps,
                    num_warmup_steps=num_warmup_steps,
                    progress_bar=progress_bar,
                    callback_on_step_end=callback_on_step_end,
                    callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                    sync_only=True,
                )

        image = None
        if output_type == "latent":
            image = latents
        else:
            def process_latents(value):
                value = self._unpack_latents(
                    value, height, width, self.vae_scale_factor
                )
                return (
                    value / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor

            runtime_state = get_runtime_state()
            if (
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
                image = self.vae.decode(latents, return_dict=False)[0]
            elif latents is not None and is_dp_last_group():
                latents = process_latents(latents)
                image = self.vae.decode(latents, return_dict=False)[0]

            if image is not None:
                image = self.image_processor.postprocess(
                    image, output_type=output_type
                )

        self._set_image_conditioning(None, None)
        if not is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return FluxPipelineOutput(images=image)
