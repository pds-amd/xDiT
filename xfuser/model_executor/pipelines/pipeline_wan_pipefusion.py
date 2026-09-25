"""PipeFusion for the Wan2.2 TI2V 5B image-to-video pipeline."""

import os
from typing import Any, Dict, List, Optional, Union

import torch
from diffusers import WanImageToVideoPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

from xfuser.config import EngineConfig
from xfuser.model_executor.pipefusion import (
    PipeFusionAsyncCallbacks,
    PipeFusionAsyncDriver,
    PipeFusionPatchLayout,
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
from xfuser.model_executor.models.transformers.transformer_wan import (  # noqa: F401
    xFuserWanPipeFusionTransformerWrapper,
)

from .base_pipeline import xFuserPipelineBaseWrapper


class xFuserWanTI2VPipeFusionPipeline(xFuserPipelineBaseWrapper):
    """Wan denoising with full-temporal-extent latent-height stripes."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        **kwargs,
    ):
        pipeline = WanImageToVideoPipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        return cls(pipeline, engine_config, cache_args)

    @property
    def interrupt(self):
        return self._interrupt

    def _model_input(self, latents, condition, first_frame_mask):
        return ((1 - first_frame_mask) * condition + first_frame_mask * latents).to(
            self.transformer.dtype
        )

    def _timestep(self, t, first_frame_mask, batch):
        _, p_h, p_w = self.transformer.config.patch_size
        values = (
            first_frame_mask[0][0][:, ::p_h, ::p_w] * t
        ).flatten()
        return values.unsqueeze(0).expand(batch, -1)

    def _backbone_forward(
        self,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        t,
        do_cfg,
        guidance_scale,
        pipeline_hidden_states=None,
        patch_start_height=0,
    ):
        cfg_size = get_classifier_free_guidance_world_size()
        local_cfg = do_cfg and cfg_size == 1
        model_input = self._model_input(latents, condition, first_frame_mask)
        if local_cfg:
            model_input = torch.cat([model_input, model_input], dim=0)
        timestep = self._timestep(t, first_frame_mask, model_input.shape[0])
        cache_key = (
            "uncond"
            if do_cfg and cfg_size == 2 and get_classifier_free_guidance_rank() == 0
            else "cond"
        )
        with self.transformer.cache_context(cache_key):
            output = self.transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pipeline_hidden_states=pipeline_hidden_states,
                patch_start_height=patch_start_height,
                return_dict=False,
            )[0]
        if not is_pipeline_last_stage():
            return output
        if do_cfg:
            if cfg_size == 2:
                negative, positive = get_cfg_group().all_gather(
                    output, separate_tensors=True
                )
            else:
                negative, positive = output.chunk(2, dim=0)
            output = negative + guidance_scale * (positive - negative)
        return output

    def _sync_pipeline(
        self,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scale,
        progress_bar,
        sync_only=False,
    ):
        state = get_runtime_state()
        state.set_patched_mode(False)
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue
            state.increment_step_counter()
            previous = latents if is_pipeline_last_stage() else None
            received = None
            if get_pipeline_parallel_world_size() > 1 and not (
                is_pipeline_first_stage() and i == 0
            ):
                received = get_pp_group().pipeline_recv()
            if is_pipeline_first_stage():
                latents = received if received is not None else latents
            output = self._backbone_forward(
                latents,
                condition,
                first_frame_mask,
                prompt_embeds,
                t,
                do_cfg,
                guidance_scale,
                pipeline_hidden_states=None if is_pipeline_first_stage() else received,
            )
            if is_pipeline_last_stage():
                latents = self.scheduler.step(
                    output, t, previous, return_dict=False
                )[0]
            else:
                latents = output
            progress_bar.update()
            if not (
                sync_only and is_pipeline_last_stage() and i == len(timesteps) - 1
            ):
                get_pp_group().pipeline_send(latents)
        return latents

    def _async_pipeline(
        self,
        latents,
        condition,
        first_frame_mask,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scale,
        progress_bar,
    ):
        if len(timesteps) == 0:
            return latents
        state = get_runtime_state()
        heights = state.pp_patches_height
        patches = self._init_async_pipeline(
            len(timesteps),
            latents,
            state.runtime_config.warmup_steps,
            split_sizes=heights,
            split_dim=3,
            queue_receives=False,
        )
        layout = PipeFusionPatchLayout.from_runtime_state(
            state,
            split_dim=3,
            split_sizes=heights,
            name="wan-height-stripes",
        )
        condition_patches = layout.split(condition)
        mask_patches = layout.split(first_frame_mask)
        transport = PipeFusionTransport(
            get_pp_group(),
            first_stage=is_pipeline_first_stage(),
            num_steps=len(timesteps),
            num_patches=state.num_pipeline_patch,
        )
        previous = [None] * state.num_pipeline_patch if is_pipeline_last_stage() else None
        computation_mask = self._pipefusion_async_computation_mask(
            len(timesteps) + state.runtime_config.warmup_steps,
            state.runtime_config.warmup_steps,
        )
        stage_output_cache = self._pipefusion_stage_output_cache(
            computation_mask,
            state.num_pipeline_patch,
        )
        step_outputs = []

        def begin_step(_step_idx, _timestep):
            state.increment_step_counter()
            step_outputs.clear()

        def prepare_patch(work, _timestep, received):
            patch_idx = work.patch_index
            if is_pipeline_last_stage():
                previous[patch_idx] = patches[patch_idx]
            if is_pipeline_first_stage() and received is not None:
                patches[patch_idx] = received
            model_input = (
                patches[patch_idx]
                if is_pipeline_first_stage() or is_pipeline_last_stage()
                else condition_patches[patch_idx]
            )
            return model_input, received

        def forward_patch(work, timestep, prepared):
            model_input, received = prepared
            patch_idx = work.patch_index
            return self._backbone_forward(
                model_input,
                condition_patches[patch_idx],
                mask_patches[patch_idx],
                prompt_embeds,
                timestep,
                do_cfg,
                guidance_scale,
                pipeline_hidden_states=(
                    None if is_pipeline_first_stage() else received
                ),
                patch_start_height=state.pp_patches_start_end_idx_global[
                    patch_idx
                ][0],
            )

        def commit_patch(work, _timestep, output):
            if is_pipeline_last_stage():
                step_outputs.append(output)
                return None
            patches[work.patch_index] = output
            return output

        def end_step(step_idx, timestep):
            nonlocal latents, patches
            deferred = None
            if is_pipeline_last_stage():
                latents = self.scheduler.step(
                    torch.cat(step_outputs, dim=3),
                    timestep,
                    torch.cat(previous, dim=3),
                    return_dict=False,
                )[0]
                patches = list(latents.split(heights, dim=3))
                if step_idx != len(timesteps) - 1:
                    deferred = list(enumerate(patches))
            progress_bar.update()
            return deferred

        hooks = PipeFusionAsyncCallbacks(
            prepare_patch_fn=prepare_patch,
            forward_patch_fn=forward_patch,
            commit_patch_fn=commit_patch,
            finalize_fn=lambda: (
                torch.cat(patches, dim=3)
                if is_pipeline_last_stage()
                else None
            ),
            begin_step_fn=begin_step,
            end_step_fn=end_step,
        )
        return PipeFusionAsyncDriver(
            transport=transport,
            output_cache=stage_output_cache,
            hooks=hooks,
            advance_patch=state.next_patch,
        ).run(timesteps)

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        image,
        prompt=None,
        negative_prompt=None,
        height=480,
        width=832,
        num_frames=81,
        num_inference_steps=50,
        guidance_scale=5.0,
        guidance_scale_2=None,
        num_videos_per_prompt=1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="np",
        return_dict=True,
        max_sequence_length=512,
        **kwargs,
    ):
        self.check_inputs(
            prompt,
            negative_prompt,
            image,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            None,
            ["latents"],
            guidance_scale_2,
        )
        if num_frames % self.vae_scale_factor_temporal != 1:
            num_frames = (
                num_frames // self.vae_scale_factor_temporal
                * self.vae_scale_factor_temporal
                + 1
            )
        self._guidance_scale = guidance_scale
        self._interrupt = False
        device = self._execution_device
        batch_size = (
            1 if isinstance(prompt, str) else
            len(prompt) if isinstance(prompt, list) else prompt_embeds.shape[0]
        )
        do_cfg = guidance_scale > 1
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype)
        cfg_size = get_classifier_free_guidance_world_size()
        if do_cfg and cfg_size not in (1, 2):
            raise ValueError("Wan TI2V PipeFusion supports CFG degree 1 or 2")
        if do_cfg:
            if cfg_size == 2:
                prompt_embeds = (
                    negative_prompt_embeds
                    if get_classifier_free_guidance_rank() == 0
                    else prompt_embeds
                )
            else:
                prompt_embeds = torch.cat(
                    [negative_prompt_embeds, prompt_embeds], dim=0
                )

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        image = self.video_processor.preprocess(
            image, height=height, width=width
        ).to(device, dtype=torch.float32)
        latents, condition, first_frame_mask = self.prepare_latents(
            image,
            batch_size * num_videos_per_prompt,
            self.vae.config.z_dim,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
            None,
        )
        state = get_runtime_state()
        state.set_video_input_parameters(
            height=height,
            width=width,
            num_frames=num_frames,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            split_text_embed_in_sp=False,
        )
        warmup = state.runtime_config.warmup_steps
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            if get_pipeline_parallel_world_size() > 1 and len(timesteps) > warmup:
                latents = self._sync_pipeline(
                    latents, condition, first_frame_mask, prompt_embeds,
                    timesteps[:warmup], do_cfg, guidance_scale, progress_bar,
                )
                latents = self._async_pipeline(
                    latents, condition, first_frame_mask, prompt_embeds,
                    timesteps[warmup:], do_cfg, guidance_scale, progress_bar,
                )
            else:
                latents = self._sync_pipeline(
                    latents, condition, first_frame_mask, prompt_embeds,
                    timesteps, do_cfg, guidance_scale, progress_bar, sync_only=True,
                )

        if latents is not None and is_pipeline_last_stage():
            latents = (
                (1 - first_frame_mask) * condition
                + first_frame_mask * latents
            )

        def process_latents(value):
            value = value.to(self.vae.dtype)
            mean = torch.tensor(self.vae.config.latents_mean).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(value.device, value.dtype)
            inv_std = (
                1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                )
            ).to(value.device, value.dtype)
            return value / inv_std + mean

        video = None
        if output_type == "latent":
            video = latents
        elif state.runtime_config.use_parallel_vae and state.parallel_config.vae_parallel_size > 0:
            latents = self.gather_latents_for_vae(latents)
            if latents is not None:
                latents = process_latents(latents)
            self.send_to_vae_decode(latents)
        elif state.runtime_config.use_parallel_vae:
            latents = self.gather_broadcast_latents(latents)
            video = self.vae.decode(process_latents(latents), return_dict=False)[0]
        elif is_dp_last_group() and latents is not None:
            video = self.vae.decode(process_latents(latents), return_dict=False)[0]
        if video is not None and output_type != "latent":
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        if not is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (video,)
        return WanPipelineOutput(frames=video)
