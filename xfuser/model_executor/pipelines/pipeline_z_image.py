"""PipeFusion pipeline wrapper for Z-Image and Z-Image-Turbo."""

import math
import os
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from diffusers import ZImagePipeline
from diffusers.pipelines.z_image.pipeline_output import ZImagePipelineOutput
from diffusers.pipelines.z_image.pipeline_z_image import (
    calculate_shift,
    get_default_z_image_sigmas,
    retrieve_timesteps,
)

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
from xfuser.model_executor.models.transformers.transformer_z_image import (  # noqa: F401
    xFuserZImagePipeFusionTransformerWrapper,
)

from .base_pipeline import xFuserPipelineBaseWrapper
from .register import xFuserPipelineWrapperRegister


@xFuserPipelineWrapperRegister.register(ZImagePipeline)
class xFuserZImagePipeline(xFuserPipelineBaseWrapper):
    """Z-Image denoising with spatial patches flowing across PP stages."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        engine_config: EngineConfig,
        cache_args: Dict = {},
        return_org_pipeline: bool = False,
        **kwargs,
    ):
        pipeline = ZImagePipeline.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        if return_org_pipeline:
            return pipeline
        return cls(pipeline, engine_config, cache_args)

    def _align_patch_metadata(self) -> None:
        """Balance height stripes on the transformer's 32-token boundary."""
        state = get_runtime_state()
        patch_size = self.transformer.config.all_patch_size
        if isinstance(patch_size, (list, tuple)):
            patch_size = patch_size[0]
        latent_width = state.input_config.width // state.vae_scale_factor
        token_width = latent_width // patch_size
        row_alignment = 32 // math.gcd(32, token_width)
        height_alignment = patch_size * row_alignment
        total_height = sum(state.pp_patches_height)
        if total_height % height_alignment:
            raise ValueError(
                "Z-Image latent height cannot be partitioned on a 32-token boundary"
            )
        units, remainder = divmod(
            total_height // height_alignment, state.num_pipeline_patch
        )
        heights = [
            (units + (index < remainder)) * height_alignment
            for index in range(state.num_pipeline_patch)
        ]
        starts = [sum(heights[:index]) for index in range(len(heights) + 1)]
        token_starts = [
            (start // patch_size) * token_width for start in starts
        ]
        state.pp_patches_height = heights
        state.pp_patches_start_idx_local = starts
        state.pp_patches_start_end_idx_global = [
            starts[index : index + 2] for index in range(len(heights))
        ]
        state.pp_patches_token_start_idx_local = token_starts
        state.pp_patches_token_start_end_idx_global = [
            token_starts[index : index + 2]
            for index in range(len(heights))
        ]
        state.pp_patches_token_num = [
            token_starts[index + 1] - token_starts[index]
            for index in range(len(heights))
        ]
        # set_input_parameters() sized the P2P buffers from the unaligned
        # layout. Rebuild them after replacing the authoritative patch metadata.
        state._reset_recv_buffer()

    @property
    def interrupt(self):
        return self._interrupt

    def _combine_cfg(
        self,
        model_out: torch.Tensor,
        guidance_scale: float,
        cfg_normalization: bool,
    ) -> torch.Tensor:
        cfg_size = get_classifier_free_guidance_world_size()
        if cfg_size == 2:
            pos, neg = get_cfg_group().all_gather(
                model_out, separate_tensors=True
            )
        else:
            pos, neg = model_out.chunk(2, dim=0)
        pred = pos.float() + guidance_scale * (pos.float() - neg.float())
        if cfg_normalization and float(cfg_normalization) > 0.0:
            normalized = []
            for positive, value in zip(pos.float(), pred):
                positive_norm = torch.linalg.vector_norm(positive)
                value_norm = torch.linalg.vector_norm(value)
                max_norm = positive_norm * float(cfg_normalization)
                normalized.append(
                    value * (max_norm / value_norm)
                    if value_norm > max_norm
                    else value
                )
            pred = torch.stack(normalized, dim=0)
        return pred

    def _backbone_forward(
        self,
        hidden_states: torch.Tensor,
        source_latents: torch.Tensor,
        prompt_embeds: List[torch.Tensor],
        timestep: torch.Tensor,
        do_cfg: bool,
        guidance_scale: float,
        cfg_normalization: bool,
        patch_start_height: int,
        full_image_tokens: int,
    ) -> torch.Tensor:
        cfg_size = get_classifier_free_guidance_world_size()
        local_cfg = do_cfg and cfg_size == 1
        source = source_latents.to(self.transformer.dtype)
        if local_cfg:
            source = source.repeat(2, 1, 1, 1)
        source_list = list(source.unsqueeze(2).unbind(dim=0))

        if is_pipeline_first_stage():
            hidden_states = source
        effective_batch = len(source_list)
        timestep_input = ((1000 - timestep.expand(effective_batch)) / 1000)
        output = self.transformer(
            source_list,
            timestep_input,
            prompt_embeds,
            hidden_states=None if is_pipeline_first_stage() else hidden_states,
            patch_start_height=patch_start_height,
            full_image_tokens=full_image_tokens,
            return_dict=False,
        )[0]
        if not is_pipeline_last_stage():
            return output

        model_out = torch.stack([value.float() for value in output], dim=0)
        if do_cfg:
            model_out = self._combine_cfg(
                model_out, guidance_scale, cfg_normalization
            )
        return -model_out.squeeze(2)

    def _sync_pipeline(
        self,
        latents,
        source_latents,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scales,
        cfg_normalization,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
        sync_only=False,
    ):
        state = get_runtime_state()
        state.set_patched_mode(patch_mode=False)
        full_image_tokens = sum(state.pp_patches_token_num)
        for i, timestep in enumerate(timesteps):
            if self.interrupt:
                continue
            if is_pipeline_last_stage():
                previous_latents = latents
            if get_pipeline_parallel_world_size() > 1 and not (
                is_pipeline_first_stage() and i == 0
            ):
                latents = get_pp_group().pipeline_recv()
            if is_pipeline_first_stage():
                source_latents = latents

            latents = self._backbone_forward(
                latents,
                source_latents,
                prompt_embeds,
                timestep,
                do_cfg,
                guidance_scales[i],
                cfg_normalization,
                patch_start_height=0,
                full_image_tokens=full_image_tokens,
            )
            if is_pipeline_last_stage():
                latents = self.scheduler.step(
                    latents.to(torch.float32),
                    timestep,
                    previous_latents,
                    return_dict=False,
                )[0]
                if callback_on_step_end is not None:
                    callback_kwargs = {
                        name: locals()[name]
                        for name in callback_on_step_end_tensor_inputs
                    }
                    callback_outputs = callback_on_step_end(
                        self, i, timestep, callback_kwargs
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
        return latents

    def _async_pipeline(
        self,
        latents,
        source_latents,
        prompt_embeds,
        timesteps,
        do_cfg,
        guidance_scales,
        cfg_normalization,
        num_warmup_steps,
        progress_bar,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=("latents",),
    ):
        if len(timesteps) == 0:
            return latents
        state = get_runtime_state()
        num_patches = state.num_pipeline_patch
        patch_latents = self._init_async_pipeline(
            len(timesteps),
            latents,
            state.runtime_config.warmup_steps,
            queue_receives=False,
        )
        layout = PipeFusionPatchLayout.from_runtime_state(state)
        source_patches = layout.split(
            source_latents
        )
        transport = PipeFusionTransport(
            get_pp_group(),
            first_stage=is_pipeline_first_stage(),
            num_steps=len(timesteps),
            num_patches=num_patches,
        )
        previous = [None] * num_patches if is_pipeline_last_stage() else None
        computation_mask = self._pipefusion_async_computation_mask(
            len(timesteps) + state.runtime_config.warmup_steps,
            state.runtime_config.warmup_steps,
        )
        stage_output_cache = self._pipefusion_stage_output_cache(
            computation_mask,
            num_patches,
        )
        full_image_tokens = sum(state.pp_patches_token_num)

        def prepare_patch(work, _timestep, received):
            patch_idx = work.patch_index
            if is_pipeline_last_stage():
                previous[patch_idx] = patch_latents[patch_idx]
            if received is not None:
                patch_latents[patch_idx] = received
            if is_pipeline_first_stage():
                source_patches[patch_idx] = patch_latents[patch_idx]
            return patch_latents[patch_idx]

        def forward_patch(work, timestep, prepared):
            patch_idx = work.patch_index
            return self._backbone_forward(
                prepared,
                source_patches[patch_idx],
                prompt_embeds,
                timestep,
                do_cfg,
                guidance_scales[work.step_index],
                cfg_normalization,
                patch_start_height=state.pp_patches_start_end_idx_global[
                    patch_idx
                ][0],
                full_image_tokens=full_image_tokens,
            )

        def commit_patch(work, timestep, output):
            patch_idx = work.patch_index
            patch_latents[patch_idx] = output
            if not is_pipeline_last_stage():
                return output
            patch_latents[patch_idx] = self.scheduler.step(
                output.to(torch.float32),
                timestep,
                previous[patch_idx],
                return_dict=False,
            )[0]
            if work.step_index == work.num_steps - 1:
                return None
            return patch_latents[patch_idx]

        def end_step(step_index, _timestep):
            if (
                step_index == len(timesteps) - 1
                or (
                    (step_index + state.runtime_config.warmup_steps + 1)
                    > num_warmup_steps
                    and (step_index + state.runtime_config.warmup_steps + 1)
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
                torch.cat(patch_latents, dim=2)
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

    @torch.no_grad()
    @xFuserPipelineBaseWrapper.enable_data_parallel
    @xFuserPipelineBaseWrapper.check_to_use_naive_forward
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        height = height or 1024
        width = width or 1024
        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale or width % vae_scale:
            raise ValueError(
                f"Height and width must be divisible by {vae_scale}."
            )
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        device = self._execution_device

        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)
        do_cfg = guidance_scale > 1
        if prompt_embeds is None or prompt is not None:
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_cfg,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )
        elif do_cfg and negative_prompt_embeds is None:
            raise ValueError(
                "negative_prompt_embeds is required for classifier-free guidance"
            )

        if num_images_per_prompt > 1:
            prompt_embeds = [
                value
                for value in prompt_embeds
                for _ in range(num_images_per_prompt)
            ]
            if do_cfg:
                negative_prompt_embeds = [
                    value
                    for value in negative_prompt_embeds
                    for _ in range(num_images_per_prompt)
                ]

        cfg_size = get_classifier_free_guidance_world_size()
        if do_cfg and cfg_size not in (1, 2):
            raise ValueError("Z-Image CFG supports CFG degree 1 or 2")
        if do_cfg:
            combined_prompts = prompt_embeds + negative_prompt_embeds
            if cfg_size == 2:
                branch_size = len(prompt_embeds)
                rank = get_classifier_free_guidance_rank()
                prompt_embeds = combined_prompts[
                    rank * branch_size : (rank + 1) * branch_size
                ]
            else:
                prompt_embeds = combined_prompts

        state = get_runtime_state()
        state.set_input_parameters(
            height=height,
            width=width,
            batch_size=batch_size,
            num_inference_steps=num_inference_steps,
            max_condition_sequence_length=max_sequence_length,
            split_text_embed_in_sp=False,
        )
        self._align_patch_metadata()
        if any(tokens % 32 for tokens in state.pp_patches_token_num):
            raise ValueError(
                "Each Z-Image PipeFusion patch must contain a multiple of 32 "
                "image tokens; adjust the resolution or number of pipeline patches."
            )
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            self.transformer.in_channels,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        source_latents = latents
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        if sigmas is None:
            sigmas = get_default_z_image_sigmas(num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        self.scheduler.set_begin_index(0)
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)
        t_norms = ((1000 - timesteps.float()) / 1000).tolist()
        pipeline_warmup = state.runtime_config.warmup_steps
        use_cfg_truncation = (
            do_cfg
            and cfg_truncation is not None
            and float(cfg_truncation) <= 1
        )

        def scales_for(values):
            if not use_cfg_truncation:
                return [guidance_scale] * len(values)
            return [
                0.0 if value > float(cfg_truncation) else guidance_scale
                for value in values
            ]

        if (
            callback_on_step_end is not None
            and get_pipeline_parallel_world_size() > 1
            and len(timesteps) > pipeline_warmup
        ):
            raise ValueError(
                "callback_on_step_end is not supported by the asynchronous "
                "Z-Image PipeFusion schedule"
            )
        common = dict(
            source_latents=source_latents,
            prompt_embeds=prompt_embeds,
            do_cfg=do_cfg,
            cfg_normalization=cfg_normalization,
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
                guidance_scales = scales_for(t_norms[:pipeline_warmup])
                latents = self._sync_pipeline(
                    latents=latents,
                    timesteps=timesteps[:pipeline_warmup],
                    guidance_scales=guidance_scales,
                    **common,
                )
                guidance_scales = scales_for(t_norms[pipeline_warmup:])
                latents = self._async_pipeline(
                    latents=latents,
                    timesteps=timesteps[pipeline_warmup:],
                    guidance_scales=guidance_scales,
                    **common,
                )
            else:
                guidance_scales = scales_for(t_norms)
                latents = self._sync_pipeline(
                    latents=latents,
                    timesteps=timesteps,
                    guidance_scales=guidance_scales,
                    sync_only=True,
                    **common,
                )

        image = None
        if output_type == "latent":
            image = latents
        elif latents is not None:
            def process(value):
                value = value.to(self.vae.dtype)
                return (
                    value / self.vae.config.scaling_factor
                ) + self.vae.config.shift_factor

            if (
                state.runtime_config.use_parallel_vae
                and state.parallel_config.vae_parallel_size > 0
            ):
                latents = self.gather_latents_for_vae(latents)
                if latents is not None:
                    latents = process(latents)
                self.send_to_vae_decode(latents)
            elif state.runtime_config.use_parallel_vae:
                latents = self.gather_broadcast_latents(latents)
                image = self.vae.decode(
                    process(latents), return_dict=False
                )[0]
            elif is_dp_last_group():
                image = self.vae.decode(
                    process(latents), return_dict=False
                )[0]
            if image is not None:
                image = self.image_processor.postprocess(
                    image, output_type=output_type
                )

        if not is_dp_last_group():
            return None
        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return ZImagePipelineOutput(images=image)
