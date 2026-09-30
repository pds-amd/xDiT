"""Optional Cache-DiT configuration translation."""

import dataclasses
import json
import logging
from typing import Any, Dict, Optional

import torch.distributed as dist

from xfuser.model_executor.cache.presets import (
    DBCachePreset,
    PIPEFUSION_CACHE_PLAN_KEY,
    PipeFusionStaticMask,
)
from xfuser.model_executor.pipefusion.cache import build_pipefusion_static_mask

logger = logging.getLogger(__name__)

_XDIT_SCM_POLICIES = {
    "alternating": PipeFusionStaticMask.ALTERNATING_MIDDLE,
    "alternating_wide": PipeFusionStaticMask.WIDE_ALTERNATING_MIDDLE,
}
_XDIT_CONTEXT_STATIC_MASKS_KEY = "xdit_context_static_masks"
_XDIT_AUTO_STATIC_SCM_KEY = "xdit_auto_static_scm"


def is_rank0() -> bool:
    return (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_rank() == 0
    )


def import_cache_dit():
    try:
        from cache_dit import (
            BlockAdapter,
            DBCacheConfig,
            ForwardPattern,
            enable_cache,
        )

        return enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
    except ImportError as error:
        raise ImportError(
            "cache-dit is required for --cache_method dbcache. Install: "
            "pip install cache-dit or pip install 'xdit[cache-dit]'"
        ) from error


def build_calibrator_config(
    enable_encoder_calibrator: Optional[bool] = None,
) -> Optional[Any]:
    try:
        from cache_dit import TaylorSeerCalibratorConfig

        kwargs: Dict[str, Any] = {"taylorseer_order": 1}
        if enable_encoder_calibrator is not None:
            kwargs["enable_encoder_calibrator"] = enable_encoder_calibrator
        return TaylorSeerCalibratorConfig(**kwargs)
    except ImportError:
        if is_rank0():
            logger.warning(
                "TaylorSeerCalibratorConfig unavailable; running without "
                "a calibrator."
            )
        return None


def build_scm_mask(policy: Optional[str], num_steps: int) -> Optional[Any]:
    if not policy:
        return None
    if policy in _XDIT_SCM_POLICIES:
        return list(
            build_pipefusion_static_mask(_XDIT_SCM_POLICIES[policy], num_steps)
        )
    try:
        import cache_dit

        return cache_dit.steps_mask(mask_policy=policy, total_steps=num_steps)
    except (ImportError, AttributeError):
        if is_rank0():
            logger.warning("cache_dit.steps_mask unavailable; SCM ignored.")
        return None


def resolve_enable_separate_cfg(requested: bool) -> bool:
    if not requested:
        return False

    from xfuser.core.distributed import (
        get_classifier_free_guidance_world_size,
        get_pipeline_parallel_world_size,
    )

    return (
        get_classifier_free_guidance_world_size() == 1
        and get_pipeline_parallel_world_size() == 1
    )


def build_config(
    num_steps: int,
    preset_kwargs,
    cache_config_json: Optional[str],
    enable_separate_cfg: bool,
    dbcache_config_type: Any,
):
    """Build a Cache-DiT config and optional TaylorSeer calibrator."""
    overrides: Dict[str, Any] = {}
    share_pipefusion_decisions = None
    context_static_masks = None
    auto_static_scm = False
    if cache_config_json:
        try:
            overrides = json.loads(cache_config_json)
            if not isinstance(overrides, dict):
                raise TypeError("cache_config must be a JSON object")
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"--cache_config is not valid JSON: {error}"
            ) from error
    # PipeFusion stage-cache selection is xDiT planning metadata, not an
    # upstream Cache-DiT config field.
    overrides.pop(PIPEFUSION_CACHE_PLAN_KEY, None)
    context_static_masks = overrides.pop(_XDIT_CONTEXT_STATIC_MASKS_KEY, None)
    auto_static_scm = overrides.pop(_XDIT_AUTO_STATIC_SCM_KEY, False)
    if not isinstance(auto_static_scm, bool):
        raise ValueError(f"{_XDIT_AUTO_STATIC_SCM_KEY} must be a boolean.")
    if auto_static_scm and context_static_masks is not None:
        raise ValueError(
            f"{_XDIT_AUTO_STATIC_SCM_KEY} cannot be combined with "
            f"{_XDIT_CONTEXT_STATIC_MASKS_KEY}."
        )
    if context_static_masks is not None:
        if not isinstance(context_static_masks, dict):
            raise ValueError(
                f"{_XDIT_CONTEXT_STATIC_MASKS_KEY} must be an object mapping "
                "context prefixes to binary masks."
            )
        context_static_masks = {
            str(prefix): tuple(mask)
            for prefix, mask in context_static_masks.items()
        }
        for prefix, mask in context_static_masks.items():
            if not prefix or not mask or any(value not in (0, 1) for value in mask):
                raise ValueError(
                    f"{_XDIT_CONTEXT_STATIC_MASKS_KEY}[{prefix!r}] must be a "
                    "non-empty binary mask."
                )

    if isinstance(preset_kwargs, DBCachePreset):
        preset_fields = {field.name for field in dataclasses.fields(DBCachePreset)}
        preset_overrides = {
            key: value
            for key, value in overrides.items()
            if key in preset_fields
        }
        overrides = {
            key: value
            for key, value in overrides.items()
            if key not in preset_fields
        }
        preset = (
            dataclasses.replace(preset_kwargs, **preset_overrides)
            if preset_overrides
            else preset_kwargs
        )
        share_pipefusion_decisions = preset.share_pipefusion_decisions
        config_kwargs: Dict[str, Any] = {
            "Fn_compute_blocks": preset.Fn_compute_blocks,
            "Bn_compute_blocks": preset.Bn_compute_blocks,
            "residual_diff_threshold": preset.residual_diff_threshold,
            "max_warmup_steps": preset.max_warmup_steps,
            "max_cached_steps": preset.max_cached_steps,
        }
        if preset.enable_separate_cfg is not None:
            enable_separate_cfg = preset.enable_separate_cfg
        scm_mask = build_scm_mask(preset.scm_policy, num_steps)
        if scm_mask is not None:
            config_kwargs["steps_computation_policy"] = (
                preset.steps_computation_policy
            )
        calibrator_config = (
            None
            if preset.enable_taylorseer is False
            else build_calibrator_config(preset.enable_encoder_calibrator)
        )
    else:
        config_kwargs = dict(preset_kwargs or {})
        scm_mask = None
        calibrator_config = build_calibrator_config()

    config_kwargs["num_inference_steps"] = num_steps
    config_kwargs.update(overrides)
    if scm_mask is not None and "steps_computation_mask" not in config_kwargs:
        config_kwargs["steps_computation_mask"] = scm_mask

    valid_fields = {
        field.name for field in dataclasses.fields(dbcache_config_type)
    }
    enable_separate_cfg = resolve_enable_separate_cfg(
        config_kwargs.get("enable_separate_cfg", enable_separate_cfg)
    )
    if enable_separate_cfg:
        config_kwargs.setdefault("enable_separate_cfg", True)
    elif "enable_separate_cfg" in valid_fields:
        config_kwargs["enable_separate_cfg"] = False
    else:
        config_kwargs.pop("enable_separate_cfg", None)

    unknown = set(config_kwargs) - valid_fields
    if unknown:
        raise ValueError(
            "Unknown --cache_config keys for DBCacheConfig: "
            f"{sorted(unknown)}"
        )
    config = dbcache_config_type(
        **{
            key: value
            for key, value in config_kwargs.items()
            if key in valid_fields
        }
    )
    if share_pipefusion_decisions is not None:
        # This is an xDiT transport policy, not a Cache-DiT constructor
        # argument. Keep its upstream config contract unchanged.
        config._xdit_share_pipefusion_decisions = (
            share_pipefusion_decisions
        )
    if context_static_masks is not None:
        config._xdit_context_static_masks = context_static_masks
    if auto_static_scm:
        config._xdit_auto_static_scm = True
    return config, calibrator_config
