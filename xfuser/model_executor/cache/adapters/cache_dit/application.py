"""Cache-DiT adapter construction and application."""

import dataclasses
import json
import logging
from typing import Any, Dict, List, Optional

import torch

from xfuser.model_executor.cache.presets import (
    CacheDitAdapterConfig,
    DBCachePreset,
    PIPEFUSION_CACHE_PLAN_KEY,
    PipeFusionCachePlan,
    PipeFusionCacheUnit,
)

from .config import build_config, import_cache_dit, is_rank0
from .context import (
    install_cache_dit_phase_tracing,
    install_cache_decision_sync,
    install_pipefusion_patch_contexts,
    is_parallelized_flag,
)

logger = logging.getLogger(__name__)


def unwrap_fsdp(transformer):
    """Return the underlying module for FSDP1 block discovery."""
    inner = getattr(transformer, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    if type(transformer).__name__ == "FullyShardedDataParallel":
        return getattr(transformer, "module", transformer)
    return transformer


def build_adapter(
    transformer,
    pipe,
    adapter_config: CacheDitAdapterConfig,
    block_adapter_type,
    forward_pattern_type,
    db_config,
    calibrator_config,
    params_modifier_type,
):
    """Build an adapter for the non-empty local block groups."""
    found = []
    for attribute, pattern_name in adapter_config.blocks:
        blocks = getattr(transformer, attribute, None)
        if blocks is not None and len(blocks) > 0:
            found.append((attribute, getattr(forward_pattern_type, pattern_name)))
    if not found:
        raise RuntimeError(
            "CacheDitAdapterConfig specifies blocks "
            f"{[name for name, _ in adapter_config.blocks]!r} but none exist "
            f"on {type(transformer).__name__}."
        )

    attributes, patterns = zip(*found)
    params_modifiers = []
    for attribute in attributes:
        local_blocks = len(getattr(transformer, attribute))
        local_config = dataclasses.replace(
            db_config,
            Fn_compute_blocks=min(db_config.Fn_compute_blocks, local_blocks),
            Bn_compute_blocks=min(
                db_config.Bn_compute_blocks,
                max(local_blocks - db_config.Fn_compute_blocks, 0),
            ),
        )
        params_modifiers.append(
            params_modifier_type(
                cache_config=local_config,
                calibrator_config=calibrator_config,
            )
        )
    kwargs = {
        "pipe": pipe,
        "transformer": transformer,
        "blocks": (
            getattr(transformer, attributes[0])
            if len(attributes) == 1
            else [getattr(transformer, attribute) for attribute in attributes]
        ),
        "blocks_name": attributes[0] if len(attributes) == 1 else list(attributes),
        "forward_pattern": patterns[0] if len(patterns) == 1 else list(patterns),
        "params_modifiers": params_modifiers,
        "check_forward_pattern": False,
    }
    return block_adapter_type(**kwargs)


def _validate_stage_cache_overrides(cache_config: Optional[str]) -> None:
    if not cache_config:
        return
    try:
        overrides = json.loads(cache_config)
        if not isinstance(overrides, dict):
            raise TypeError("cache_config must be a JSON object")
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(
            f"--cache_config is not valid JSON: {error}"
        ) from error
    overrides.pop(PIPEFUSION_CACHE_PLAN_KEY, None)
    policy_keys = {
        "scm_policy",
        "steps_computation_policy",
        "steps_computation_mask",
    }
    attempted = policy_keys & set(overrides)
    if attempted:
        raise ValueError(
            "PipeFusion stage-payload cache policy is model-owned and does "
            f"not allow --cache_config to override {sorted(attempted)}."
        )


def _configure_pipefusion_stage_tail_cache(
    db_config,
    plan: PipeFusionCachePlan,
    num_steps: int,
) -> None:
    """Configure Cache-DiT Mn reuse plus a static Bn refinement tail."""
    if plan.tail_compute_blocks is None:
        return
    from xfuser.core.distributed import get_runtime_state
    from xfuser.model_executor.pipefusion import (
        build_pipefusion_static_mask,
        normalize_pipefusion_scm_mask,
    )

    warmup_steps = get_runtime_state().runtime_config.warmup_steps
    mask = normalize_pipefusion_scm_mask(
        build_pipefusion_static_mask(plan.static_mask, num_steps),
        num_steps,
    )
    async_mask = mask[warmup_steps:]
    if not async_mask:
        raise ValueError(
            "PipeFusion stage-tail cache requires at least one async step."
        )
    # Each patch context starts when the async suffix starts. Its Cache-DiT
    # step index is therefore relative to that suffix, not global denoising.
    async_mask = (1,) + async_mask[1:]
    # Cache-DiT requires at least one Fn block; it is the stable probe/base
    # state to which the cached Mn residual is applied.
    db_config.Fn_compute_blocks = 1
    db_config.Bn_compute_blocks = plan.tail_compute_blocks
    db_config.steps_computation_mask = list(async_mask)
    db_config.steps_computation_policy = "static"


def apply_cache_dit_cache(
    transformer: torch.nn.Module,
    num_steps: int,
    pipe: Optional[Any] = None,
    preset_kwargs: Optional[Dict[str, Any]] = None,
    cache_config: Optional[str] = None,
    adapter_config: Optional[CacheDitAdapterConfig] = None,
    pipefusion_cache_plan: Optional[PipeFusionCachePlan] = None,
) -> torch.nn.Module:
    """Apply one Cache-DiT adapter or a model-owned PP stage cache plan."""
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    pipefusion = get_pipeline_parallel_world_size() > 1
    stage_payload_plan = (
        pipefusion
        and pipefusion_cache_plan is not None
        and pipefusion_cache_plan.unit is not PipeFusionCacheUnit.BLOCK_LOCAL
        and pipefusion_cache_plan.tail_compute_blocks is None
    )
    if stage_payload_plan:
        _validate_stage_cache_overrides(cache_config)
        if pipe is None:
            raise ValueError(
                "PipeFusion stage-payload caching requires the owning pipeline."
            )
        from xfuser.model_executor.pipefusion import (
            build_pipefusion_static_mask,
            install_pipefusion_cache_plan,
            supports_pipefusion_stage_cache,
        )

        if not supports_pipefusion_stage_cache(pipe):
            raise ValueError(
                f"{type(pipe).__name__} does not implement PipeFusion "
                "stage-payload caching."
            )
        install_pipefusion_cache_plan(pipe, pipefusion_cache_plan)
        if is_rank0():
            mask = build_pipefusion_static_mask(
                pipefusion_cache_plan.static_mask,
                num_steps,
            )
            logger.info(
                "Enabled PipeFusion %s cache with mask %s.",
                pipefusion_cache_plan.unit.value,
                "".join(map(str, mask)),
            )
        return transformer

    routing_transformer = unwrap_fsdp(transformer)
    enable_cache, dbcache_config_type, block_adapter_type, forward_pattern_type = (
        import_cache_dit()
    )
    install_cache_dit_phase_tracing()
    enable_separate_cfg = (
        adapter_config.enable_separate_cfg if adapter_config is not None else False
    )
    db_config, calibrator_config = build_config(
        num_steps=num_steps,
        preset_kwargs=preset_kwargs,
        cache_config_json=cache_config,
        enable_separate_cfg=enable_separate_cfg,
        dbcache_config_type=dbcache_config_type,
    )
    if pipefusion and pipefusion_cache_plan is not None:
        _configure_pipefusion_stage_tail_cache(
            db_config,
            pipefusion_cache_plan,
            num_steps,
        )
    if pipe is not None and hasattr(pipe, "_xdit_pipefusion_cache_plan"):
        delattr(pipe, "_xdit_pipefusion_cache_plan")

    from cache_dit import ParamsModifier

    routing_transformer._is_parallelized = is_parallelized_flag()
    if adapter_config is not None:
        adapter = build_adapter(
            routing_transformer,
            pipe,
            adapter_config,
            block_adapter_type,
            forward_pattern_type,
            db_config,
            calibrator_config,
            ParamsModifier,
        )
    else:
        if is_rank0():
            logger.warning(
                "No CacheDitAdapterConfig for %s; falling back to auto=True.",
                type(routing_transformer).__name__,
            )
        adapter = block_adapter_type(pipe=pipe, auto=True)

    enable_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_kwargs["calibrator_config"] = calibrator_config
    enable_cache(adapter, **enable_kwargs)
    routing_transformer._xdit_share_pipefusion_decisions = bool(
        getattr(db_config, "_xdit_share_pipefusion_decisions", False)
    )
    routing_transformer._xdit_context_static_masks = getattr(
        db_config, "_xdit_context_static_masks", None
    )
    routing_transformer._xdit_auto_static_scm = bool(
        getattr(db_config, "_xdit_auto_static_scm", False)
    )
    install_cache_decision_sync(routing_transformer)
    if pipefusion:
        install_pipefusion_patch_contexts(routing_transformer)
    if is_rank0():
        logger.info(
            "Applied dbcache to %s: F%sB%s threshold=%s calibrator=%s "
            "enable_separate_cfg=%s",
            type(routing_transformer).__name__,
            db_config.Fn_compute_blocks,
            db_config.Bn_compute_blocks,
            db_config.residual_diff_threshold,
            type(calibrator_config).__name__ if calibrator_config else "none",
            getattr(db_config, "enable_separate_cfg", False),
        )
    return transformer


def apply_cache_dit_cache_multi(
    pipe: Any,
    num_steps: int,
    adapter_configs: List[CacheDitAdapterConfig],
    presets: List[DBCachePreset],
    cache_config: Optional[str] = None,
    pipefusion_cache_plan: Optional[PipeFusionCachePlan] = None,
) -> None:
    """Apply one coordinated Cache-DiT adapter to multiple transformers."""
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    pipefusion = get_pipeline_parallel_world_size() > 1
    if (
        pipefusion
        and pipefusion_cache_plan is not None
        and pipefusion_cache_plan.unit is not PipeFusionCacheUnit.BLOCK_LOCAL
        and pipefusion_cache_plan.tail_compute_blocks is None
    ):
        raise ValueError(
            "PipeFusion stage-payload caching is not supported for "
            "multi-transformer pipelines."
        )
    if len(adapter_configs) != len(presets):
        raise ValueError(
            "adapter_configs and presets must have the same length."
        )

    enable_cache, dbcache_config_type, block_adapter_type, forward_pattern_type = (
        import_cache_dit()
    )
    install_cache_dit_phase_tracing()
    from cache_dit import ParamsModifier

    transformers = []
    for config in adapter_configs:
        transformer = getattr(pipe, config.transformer_attr, None)
        if transformer is None:
            raise RuntimeError(
                "apply_cache_dit_cache_multi: pipe has no attribute "
                f"{config.transformer_attr!r}."
            )
        transformers.append(unwrap_fsdp(transformer))

    for transformer in transformers:
        transformer._is_parallelized = is_parallelized_flag()
    cfg_flags = {config.enable_separate_cfg for config in adapter_configs}
    if len(cfg_flags) != 1:
        raise ValueError(
            "All adapter_configs must agree on enable_separate_cfg; got "
            f"{cfg_flags}."
        )
    configs, calibrators = zip(
        *[
            build_config(
                num_steps=num_steps,
                preset_kwargs=preset,
                cache_config_json=cache_config,
                enable_separate_cfg=adapter_configs[0].enable_separate_cfg,
                dbcache_config_type=dbcache_config_type,
            )
            for preset in presets
        ]
    )
    if pipefusion and pipefusion_cache_plan is not None:
        for db_config in configs:
            _configure_pipefusion_stage_tail_cache(
                db_config,
                pipefusion_cache_plan,
                num_steps,
            )

    found_blocks, found_attrs, found_patterns, modifiers = [], [], [], []
    for transformer, config, db_config, calibrator in zip(
        transformers,
        adapter_configs,
        configs,
        calibrators,
    ):
        for attribute, pattern_name in config.blocks:
            blocks = getattr(transformer, attribute, None)
            if blocks is not None and len(blocks) > 0:
                found_blocks.append(blocks)
                found_attrs.append(attribute)
                found_patterns.append(
                    getattr(forward_pattern_type, pattern_name)
                )
                break
        else:
            raise RuntimeError(
                f"CacheDitAdapterConfig blocks {config.blocks!r} not found "
                f"on {type(transformer).__name__}."
            )
        modifiers.append(
            ParamsModifier(cache_config=db_config, calibrator_config=calibrator)
        )

    adapter = block_adapter_type(
        pipe=pipe,
        transformer=transformers,
        blocks=found_blocks,
        blocks_name=found_attrs,
        forward_pattern=found_patterns,
        params_modifiers=modifiers,
        check_forward_pattern=False,
    )
    enable_kwargs: Dict[str, Any] = {"cache_config": configs[0]}
    if calibrators[0] is not None:
        enable_kwargs["calibrator_config"] = calibrators[0]
    enable_cache(adapter, **enable_kwargs)
    for transformer, config in zip(transformers, configs):
        transformer._xdit_share_pipefusion_decisions = bool(
            getattr(config, "_xdit_share_pipefusion_decisions", False)
        )
        install_cache_decision_sync(transformer)
        if pipefusion:
            install_pipefusion_patch_contexts(transformer)
