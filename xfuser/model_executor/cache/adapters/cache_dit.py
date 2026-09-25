"""
cache-dit adapter (optional dependency).
Provides DBCache step-caching via cache-dit's enable_cache() + BlockAdapter.

Model block layouts come from DBCacheSettings entries in ModelSettings.step_cache_config.
No model-specific dispatch in this file.
"""
import dataclasses
import json
import logging
import os
import types
from typing import Any, Dict, List, Optional

import torch
import torch.distributed as dist
from xfuser.model_executor.cache.presets import CacheDitAdapterConfig, DBCachePreset

logger = logging.getLogger(__name__)
_LOG_PIPEFUSION_CACHE_DECISIONS = (
    os.getenv("XDIT_LOG_PIPEFUSION_CACHE_DECISIONS") == "1"
)


def _is_rank0() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def _import_cache_dit():
    try:
        from cache_dit import enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
        return enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern
    except ImportError:
        raise ImportError(
            "cache-dit is required for --cache_method dbcache. "
            "Install: pip install cache-dit  or  pip install 'xdit[cache-dit]'"
        )


def _build_calibrator_config(enable_encoder_calibrator: Optional[bool] = None) -> Optional[Any]:
    """Build TaylorSeerCalibratorConfig (default calibrator)."""
    try:
        from cache_dit import TaylorSeerCalibratorConfig
        kwargs: Dict[str, Any] = {"taylorseer_order": 1}
        if enable_encoder_calibrator is not None:
            kwargs["enable_encoder_calibrator"] = enable_encoder_calibrator
        return TaylorSeerCalibratorConfig(**kwargs)
    except ImportError:
        if _is_rank0():
            logger.warning("TaylorSeerCalibratorConfig not available in this cache-dit version; running without calibrator")
        return None


def _build_scm_mask(
    policy: Optional[str],
    num_steps: int
) -> Optional[Any]:
    """Build steps_computation_mask from scm_policy field."""
    if not policy:
        return None
    if policy == "pipefusion":
        mask = [1] * num_steps
        first_cache_step = round((num_steps - 1) / 3)
        last_cache_step = round((num_steps - 1) * 5 / 6)
        for index in range(first_cache_step, last_cache_step + 1, 2):
            if 0 < index < num_steps - 1:
                mask[index] = 0
        return mask
    try:
        import cache_dit as cd
        return cd.steps_mask(mask_policy=policy, total_steps=num_steps)
    except (ImportError, AttributeError):
        if _is_rank0():
            logger.warning(
                "cache_dit.steps_mask not available; scm_policy ignored")
        return None


def _resolve_enable_separate_cfg(requested: bool) -> bool:
    """Resolve the requested cache mode against the active CFG topology."""
    if not requested:
        return False

    from xfuser.core.distributed import (
        get_classifier_free_guidance_world_size,
        get_pipeline_parallel_world_size,
    )

    # PipeFusion pipelines fold local true-CFG branches into one batch so each
    # stage performs one cacheable invocation per patch. Alternating cache-dit
    # contexts would compare adjacent patches/branches instead of timesteps.
    return (
        get_classifier_free_guidance_world_size() == 1
        and get_pipeline_parallel_world_size() == 1
    )


def _build_config(
    num_steps: int,
    preset_kwargs,
    cache_config_json: Optional[str],
    enable_separate_cfg: bool,
    DBCacheConfig: Any,
):
    """Build (DBCacheConfig, calibrator_config) from a DBCachePreset or plain dict.

    cache_config JSON overrides apply at the preset level first (so preset-only fields
    like scm_policy / enable_encoder_calibrator are overridable), then any remaining
    keys pass through as raw DBCacheConfig overrides.
    """
    overrides: Dict[str, Any] = {}
    if cache_config_json:
        try:
            overrides = json.loads(cache_config_json)
            if not isinstance(overrides, dict):
                raise TypeError("cache_config must be a JSON object")
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"--cache_config is not valid JSON: {e}") from e
    if isinstance(preset_kwargs, DBCachePreset):
        preset_fields = {f.name for f in dataclasses.fields(DBCachePreset)}
        preset_overrides = {k: v for k, v in overrides.items() if k in preset_fields}
        overrides = {k: v for k, v in overrides.items() if k not in preset_fields}
        p = dataclasses.replace(preset_kwargs, **preset_overrides) if preset_overrides else preset_kwargs
        config_kwargs: Dict[str, Any] = {
            "Fn_compute_blocks": p.Fn_compute_blocks,
            "Bn_compute_blocks": p.Bn_compute_blocks,
            "residual_diff_threshold": p.residual_diff_threshold,
            "max_warmup_steps": p.max_warmup_steps,
            "max_cached_steps": p.max_cached_steps,
        }
        if p.enable_separate_cfg is not None:
            enable_separate_cfg = p.enable_separate_cfg
        scm_mask = _build_scm_mask(p.scm_policy, num_steps)
        if scm_mask is not None:
            config_kwargs["steps_computation_policy"] = (
                p.steps_computation_policy
            )
        # enable_taylorseer=False disables the calibrator entirely (plain Fn-block
        # residual cache), used for "true" FBCache. None/True keeps default TaylorSeer.
        if p.enable_taylorseer is False:
            calibrator_config = None
        else:
            calibrator_config = _build_calibrator_config(p.enable_encoder_calibrator)
    else:
        config_kwargs = dict(preset_kwargs or {})
        scm_mask = None
        calibrator_config = _build_calibrator_config()

    config_kwargs["num_inference_steps"] = num_steps

    config_kwargs.update(overrides)

    if scm_mask is not None and "steps_computation_mask" not in config_kwargs:
        config_kwargs["steps_computation_mask"] = scm_mask

    valid_fields = {f.name for f in dataclasses.fields(DBCacheConfig)}
    enable_separate_cfg = _resolve_enable_separate_cfg(
        config_kwargs.get("enable_separate_cfg", enable_separate_cfg)
    )
    if enable_separate_cfg:
        config_kwargs.setdefault("enable_separate_cfg", True)
    elif "enable_separate_cfg" in valid_fields:
        # Preset and CLI overrides must not restore alternating two-call mode
        # when each CFG-parallel rank executes only one branch.
        config_kwargs["enable_separate_cfg"] = False
    else:
        config_kwargs.pop("enable_separate_cfg", None)

    unknown = set(config_kwargs) - valid_fields
    if unknown:
        raise ValueError(
            f"Unknown --cache_config keys for DBCacheConfig: {sorted(unknown)}"
        )
    config_kwargs = {k: v for k, v in config_kwargs.items()
                     if k in valid_fields}

    return DBCacheConfig(**config_kwargs), calibrator_config


def _unwrap_fsdp(transformer):
    """Return the real transformer when wrapped by FSDP1.

    shard_component uses FSDP1 for non-quantized models (e.g. Wan), wrapping the module
    in FullyShardedDataParallel. Block containers like `blocks` may not be accessible on
    the shell, causing _build_adapter block lookup to return None. FSDP2 (fully_shard)
    shards in place and keeps the original type, so only FSDP1 needs unwrapping.
    The wrapper.forward delegates to the inner module's (cache-patched) forward.
    """
    inner = getattr(transformer, "_fsdp_wrapped_module", None)
    if inner is not None:
        return inner
    if type(transformer).__name__ == "FullyShardedDataParallel":
        return getattr(transformer, "module", transformer)
    return transformer


def _build_adapter(
    transformer,
    pipe,
    adapter_cfg,
    BlockAdapter,
    ForwardPattern,
    db_config,
    calibrator_config,
    ParamsModifier,
):
    """Build a BlockAdapter from a CacheDitAdapterConfig.

    Skips block attrs that don't exist on the transformer so a single config can
    cover models where e.g. single_transformer_blocks is optional.
    All builders pass check_forward_pattern=False: the runner per-block-compiles the
    transformer before cache application, replacing blocks with OptimizedModule whose
    forward signature is (*args, **kwargs), breaking cache-dit's pattern introspection.
    We pin the correct ForwardPattern per model so the check is redundant.
    """
    found = []
    for attr, pat in adapter_cfg.blocks:
        blocks = getattr(transformer, attr, None)
        # PipeFusion keeps the declared ModuleList on every stage but empties
        # containers whose blocks belong to another stage. Passing an empty list
        # to cache-dit creates a cache pattern that can never satisfy Fn.
        if blocks is not None and len(blocks) > 0:
            found.append((attr, getattr(ForwardPattern, pat)))
    if not found:
        raise RuntimeError(
            f"CacheDitAdapterConfig specifies blocks {[a for a, _ in adapter_cfg.blocks]!r} "
            f"but none exist on {type(transformer).__name__}. Check DBCacheSettings.adapter."
        )
    attrs, patterns = zip(*found)
    params_modifiers = [
        ParamsModifier(
            cache_config=db_config,
            calibrator_config=calibrator_config,
        )
        for _ in attrs
    ]
    if len(attrs) == 1:
        return BlockAdapter(
            pipe=pipe, transformer=transformer,
            blocks=getattr(transformer, attrs[0]),
            blocks_name=attrs[0],
            forward_pattern=patterns[0],
            params_modifiers=params_modifiers,
            check_forward_pattern=False,
        )
    return BlockAdapter(
        pipe=pipe, transformer=transformer,
        blocks=[getattr(transformer, a) for a in attrs],
        blocks_name=list(attrs),
        forward_pattern=list(patterns),
        params_modifiers=params_modifiers,
        check_forward_pattern=False,
    )


_SP_SYNC_PATCHED = False


def _install_sp_can_cache_sync() -> None:
    """Force cache_dit's skip decision to agree bit-for-bit across all ranks.

    cache_dit's CachedContextManager.can_cache derives a bool from an AVG all_reduce over the default
    (world) process group; AVG is not bit-identical across RCCL ranks, so near-threshold the
    bool can flip per-rank. One divergent step desyncs collective counts (FSDP all-gather,
    ulysses all-to-all) -> NCCL hang. We wrap can_cache to broadcast the rank-0 result over
    the world group -- matching the group cache_dit reduces over, so it also covers the FSDP
    dimension (fully_shard) and pure-FSDP configs with ulysses=1, which an SP-only broadcast
    missed. Pure PipeFusion is handled locally below; the remaining paths run no data
    parallelism, so world-wide agreement is correct.

    Compile-path desync: torch.compile only traces paths that actually execute. During warmup
    max_warmup_steps forces can_cache=False, so only the full-compute graph is compiled. The
    first True result triggers compilation of the skip-path graph -- if ranks reach a
    distributed collective while one rank is compiling, the others time out. We insert a
    one-time world barrier at the warmup->cache transition (first True per instance) so all
    ranks synchronize before any collective fires inside the newly compiled path.
    Barrier fires at most once per CachedContextManager instance; steady-state overhead is zero.

    Idempotent; patched once.
    """
    from xfuser.core.distributed import (
        get_pipeline_parallel_world_size,
        get_sequence_parallel_world_size,
    )

    global _SP_SYNC_PATCHED
    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() <= 1
    ):
        return
    if _SP_SYNC_PATCHED:
        return
    try:
        from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
    except ImportError as e:
        raise ImportError(
            "Distributed dbcache requires cache-dit 1.5.x: "
            "the CachedContextManager synchronization hook is unavailable."
        ) from e
    from xfuser.core.distributed import get_world_group

    orig_can_cache = CachedContextManager.can_cache

    @torch.compiler.disable
    def can_cache(self, *args, **kwargs):
        pipefusion = get_pipeline_parallel_world_size() > 1
        sequence_parallel = get_sequence_parallel_world_size() > 1
        if pipefusion:
            # cache-dit's CachedBlocks_Pattern_Base reports every initialized
            # distributed run as parallelized, even when stages own disjoint
            # blocks. Its default-WORLD all-reduce would mix pipeline stages;
            # hybrid runs synchronize the resulting decision only within each
            # sequence-parallel replica group below.
            kwargs["parallelized"] = False
        result = orig_can_cache(self, *args, **kwargs)
        if not (dist.is_available() and dist.is_initialized()):
            return result
        if pipefusion and not sequence_parallel:
            if _LOG_PIPEFUSION_CACHE_DECISIONS:
                from xfuser.core.distributed import get_runtime_state

                runtime_state = get_runtime_state()
                context = self.get_context()
                logger.info(
                    "PipeFusion DBCache context=%s patch=%d step=%d "
                    "diff=%s hit=%s",
                    context.name,
                    runtime_state.pipeline_patch_idx,
                    context.get_current_step(),
                    self.get_current_step_residual_diff(),
                    result,
                )
            return result
        if pipefusion:
            from xfuser.core.distributed import get_sp_group

            sync_group = get_sp_group()
        else:
            sync_group = get_world_group()
        t = torch.tensor(
            [1 if result else 0], device=torch.cuda.current_device(), dtype=torch.int32)
        sync_group.broadcast(t, src=0)
        agreed = bool(t.item())
        if agreed and not getattr(self, '_xdit_cache_warmed', False):
            # First skip step: barrier so all ranks finish compiling the skip-path
            # graph before any collective fires inside it.
            dist.barrier(group=sync_group.device_group)
            self._xdit_cache_warmed = True
        return agreed

    CachedContextManager.can_cache = can_cache
    _SP_SYNC_PATCHED = True


def _is_parallelized_flag() -> bool:
    from xfuser.core.distributed import get_sequence_parallel_world_size

    # cache-dit uses this flag to all-reduce residual statistics. PipeFusion
    # stages hold different blocks and must make independent decisions from
    # their local stage inputs. Only replicated sequence-parallel peers reduce
    # their local statistic.
    return get_sequence_parallel_world_size() > 1


def _install_pipefusion_patch_contexts(transformer: torch.nn.Module) -> None:
    """Give each PipeFusion patch an independent Cache-DiT history.

    Cache-DiT normally advances one context on every transformer invocation.
    PipeFusion invokes a stage once per spatial patch, so one shared context
    compares adjacent patches instead of the same patch across timesteps.
    """
    from xfuser.core.distributed import get_runtime_state

    manager = getattr(transformer, "_context_manager", None)
    base_names = getattr(transformer, "_context_names", None)
    if manager is None or not base_names:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager/_context_names after "
            "enable_cache(); cannot install PipeFusion patch contexts."
        )
    if getattr(manager, "_xdit_pipefusion_patch_contexts_installed", False):
        return

    pipefusion_base_names = {
        base_name for base_name in base_names if isinstance(base_name, str)
    }
    if not pipefusion_base_names:
        raise RuntimeError(
            "Cache-DiT exposed no named cache contexts for PipeFusion."
        )
    num_patches = get_runtime_state().num_pipeline_patch
    if _is_rank0():
        logger.info(
            "Installed %d PipeFusion DBCache patch contexts across %d block groups.",
            num_patches * len(pipefusion_base_names),
            len(pipefusion_base_names),
        )

    original_reset_context = manager.reset_context

    @torch.compiler.disable
    def reset_patch_contexts(
        self,
        cached_context,
        *args,
        _base_names=pipefusion_base_names,
        _reset_context=original_reset_context,
        **kwargs,
    ):
        context = _reset_context(cached_context, *args, **kwargs)
        if isinstance(cached_context, str) and cached_context in _base_names:
            for patch_idx in range(get_runtime_state().num_pipeline_patch):
                _reset_context(
                    f"{cached_context}:pipefusion_patch_{patch_idx}",
                    *args,
                    **kwargs,
                )
        return context

    manager.reset_context = types.MethodType(reset_patch_contexts, manager)

    original_set_context = manager.set_context

    @torch.compiler.disable
    def set_patch_context(
        self,
        cached_context,
        *args,
        _base_names=pipefusion_base_names,
        _set_context=original_set_context,
        **kwargs,
    ):
        runtime_state = get_runtime_state()
        if (
            runtime_state.patch_mode
            and isinstance(cached_context, str)
            and cached_context in _base_names
        ):
            if runtime_state.pipeline_patch_idx >= runtime_state.num_pipeline_patch:
                raise RuntimeError(
                    "PipeFusion patch index exceeds the active patch layout."
                )
            cached_context = (
                f"{cached_context}:pipefusion_patch_"
                f"{runtime_state.pipeline_patch_idx}"
            )
        return _set_context(cached_context, *args, **kwargs)

    manager.set_context = types.MethodType(set_patch_context, manager)
    manager._xdit_pipefusion_patch_contexts_installed = True


def apply_cache_dit_cache(
    transformer: torch.nn.Module,
    num_steps: int,
    pipe: Optional[Any] = None,
    preset_kwargs: Optional[Dict[str, Any]] = None,
    cache_config: Optional[str] = None,
    adapter_config: Optional[CacheDitAdapterConfig] = None,
) -> torch.nn.Module:
    """Apply DBCache to a single transformer via cache-dit's enable_cache() with BlockAdapter."""
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    pipefusion = get_pipeline_parallel_world_size() > 1

    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()
    global_pipefusion_scm = (
        pipefusion
        and adapter_config is not None
        and adapter_config.pipefusion_global_scm_cache
    )
    if global_pipefusion_scm and isinstance(preset_kwargs, DBCachePreset):
        # Global output reuse is a static pipeline policy, not block-level
        # DBCache. Keep the ordinary preset intact for non-PipeFusion runs,
        # while still allowing explicit --cache_config overrides below.
        preset_kwargs = dataclasses.replace(
            preset_kwargs,
            scm_policy="pipefusion",
            steps_computation_policy="static",
            enable_taylorseer=False,
        )

    # Route on the unwrapped module (FSDP1 wrapper hides the real type), but return
    # the original transformer so the pipe keeps its FSDP wrapper.
    routing_transformer = _unwrap_fsdp(transformer)

    enable_separate_cfg = (
        adapter_config.enable_separate_cfg
        if adapter_config is not None
        else False
    )

    db_config, calibrator_config = _build_config(
        num_steps=num_steps,
        preset_kwargs=preset_kwargs,
        cache_config_json=cache_config,
        enable_separate_cfg=enable_separate_cfg,
        DBCacheConfig=DBCacheConfig,
    )
    if global_pipefusion_scm:
        computation_mask = getattr(
            db_config,
            "steps_computation_mask",
            None,
        )
        if computation_mask is None:
            raise ValueError(
                "PipeFusion global SCM cache requires an scm_policy or an "
                "explicit steps_computation_mask."
            )
        if pipe is None:
            raise ValueError(
                "PipeFusion global SCM cache requires the owning pipeline."
            )
        from xfuser.model_executor.pipefusion import (
            install_pipefusion_scm_mask,
            supports_pipefusion_stage_cache,
        )

        if not supports_pipefusion_stage_cache(pipe):
            raise ValueError(
                f"{type(pipe).__name__} does not implement PipeFusion global "
                "stage-output caching."
            )

        install_pipefusion_scm_mask(
            pipe,
            (int(value) for value in computation_mask),
        )
        if _is_rank0():
            logger.info(
                "Enabled PipeFusion global SCM output cache with mask %s.",
                "".join(map(str, pipe._xdit_pipefusion_scm_mask)),
            )
        return transformer

    if pipe is not None and hasattr(pipe, "_xdit_pipefusion_scm_mask"):
        delattr(pipe, "_xdit_pipefusion_scm_mask")

    from cache_dit import ParamsModifier

    _install_sp_can_cache_sync()
    routing_transformer._is_parallelized = _is_parallelized_flag()

    if adapter_config is not None:
        adapter = _build_adapter(
            routing_transformer,
            pipe,
            adapter_config,
            BlockAdapter,
            ForwardPattern,
            db_config,
            calibrator_config,
            ParamsModifier,
        )
    else:
        if _is_rank0():
            logger.warning(
                f"No CacheDitAdapterConfig for {type(routing_transformer).__name__}; "
                "falling back to auto=True. Set DBCacheSettings.adapter in the model runner."
            )
        adapter = BlockAdapter(pipe=pipe, auto=True)

    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config

    enable_cache(adapter, **enable_cache_kwargs)
    if pipefusion:
        _install_pipefusion_patch_contexts(routing_transformer)

    cls_name = type(routing_transformer).__name__
    calib_name = type(calibrator_config).__name__ if calibrator_config else "none"
    if _is_rank0():
        logger.info(
            f"Applied dbcache to {cls_name}: "
            f"F{db_config.Fn_compute_blocks}B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calib_name} "
            f"enable_separate_cfg={getattr(db_config, 'enable_separate_cfg', False)}"
        )
    return transformer


def apply_cache_dit_cache_multi(
    pipe: Any,
    num_steps: int,
    adapter_configs: List[CacheDitAdapterConfig],
    presets: List[DBCachePreset],
    cache_config: Optional[str] = None,
) -> None:
    """Apply DBCache to multiple transformers in one enable_cache() call using ParamsModifier.

    presets maps 1:1 to adapter_configs; each carries per-transformer cache params.
    """
    from xfuser.core.distributed import get_pipeline_parallel_world_size

    pipefusion = get_pipeline_parallel_world_size() > 1
    if pipefusion and any(
        cfg.pipefusion_global_scm_cache for cfg in adapter_configs
    ):
        raise ValueError(
            "PipeFusion global SCM output caching is not supported for "
            "multi-transformer pipelines."
        )
    enable_cache, DBCacheConfig, BlockAdapter, ForwardPattern = _import_cache_dit()
    _install_sp_can_cache_sync()
    from cache_dit import ParamsModifier

    if len(adapter_configs) != len(presets):
        raise ValueError(
            f"adapter_configs ({len(adapter_configs)}) and presets ({len(presets)}) must have same length"
        )

    # Resolve transformers from pipe via each adapter's transformer_attr
    transformers = []
    for cfg in adapter_configs:
        t = getattr(pipe, cfg.transformer_attr, None)
        if t is None:
            raise RuntimeError(
                f"apply_cache_dit_cache_multi: pipe has no attribute {cfg.transformer_attr!r}"
            )
        transformers.append(t)

    routing_transformers = [_unwrap_fsdp(t) for t in transformers]
    parallelized = _is_parallelized_flag()
    for rt in routing_transformers:
        rt._is_parallelized = parallelized

    cfg_flags = {c.enable_separate_cfg for c in adapter_configs}
    if len(cfg_flags) > 1:
        raise ValueError(
            f"All adapter_configs must agree on enable_separate_cfg; got {cfg_flags}"
        )
    enable_separate_cfg = adapter_configs[0].enable_separate_cfg

    # Build a full config per transformer so every per-preset field (compute blocks,
    # threshold, scm policy, calibrator), not just warmup/cached steps, pipes through.
    # cache_manager applies each ParamsModifier over the shared base via update() (non-None
    # fields win), so presets[0] is the base and presets[i] overrides for transformer i.
    # Each preset is built through _build_config so --cache_config JSON stays applied.
    configs = []
    calibrators = []
    for p in presets:
        c, cal = _build_config(
            num_steps=num_steps,
            preset_kwargs=p,
            cache_config_json=cache_config,
            enable_separate_cfg=enable_separate_cfg,
            DBCacheConfig=DBCacheConfig,
        )
        configs.append(c)
        calibrators.append(cal)
    db_config, calibrator_config = configs[0], calibrators[0]

    # Per-transformer block resolution and ParamsModifiers
    found_blocks = []
    found_attrs = []
    found_patterns = []
    params_modifiers = []
    for rt, cfg, c, cal in zip(routing_transformers, adapter_configs, configs, calibrators):
        for attr, pat_name in cfg.blocks:
            blocks_obj = getattr(rt, attr, None)
            if blocks_obj is not None and len(blocks_obj) > 0:
                found_blocks.append(blocks_obj)
                found_attrs.append(attr)
                found_patterns.append(getattr(ForwardPattern, pat_name))
                break
        else:
            raise RuntimeError(
                f"CacheDitAdapterConfig blocks {[a for a, _ in cfg.blocks]!r} "
                f"not found on {type(rt).__name__} (pipe.{cfg.transformer_attr})"
            )
        params_modifiers.append(ParamsModifier(cache_config=c, calibrator_config=cal))

    adapter = BlockAdapter(
        pipe=pipe,
        transformer=routing_transformers,
        blocks=found_blocks,
        blocks_name=found_attrs,
        forward_pattern=found_patterns,
        params_modifiers=params_modifiers,
        check_forward_pattern=False,
    )

    enable_cache_kwargs: Dict[str, Any] = {"cache_config": db_config}
    if calibrator_config is not None:
        enable_cache_kwargs["calibrator_config"] = calibrator_config

    enable_cache(adapter, **enable_cache_kwargs)
    if pipefusion:
        for transformer in routing_transformers:
            _install_pipefusion_patch_contexts(transformer)

    names = [type(rt).__name__ for rt in routing_transformers]
    calib_name = type(calibrator_config).__name__ if calibrator_config else "none"
    if _is_rank0():
        logger.info(
            f"Applied dbcache to [{', '.join(names)}] (multi): "
            f"F{db_config.Fn_compute_blocks}B{db_config.Bn_compute_blocks} "
            f"threshold={db_config.residual_diff_threshold} "
            f"calibrator={calib_name} "
            f"enable_separate_cfg={getattr(db_config, 'enable_separate_cfg', False)} "
            f"warmup_steps={[p.max_warmup_steps for p in presets]}"
        )
