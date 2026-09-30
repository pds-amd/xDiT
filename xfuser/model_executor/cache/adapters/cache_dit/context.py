"""Distributed and PipeFusion Cache-DiT context integration."""

import logging
import os
import types

import torch
import torch.distributed as dist
from torch.profiler import record_function

from .config import is_rank0

logger = logging.getLogger(__name__)


def install_cache_dit_phase_tracing() -> bool:
    """Annotate Cache-DiT 1.5.2 block phases for PipeFusion profiling."""
    try:
        import cache_dit
        from cache_dit.caching.cache_blocks.pattern_base import (
            CachedBlocks_Pattern_Base,
        )
    except ImportError:
        return False
    if getattr(cache_dit, "__version__", None) != "1.5.2":
        return False
    if getattr(
        CachedBlocks_Pattern_Base,
        "_xdit_phase_tracing_installed",
        False,
    ):
        return True

    for method_name, phase_name in (
        ("call_Fn_blocks", "fn"),
        ("call_Mn_blocks", "middle"),
        ("call_Bn_blocks", "tail"),
    ):
        original = getattr(CachedBlocks_Pattern_Base, method_name)

        def traced(self, *args, _original=original, _phase=phase_name, **kwargs):
            with record_function(f"xdit::cache_dit.{_phase}"):
                return _original(self, *args, **kwargs)

        setattr(CachedBlocks_Pattern_Base, method_name, traced)
    CachedBlocks_Pattern_Base._xdit_phase_tracing_installed = True
    return True


def is_parallelized_flag() -> bool:
    from xfuser.core.distributed import get_sequence_parallel_world_size

    return get_sequence_parallel_world_size() > 1


def _context_static_reuse_decision(context_manager, masks):
    """Return a configured static reuse decision for one cache context."""
    if not masks:
        return None
    context_name = getattr(context_manager.get_context(), "name", "")
    for prefix, mask in masks.items():
        if context_name == prefix or context_name.startswith(f"{prefix}_"):
            step = context_manager.get_current_step()
            return mask[step % len(mask)] == 0
    return None


def _context_mask_key(context_name: str) -> str:
    """Normalize Cache-DiT's object-suffixed context name."""
    prefix, separator, suffix = context_name.rpartition("_")
    return prefix if separator and suffix.isdigit() else context_name


def _derive_auto_static_masks(auto_state, context_manager) -> bool:
    """Freeze one calibration inference's synchronized decisions as masks."""
    context = context_manager.get_context()
    num_steps = getattr(context.cache_config, "num_inference_steps", None)
    if not num_steps:
        return False
    key = _context_mask_key(getattr(context, "name", ""))
    if not key:
        return False
    decisions = auto_state["decisions"]
    decisions.setdefault(key, {})[context_manager.get_current_step()] = (
        auto_state["current_decision"]
    )
    expected = auto_state["expected_contexts"]
    if not expected or not all(
        num_steps - 1 in decisions.get(name, {}) for name in expected
    ):
        return False
    auto_state["masks"] = {
        name: tuple(
            0 if decisions[name].get(step, False) else 1
            for step in range(num_steps)
        )
        for name in expected
    }
    return True


def install_cache_decision_sync(transformer: torch.nn.Module) -> None:
    """Synchronize adaptive Cache-DiT decisions across replicated ranks."""
    from xfuser.core.distributed import (
        get_pipeline_parallel_world_size,
        get_sequence_parallel_world_size,
    )

    if (
        not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() <= 1
    ):
        return
    manager = getattr(transformer, "_context_manager", None)
    if manager is None:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager after enable_cache; "
            "cannot synchronize cache decisions."
        )
    if getattr(manager, "_xdit_cache_decision_sync_installed", False):
        return

    from xfuser.core.distributed import get_world_group

    original_can_cache = manager.can_cache
    context_static_masks = getattr(
        transformer, "_xdit_context_static_masks", None
    )
    context_names = {
        _context_mask_key(name)
        for name in getattr(transformer, "_context_names", ())
        if isinstance(name, str)
    }
    auto_state = (
        {
            "decisions": {},
            "expected_contexts": context_names,
            "masks": None,
            "current_decision": False,
        }
        if getattr(transformer, "_xdit_auto_static_scm", False)
        else None
    )

    @torch.compiler.disable
    def can_cache(context_manager, *args, **kwargs):
        pipefusion = get_pipeline_parallel_world_size() > 1
        sequence_parallel = get_sequence_parallel_world_size() > 1
        if pipefusion:
            kwargs["parallelized"] = False
        leader_key = None
        shared = False
        result = _context_static_reuse_decision(
            context_manager,
            (
                auto_state["masks"]
                if auto_state is not None and auto_state["masks"] is not None
                else context_static_masks
            ),
        )
        if (
            result is None
            and (
                pipefusion
                and getattr(
                    context_manager, "_xdit_share_pipefusion_decisions", False
                )
            )
        ):
            from xfuser.core.distributed import get_runtime_state

            state = get_runtime_state()
            if state.patch_mode:
                context_name = getattr(
                    context_manager.get_context(), "name", ""
                )
                context_name = context_name.split(
                    ":pipefusion_patch_", 1
                )[0]
                leader_key = (
                    context_name,
                    context_manager.get_current_step(),
                    kwargs.get("prefix", "Fn"),
                )
                decisions = getattr(
                    context_manager,
                    "_xdit_pipefusion_leader_decisions",
                    None,
                )
                if decisions is None:
                    decisions = {}
                    context_manager._xdit_pipefusion_leader_decisions = decisions
                if state.pipeline_patch_idx == 0:
                    if leader_key[1] == 0:
                        decisions.clear()
                elif leader_key in decisions:
                    result = decisions[leader_key]
                    shared = True
        if result is None:
            with record_function("xdit::cache_dit.can_cache"):
                result = original_can_cache(*args, **kwargs)
        log_decision = pipefusion and os.environ.get(
            "XDIT_LOG_PIPEFUSION_CACHE_DECISIONS"
        ) == "1"
        log_all_decisions = (
            os.environ.get("XDIT_LOG_CACHE_DECISIONS") == "1"
            or os.environ.get("XDIT_LOG_PIPEFUSION_CACHE_DECISIONS") == "1"
        )
        if pipefusion and not sequence_parallel:
            if leader_key is not None and not shared:
                context_manager._xdit_pipefusion_leader_decisions[
                    leader_key
                ] = result
            if log_decision:
                from xfuser.core.distributed import get_runtime_state

                state = get_runtime_state()
                logger.info(
                    "PipeFusion Cache-DiT decision: context=%s step=%s "
                    "patch=%s prefix=%s hit=%s shared=%s",
                    getattr(context_manager.get_context(), "name", None),
                    context_manager.get_current_step(),
                    state.pipeline_patch_idx,
                    kwargs.get("prefix", "Fn"),
                    result,
                    shared,
                )
            return result
        if pipefusion:
            from xfuser.core.distributed import get_sp_group

            sync_group = get_sp_group()
        else:
            sync_group = get_world_group()
        decision = torch.tensor(
            [1 if result else 0],
            device=torch.cuda.current_device(),
            dtype=torch.int32,
        )
        with record_function("xdit::cache_dit.sync_decision"):
            sync_group.broadcast(decision, src=0)
            agreed = bool(decision.item())
        if auto_state is not None and auto_state["masks"] is None:
            auto_state["current_decision"] = agreed
            if _derive_auto_static_masks(auto_state, context_manager):
                transformer._xdit_context_static_masks = auto_state["masks"]
                if is_rank0():
                    logger.info(
                        "Derived Cache-DiT auto-static masks: %s",
                        {
                            name: "".join(map(str, mask))
                            for name, mask in auto_state["masks"].items()
                        },
                    )
        if agreed and not getattr(
            context_manager, "_xdit_cache_warmed", False
        ):
            dist.barrier(group=sync_group.device_group)
            context_manager._xdit_cache_warmed = True
        if leader_key is not None and not shared:
            context_manager._xdit_pipefusion_leader_decisions[
                leader_key
            ] = agreed
        if log_all_decisions and is_rank0():
            logger.info(
                "Cache-DiT reuse: context=%s step=%s prefix=%s reuse=%s",
                getattr(context_manager.get_context(), "name", None),
                context_manager.get_current_step(),
                kwargs.get("prefix", "Fn"),
                agreed,
            )
        if log_decision:
            from xfuser.core.distributed import get_runtime_state

            state = get_runtime_state()
            logger.info(
                "PipeFusion Cache-DiT decision: context=%s step=%s patch=%s "
                "prefix=%s hit=%s shared=%s",
                getattr(context_manager.get_context(), "name", None),
                context_manager.get_current_step(),
                state.pipeline_patch_idx,
                kwargs.get("prefix", "Fn"),
                agreed,
                shared,
            )
        return agreed

    manager.can_cache = types.MethodType(can_cache, manager)
    manager._xdit_share_pipefusion_decisions = getattr(
        transformer, "_xdit_share_pipefusion_decisions", False
    )
    manager._xdit_cache_decision_sync_installed = True


def install_pipefusion_patch_contexts(transformer: torch.nn.Module) -> None:
    """Give every PipeFusion patch an independent Cache-DiT history."""
    from xfuser.core.distributed import get_runtime_state

    manager = getattr(transformer, "_context_manager", None)
    base_names = getattr(transformer, "_context_names", None)
    if manager is None or not base_names:
        raise RuntimeError(
            "Cache-DiT did not expose _context_manager/_context_names after "
            "enable_cache; cannot install PipeFusion patch contexts."
        )
    if getattr(manager, "_xdit_pipefusion_patch_contexts_installed", False):
        return

    patch_context_names = {
        name for name in base_names if isinstance(name, str)
    }
    if not patch_context_names:
        raise RuntimeError("Cache-DiT exposed no named PipeFusion contexts.")
    if is_rank0():
        logger.info(
            "Installed %d PipeFusion DBCache patch contexts across %d block "
            "groups.",
            get_runtime_state().num_pipeline_patch * len(patch_context_names),
            len(patch_context_names),
        )

    original_reset_context = manager.reset_context

    @torch.compiler.disable
    def reset_patch_contexts(
        self,
        cached_context,
        *args,
        _names=patch_context_names,
        _reset=original_reset_context,
        **kwargs,
    ):
        context = _reset(cached_context, *args, **kwargs)
        if isinstance(cached_context, str) and cached_context in _names:
            for patch_index in range(get_runtime_state().num_pipeline_patch):
                _reset(
                    f"{cached_context}:pipefusion_patch_{patch_index}",
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
        _names=patch_context_names,
        _set=original_set_context,
        **kwargs,
    ):
        state = get_runtime_state()
        if (
            state.patch_mode
            and isinstance(cached_context, str)
            and cached_context in _names
        ):
            if state.pipeline_patch_idx >= state.num_pipeline_patch:
                raise RuntimeError(
                    "PipeFusion patch index exceeds the active patch layout."
                )
            cached_context = (
                f"{cached_context}:pipefusion_patch_{state.pipeline_patch_idx}"
            )
        return _set(cached_context, *args, **kwargs)

    manager.set_context = types.MethodType(set_patch_context, manager)
    manager._xdit_pipefusion_patch_contexts_installed = True
