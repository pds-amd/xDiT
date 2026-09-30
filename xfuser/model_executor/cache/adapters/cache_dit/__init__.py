"""xDiT's optional Cache-DiT integration.

The package root preserves the previous adapter API while implementation is
organized by configuration, distributed context handling, and application.
"""

from .application import (
    apply_cache_dit_cache,
    apply_cache_dit_cache_multi,
    build_adapter as _build_adapter,
    unwrap_fsdp as _unwrap_fsdp,
)
from .config import (
    build_calibrator_config as _build_calibrator_config,
    build_config as _build_config,
    build_scm_mask as _build_scm_mask,
    import_cache_dit as _import_cache_dit,
    is_rank0 as _is_rank0,
    resolve_enable_separate_cfg as _resolve_enable_separate_cfg,
)
from .context import (
    install_cache_decision_sync as _install_cache_decision_sync,
    install_pipefusion_patch_contexts as _install_pipefusion_patch_contexts,
    is_parallelized_flag as _is_parallelized_flag,
)

__all__ = [
    "apply_cache_dit_cache",
    "apply_cache_dit_cache_multi",
]
