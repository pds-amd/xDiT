"""
xDiT step-caching public API.

In-tree methods (teacache, fbcache): xfuser.model_executor.cache.utils
cache-dit adapter (dbcache):         xfuser.model_executor.cache.adapters
"""
from xfuser.model_executor.cache.presets import (
    DBCachePreset,
    CacheDitAdapterConfig,
    DBCacheSettings,
    ModelCacheConfig,
    PipeFusionCacheDecision,
    PipeFusionCachePlan,
    PipeFusionCacheUnit,
    PipeFusionStaticMask,
    PipeFusionTopology,
)
from xfuser.model_executor.cache.adapters import apply_cache

__all__ = [
    "DBCachePreset",
    "CacheDitAdapterConfig",
    "DBCacheSettings",
    "ModelCacheConfig",
    "PipeFusionCacheDecision",
    "PipeFusionCachePlan",
    "PipeFusionCacheUnit",
    "PipeFusionStaticMask",
    "PipeFusionTopology",
    "apply_cache",
]
