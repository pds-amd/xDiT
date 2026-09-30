import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

from xfuser.core.distributed.runtime_state import runtime_state_is_initialized
from xfuser.logger import init_logger

logger = init_logger(__name__)


class CacheEntry:
    def __init__(
        self,
        cache_type: "str",
        num_cache_tensors: int = 1,
        tensors: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        use_module_buffer: bool = False,
    ):
        self.cache_type: str = cache_type
        self.use_module_buffer = use_module_buffer
        if tensors is None:
            self.tensors: List[torch.Tensor] = [
                None,
            ] * num_cache_tensors
        elif isinstance(tensors, torch.Tensor):
            assert (
                num_cache_tensors == 1
            ), "num_cache_tensors must be 1 if you pass a single tensor to tensors argument"
            self.tensors = [
                tensors,
            ]
        elif isinstance(tensors, List):
            assert num_cache_tensors == len(
                tensors
            ), "num_cache_tensors must be equal to num of tensors"
            self.tensors = [
                tensors,
            ]


class CacheManager:
    supported_layer = ["attn"]
    supported_cache_type = ["naive_cache", "sequence_parallel_attn_cache"]

    def __init__(
        self,
    ):
        self.cache: Dict[Tuple[str, Any], CacheEntry] = {}

    def register_cache_entry(
        self,
        layer,
        layer_type: str,
        cache_type: str = "naive_cache",
        *,
        use_module_buffer: bool = False,
    ):
        if layer_type not in self.supported_layer:
            raise ValueError(
                f"Layer type: {layer_type} is not supported. Supported layer type: {self.supported_layer}"
            )
        if cache_type not in self.supported_cache_type:
            raise ValueError(
                f"Cache type: {cache_type} is not supported. Supported cache type: {self.supported_cache_type}"
            )
        if self.cache.get((layer_type, layer), None) is not None:
            logger.warning(
                f"Cache for [layer_type, layer]: [{layer_type}, {layer.__class__}] is already initialized, resetting the cache..."
            )
        self.cache[layer_type, layer] = CacheEntry(
            cache_type,
            use_module_buffer=use_module_buffer,
        )
        if use_module_buffer and isinstance(layer, torch.nn.Module):
            self._enable_module_buffer(layer, self.cache[layer_type, layer])

    def has_cache_entry(self, layer, layer_type: str = "attn") -> bool:
        return (layer_type, layer) in self.cache

    @staticmethod
    def _enable_module_buffer(
        layer: torch.nn.Module,
        entry: CacheEntry,
    ) -> None:
        entry.use_module_buffer = True
        cached = entry.tensors[0]
        if "_xdit_kv_cache" in layer._buffers:
            layer._buffers["_xdit_kv_cache"] = cached
        else:
            layer.register_buffer(
                "_xdit_kv_cache",
                cached,
                persistent=False,
            )
        entry.tensors[0] = None

    def enable_module_buffers(self) -> None:
        """Promote existing cache entries after PP model materialization.

        Registering buffers while wrappers are constructed can trigger an
        expensive CPU-side module walk in large Diffusers pipelines. PipeFusion
        needs module-owned K/V state only for compiled stages, so promotion is
        deferred until loading has completed.
        """
        for (_, layer), entry in self.cache.items():
            if isinstance(layer, torch.nn.Module):
                self._enable_module_buffer(layer, entry)

    def clear(self) -> None:
        """Release cached activations while preserving layer registrations."""
        for (_, layer), entry in self.cache.items():
            entry.tensors = [None] * len(entry.tensors)
            if isinstance(layer, torch.nn.Module) and hasattr(
                layer, "_xdit_kv_cache"
            ):
                layer._xdit_kv_cache = None

    def update_and_get_kv_cache(
        self,
        new_kv: Union[torch.Tensor, List[torch.Tensor]],
        layer: Any,
        slice_dim: int = 1,
        layer_type: str = "attn",
        custom_get_kv: Optional[Callable[[Any, Any, str], torch.Tensor]] = None,
        **kwargs,
    ):
        return_list = False
        if isinstance(new_kv, List):
            return_list = True
            new_kv = torch.cat(new_kv, dim=-1)

        if custom_get_kv is not None:
            return custom_get_kv(self, new_kv, layer, slice_dim, layer_type, **kwargs)
        else:
            entry = self.cache[layer_type, layer]
            module_cache = entry.use_module_buffer
            if module_cache:
                cache_type = entry.cache_type
                kv_cache = layer._xdit_kv_cache
            else:
                cache_type = entry.cache_type
                kv_cache = entry.tensors[0]
            if cache_type == "naive_cache":
                kv_cache = self._naive_cache_update(
                    new_kv,
                    kv_cache=kv_cache,
                    slice_dim=slice_dim,
                    normalize_pipefusion_layout=module_cache,
                    **kwargs,
                )
            elif cache_type == "sequence_parallel_attn_cache":
                kv_cache = self._sequence_parallel_cache_update(
                    new_kv,
                    kv_cache=kv_cache,
                    slice_dim=slice_dim,
                    normalize_pipefusion_layout=module_cache,
                    **kwargs,
                )
            if module_cache:
                layer._xdit_kv_cache = kv_cache
            else:
                entry.tensors[0] = kv_cache
            if return_list:
                return torch.chunk(kv_cache, 2, dim=-1)
            else:
                return kv_cache

    def _naive_cache_update(
        self,
        new_kv: Union[torch.Tensor, List[torch.Tensor]],
        kv_cache: Optional[torch.Tensor],
        slice_dim: int = 1,
        normalize_pipefusion_layout: bool = False,
    ):
        from xfuser.core.distributed.runtime_state import get_runtime_state

        if (
            not runtime_state_is_initialized()
            or get_runtime_state().num_pipeline_patch == 1
            or not get_runtime_state().patch_mode
        ):
            kv_cache = new_kv
        else:
            start_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx
            ]
            end_token_idx = get_runtime_state().pp_patches_token_start_idx_local[
                get_runtime_state().pipeline_patch_idx + 1
            ]
            kv_cache = self._update_kv_in_dim(
                kv_cache=kv_cache,
                new_kv=new_kv,
                dim=slice_dim,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
                normalize_pipefusion_layout=normalize_pipefusion_layout,
            )
        return kv_cache

    # work inside ring attn
    def _sequence_parallel_cache_update(
        self,
        new_kv: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        slice_dim: int = 1,
        normalize_pipefusion_layout: bool = False,
    ):
        from xfuser.core.distributed import (
            get_ulysses_parallel_world_size,
            get_runtime_state,
        )

        ulysses_world_size = get_ulysses_parallel_world_size()
        if (
            not runtime_state_is_initialized()
            or get_runtime_state().num_pipeline_patch == 1
        ):
            return new_kv
        elif not get_runtime_state().patch_mode:
            pp_patches_token_num = get_runtime_state().pp_patches_token_num
            kv_list = [
                kv.split(pp_patches_token_num, dim=slice_dim)
                for kv in torch.chunk(new_kv, ulysses_world_size, dim=slice_dim)
            ]
            kv_cache = torch.cat(
                [
                    kv_list[rank][pp_patch_idx]
                    for rank in range(ulysses_world_size)
                    for pp_patch_idx in range(len(pp_patches_token_num))
                ],
                dim=slice_dim,
            )
        else:
            pp_patches_token_start_idx_local = (
                get_runtime_state().pp_patches_token_start_idx_local
            )
            pp_patch_idx = get_runtime_state().pipeline_patch_idx
            start_token_idx = (
                ulysses_world_size * pp_patches_token_start_idx_local[pp_patch_idx]
            )
            end_token_idx = (
                ulysses_world_size * pp_patches_token_start_idx_local[pp_patch_idx + 1]
            )
            # pp_patches_token_num = get_runtime_state().pp_patches_token_num
            # start_token_idx = ulysses_world_size * sum(pp_patches_token_num[:get_runtime_state().pipeline_patch_idx])
            # end_token_idx = ulysses_world_size * sum(pp_patches_token_num[:get_runtime_state().pipeline_patch_idx + 1])
            kv_cache = self._update_kv_in_dim(
                kv_cache=kv_cache,
                new_kv=new_kv,
                dim=slice_dim,
                start_idx=start_token_idx,
                end_idx=end_token_idx,
                normalize_pipefusion_layout=normalize_pipefusion_layout,
            )
        return kv_cache

    def _update_kv_in_dim(
        self,
        kv_cache: torch.Tensor,
        new_kv: torch.Tensor,
        dim: int,
        start_idx: int,
        end_idx: int,
        normalize_pipefusion_layout: bool = False,
    ):
        if dim < 0:
            dim += kv_cache.dim()
        if not normalize_pipefusion_layout:
            # Preserve the established USP/Ring update behavior. PipeFusion's
            # compiled stage cache opts into the stricter normalized path below.
            if dim == 0:
                kv_cache[start_idx:end_idx, ...] = new_kv
            elif dim == 1:
                kv_cache[:, start_idx:end_idx, ...] = new_kv
            elif dim == 2:
                kv_cache[:, :, start_idx:end_idx, ...] = new_kv
            elif dim == 3:
                kv_cache[:, :, :, start_idx:end_idx, ...] = new_kv
            return kv_cache

        if dim >= kv_cache.dim():
            raise ValueError(
                f"'dim' argument {dim} must be smaller than KV cache dimensions: {kv_cache.dim()}"
            )

        target_shape = list(kv_cache.shape)
        target_shape[dim] = end_idx - start_idx
        if list(new_kv.shape) != target_shape:
            same_prefix = (
                list(new_kv.shape[: dim + 1])
                == target_shape[: dim + 1]
            )
            if not same_prefix or new_kv.numel() != math.prod(target_shape):
                raise ValueError(
                    f"KV patch shape {tuple(new_kv.shape)} cannot update "
                    f"cache slice shape {tuple(target_shape)}"
                )
            # Attention backends may expose equivalent flattened [B,S,H*D]
            # and unflattened [B,S,H,D] layouts across sync/patch graphs.
            new_kv = new_kv.reshape(target_shape)

        index = [slice(None)] * kv_cache.dim()
        index[dim] = slice(start_idx, end_idx)
        kv_cache[tuple(index)] = new_kv
        return kv_cache


_CACHE_MGR = CacheManager()


def get_cache_manager():
    global _CACHE_MGR
    assert _CACHE_MGR is not None, "Cache manager has not been initialized."
    return _CACHE_MGR
