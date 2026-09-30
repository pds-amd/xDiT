import torch
import pytest

from xfuser.core.cache_manager.cache_manager import CacheManager


def test_default_cache_entry_keeps_kv_state_in_manager():
    manager = CacheManager()
    layer = torch.nn.Linear(2, 2)
    key_value = torch.ones(1, 2, 2)
    manager.register_cache_entry(layer, "attn")

    assert manager.update_and_get_kv_cache(key_value, layer) is key_value
    assert manager.cache["attn", layer].tensors[0] is key_value
    assert "_xdit_kv_cache" not in layer._buffers


def test_module_buffers_are_promoted_after_registration():
    manager = CacheManager()
    layer = torch.nn.Linear(2, 2)
    cached = torch.ones(1, 2, 2)
    manager.register_cache_entry(layer, "attn")
    manager.cache["attn", layer].tensors[0] = cached

    manager.enable_module_buffers()

    assert manager.cache["attn", layer].use_module_buffer is True
    assert manager.cache["attn", layer].tensors == [None]
    assert layer._xdit_kv_cache is cached


def test_patch_update_accepts_equivalent_flattened_kv_layout():
    manager = CacheManager()
    cache = torch.zeros(2, 8, 3, 4)
    patch = torch.arange(2 * 2 * 12).reshape(2, 2, 12)

    updated = manager._update_kv_in_dim(
        cache,
        patch,
        dim=1,
        start_idx=2,
        end_idx=4,
        normalize_pipefusion_layout=True,
    )

    assert torch.equal(updated[:, 2:4], patch.reshape(2, 2, 3, 4))


def test_patch_update_rejects_incompatible_kv_layout():
    manager = CacheManager()
    cache = torch.zeros(1, 8, 3, 4)
    patch = torch.zeros(1, 2, 11)

    with pytest.raises(RuntimeError):
        manager._update_kv_in_dim(
            cache,
            patch,
            dim=1,
            start_idx=2,
            end_idx=4,
        )


def test_default_cache_path_rejects_flattened_kv_layout():
    manager = CacheManager()
    cache = torch.zeros(2, 8, 3, 4)
    patch = torch.zeros(2, 2, 12)

    with pytest.raises(RuntimeError):
        manager._update_kv_in_dim(
            cache,
            patch,
            dim=1,
            start_idx=2,
            end_idx=4,
        )


def test_patch_update_rejects_permuted_equal_numel_layout():
    manager = CacheManager()
    cache = torch.zeros(1, 8, 3, 4)
    patch = torch.zeros(1, 3, 2, 4)

    with pytest.raises(RuntimeError):
        manager._update_kv_in_dim(
            cache,
            patch,
            dim=1,
            start_idx=2,
            end_idx=4,
        )


def test_clear_releases_module_and_entry_tensors():
    manager = CacheManager()
    layer = torch.nn.Linear(2, 2)
    manager.register_cache_entry(
        layer,
        "attn",
        use_module_buffer=True,
    )
    cached = torch.ones(1, 2, 2)
    manager.cache["attn", layer].tensors[0] = cached
    layer._xdit_kv_cache = cached

    manager.clear()

    assert manager.cache["attn", layer].tensors == [None]
    assert layer._xdit_kv_cache is None
