import torch

from xfuser.core.cache_manager.cache_manager import CacheManager


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
    )

    assert torch.equal(updated[:, 2:4], patch.reshape(2, 2, 3, 4))


def test_patch_update_rejects_incompatible_kv_layout():
    manager = CacheManager()
    cache = torch.zeros(1, 8, 3, 4)
    patch = torch.zeros(1, 2, 11)

    try:
        manager._update_kv_in_dim(
            cache,
            patch,
            dim=1,
            start_idx=2,
            end_idx=4,
        )
    except ValueError as error:
        assert "cannot update cache slice shape" in str(error)
    else:
        raise AssertionError("incompatible KV layout was accepted")


def test_patch_update_rejects_permuted_equal_numel_layout():
    manager = CacheManager()
    cache = torch.zeros(1, 8, 3, 4)
    patch = torch.zeros(1, 3, 2, 4)

    try:
        manager._update_kv_in_dim(
            cache,
            patch,
            dim=1,
            start_idx=2,
            end_idx=4,
        )
    except ValueError as error:
        assert "cannot update cache slice shape" in str(error)
    else:
        raise AssertionError("permuted KV layout was accepted")


def test_clear_releases_module_and_entry_tensors():
    manager = CacheManager()
    layer = torch.nn.Linear(2, 2)
    manager.register_cache_entry(layer, "attn")
    cached = torch.ones(1, 2, 2)
    manager.cache["attn", layer].tensors[0] = cached
    layer._xdit_kv_cache = cached

    manager.clear()

    assert manager.cache["attn", layer].tensors == [None]
    assert layer._xdit_kv_cache is None
