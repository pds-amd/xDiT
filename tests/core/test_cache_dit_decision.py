import torch

from xfuser.model_executor.cache.adapters.cache_dit.decision import (
    CacheProbe,
    batch_decisions,
    residual_ratio,
    select_functional_branch,
)


def test_residual_ratio_matches_plain_cache_dit_metric():
    previous = torch.tensor([[[2.0, 4.0], [6.0, 8.0]]])
    current = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    ratio = residual_ratio(previous, current)

    expected = (previous - current).abs().mean() / previous.abs().mean()
    assert torch.equal(ratio, expected)
    assert ratio.device == previous.device


def test_important_token_metric_uses_global_fallback_without_host_predicate():
    previous = torch.ones(1, 2, 2)
    current = previous * 0.9

    ratio = residual_ratio(
        previous,
        current,
        important_condition_threshold=1.0,
    )

    assert torch.allclose(ratio, torch.tensor(0.1))


def test_batched_cache_decisions_remain_on_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decisions = batch_decisions(
        [
            CacheProbe(torch.tensor(0.05, device=device), threshold=0.1),
            CacheProbe(torch.tensor(0.15, device=device), threshold=0.1),
        ]
    )

    assert decisions.dtype is torch.bool
    assert decisions.device.type == device.type
    assert torch.equal(
        decisions.cpu(),
        torch.tensor([True, False]),
    )


def test_device_branch_returns_state_delta_without_branch_mutation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    value = torch.tensor([2.0], device=device)
    cached_state = torch.tensor([3.0], device=device)

    def hit(value, cached_state):
        return value + cached_state, cached_state.clone()

    def miss(value, _cached_state):
        return value * 2, value.clone()

    output, replacement_state = select_functional_branch(
        torch.tensor(False, device=device),
        hit,
        miss,
        (value, cached_state),
    )

    assert torch.equal(output.cpu(), torch.tensor([4.0]))
    assert torch.equal(replacement_state.cpu(), torch.tensor([2.0]))
