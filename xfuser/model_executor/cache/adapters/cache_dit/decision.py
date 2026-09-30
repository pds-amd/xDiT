"""Tensor-only primitives for Cache-DiT adaptive decisions.

Cache-DiT's public context API returns Python booleans. These helpers keep the
residual calculation on the device so a PipeFusion-aware wrapper can defer the
host observation or select a compiled branch.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CacheProbe:
    """Device-resident residual ratio and threshold for one cache context."""

    ratio: torch.Tensor
    threshold: float

    def decision_tensor(self) -> torch.Tensor:
        return self.ratio < self.threshold


def residual_ratio(
    previous: torch.Tensor,
    current: torch.Tensor,
    *,
    important_condition_threshold: float = 0.0,
) -> torch.Tensor:
    """Return Cache-DiT's relative residual metric without host predicates."""
    if previous.shape != current.shape:
        raise ValueError(
            "Cache decision tensors must have matching shapes: "
            f"{previous.shape} != {current.shape}."
        )

    raw_difference = (previous - current).abs()
    if important_condition_threshold <= 0:
        mean_difference = raw_difference.mean()
        mean_reference = previous.abs().mean()
    else:
        token_difference = raw_difference.mean(dim=-1) / previous.abs().mean(
            dim=-1
        )
        selected = (
            token_difference > important_condition_threshold
        ).unsqueeze(-1)
        selected = selected.expand_as(raw_difference)
        selected_count = selected.sum()
        selected_weight = selected.to(raw_difference.dtype)
        selected_difference = (
            raw_difference * selected_weight
        ).sum() / selected_count.clamp_min(1)
        selected_reference = (
            previous.abs() * selected_weight
        ).sum() / selected_count.clamp_min(1)
        mean_difference = torch.where(
            selected_count > 0,
            selected_difference,
            raw_difference.mean(),
        )
        mean_reference = torch.where(
            selected_count > 0,
            selected_reference,
            previous.abs().mean(),
        )
    return mean_difference / mean_reference


def batch_decisions(probes: list[CacheProbe]) -> torch.Tensor:
    """Return one device bool tensor for all ready cache probes."""
    if not probes:
        return torch.empty(0, dtype=torch.bool)
    return torch.stack([probe.decision_tensor() for probe in probes])


def select_functional_branch(
    predicate: torch.Tensor,
    hit_branch,
    miss_branch,
    operands: tuple[torch.Tensor, ...],
):
    """Select a cache branch without mutating state inside ``torch.cond``.

    Both branches must return output tensors followed by replacement cache-state
    tensors. The caller commits the selected state after this operation.
    """
    return torch.cond(predicate, hit_branch, miss_branch, operands)
