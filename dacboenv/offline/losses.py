"""Losses for counterfactual supervision and conservative fitted Q."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional


@dataclass(frozen=True, slots=True)
class BranchLossResult:
    """Separated fixed-horizon branch loss diagnostics."""

    total: torch.Tensor
    regression: torch.Tensor
    ranking: torch.Tensor
    contributing_pairs: int


@dataclass(frozen=True, slots=True)
class OfflineTDLossResult:
    """Separated Bellman and CQL loss diagnostics."""

    total: torch.Tensor
    td: torch.Tensor
    cql: torch.Tensor
    target: torch.Tensor
    data_q: torch.Tensor


def centered_huber_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    tie_tolerance: float = 1e-3,
    regression_weight: float = 1.0,
    ranking_weight: float = 1.0,
    gap_weighted: bool = True,
    duplicate_groups: torch.Tensor | None = None,
) -> BranchLossResult:
    """Regress centered Q5 values and rank only non-tied valid pairs."""
    valid = valid_mask.to(dtype=prediction.dtype)
    denominator = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
    target_centered = target - (target * valid).sum(dim=1, keepdim=True) / denominator
    prediction_centered = prediction - (prediction * valid).sum(dim=1, keepdim=True) / denominator
    regression_raw = functional.smooth_l1_loss(prediction_centered, target_centered, reduction="none")
    gaps = target.max(dim=1).values - target.topk(k=2, dim=1).values[:, 1]
    state_weight = torch.clamp(gaps / 1e-2, max=1.0) if gap_weighted else torch.ones_like(gaps)
    regression = (regression_raw * valid * state_weight.unsqueeze(1)).sum() / valid.sum().clamp_min(1.0)
    pair_losses: list[torch.Tensor] = []
    for left in range(target.shape[1]):
        for right in range(left + 1, target.shape[1]):
            difference = target[:, left] - target[:, right]
            pair_valid = valid_mask[:, left] & valid_mask[:, right] & (difference.abs() > tie_tolerance)
            if duplicate_groups is not None:
                pair_valid &= duplicate_groups[:, left] != duplicate_groups[:, right]
            if pair_valid.any():
                sign = difference[pair_valid].sign()
                margin = prediction[pair_valid, left] - prediction[pair_valid, right]
                weight = torch.clamp(difference[pair_valid].abs() / 1e-2, max=1.0) if gap_weighted else 1.0
                pair_losses.append((functional.softplus(-sign * margin) * weight).mean())
    ranking = torch.stack(pair_losses).mean() if pair_losses else prediction.sum() * 0.0
    return BranchLossResult(
        total=regression_weight * regression + ranking_weight * ranking,
        regression=regression,
        ranking=ranking,
        contributing_pairs=len(pair_losses),
    )


def double_dqn_targets(
    reward: torch.Tensor,
    done: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    *,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Return finite-episode Double-DQN targets for f5 block rewards."""
    next_action = online_next_q.argmax(dim=1, keepdim=True)
    bootstrap = target_next_q.gather(1, next_action).squeeze(1)
    return reward + (1.0 - done) * gamma * bootstrap


def offline_td_cql_loss(
    q_values: torch.Tensor,
    action: torch.Tensor,
    target: torch.Tensor,
    *,
    cql_coefficient: float,
) -> OfflineTDLossResult:
    """Compute Huber fitted-Q loss plus discrete conservative Q regularization."""
    data_q = q_values.gather(1, action.long().unsqueeze(1)).squeeze(1)
    td = functional.smooth_l1_loss(data_q, target)
    cql = torch.mean(torch.logsumexp(q_values, dim=1) - data_q)
    return OfflineTDLossResult(
        total=td + cql_coefficient * cql,
        td=td,
        cql=cql,
        target=target,
        data_q=data_q,
    )


def behavior_cloning_loss(logits: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """Return the discrete behavior-action cross-entropy diagnostic loss."""
    return functional.cross_entropy(logits, action.long())
