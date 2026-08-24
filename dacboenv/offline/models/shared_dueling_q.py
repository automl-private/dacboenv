"""Permutation-equivariant shared action-value networks."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import nn

ModelVariant = Literal["flat", "shared", "shared_dueling"]


@dataclass(frozen=True, slots=True)
class OfflineQModelConfig:
    """Shape and architecture settings persisted with every checkpoint."""

    variant: ModelVariant = "shared_dueling"
    state_dim: int = 13
    action_feature_dim: int = 4
    action_count: int = 5
    append_alpha_identity: bool = True
    state_hidden: int = 64
    action_hidden: int = 32
    fusion_hidden: int = 64
    separate_branch_head: bool = True


def _mlp(sizes: list[int], *, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(left, right))
        if index < len(sizes) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class OfflineQNetwork(nn.Module):
    """Shared per-action scorer with separate branch-Q5 and long-Q heads.

    ``long_q`` is the finite-episode Bellman action value used by fitted-Q/CQL.
    ``branch_q5`` is the fixed-action five-BO-evaluation counterfactual value.
    Their trunks are shared, but their heads and targets are never conflated.
    """

    def __init__(self, config: OfflineQModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.variant == "flat":
            flat_dim = config.state_dim + config.action_count * config.action_feature_dim
            self.flat_long = _mlp([flat_dim, 128, 128, config.action_count])
            self.flat_branch = _mlp([flat_dim, 128, 128, config.action_count])
            self.flat_branch_q10 = _mlp([flat_dim, 128, 128, config.action_count])
            return
        action_input = config.action_feature_dim + int(config.append_alpha_identity)
        self.state_encoder = _mlp([config.state_dim, config.state_hidden, config.state_hidden], final_activation=True)
        self.action_encoder = _mlp([action_input, config.action_hidden, config.action_hidden], final_activation=True)
        fusion_input = config.state_hidden + config.action_hidden
        self.long_advantage = _mlp([fusion_input, config.fusion_hidden, config.fusion_hidden, 1])
        self.branch_advantage = _mlp([fusion_input, config.fusion_hidden, config.fusion_hidden, 1])
        self.branch_q10_advantage = _mlp([fusion_input, config.fusion_hidden, config.fusion_hidden, 1])
        if config.variant == "shared_dueling":
            self.long_value = _mlp([config.state_hidden, config.fusion_hidden, 1])
            self.branch_value = _mlp([config.state_hidden, config.fusion_hidden, 1])
            self.branch_q10_value = _mlp([config.state_hidden, config.fusion_hidden, 1])

    def _shared(self, state: torch.Tensor, action_features: torch.Tensor, head: str) -> torch.Tensor:
        _batch, actions, _features = action_features.shape
        if actions != self.config.action_count:
            raise ValueError(f"Expected {self.config.action_count} action rows, received {actions}.")
        if self.config.append_alpha_identity:
            # Structured WEI action rows already expose alpha in column zero.
            # Appending that value makes identity explicit without tying it to
            # row position, so it follows any scientifically valid row permutation.
            alpha = action_features[..., :1]
            action_features = torch.cat((action_features, alpha), dim=-1)
        state_embedding = self.state_encoder(state)
        action_embedding = self.action_encoder(action_features)
        repeated_state = state_embedding.unsqueeze(1).expand(-1, actions, -1)
        fused = torch.cat((repeated_state, action_embedding), dim=-1)
        advantage_head = {
            "long_q": self.long_advantage,
            "branch_q5": self.branch_advantage,
            "branch_q10": self.branch_q10_advantage,
        }[head]
        advantage = advantage_head(fused).squeeze(-1)
        if self.config.variant == "shared":
            return cast("torch.Tensor", advantage)
        value_head = {
            "long_q": self.long_value,
            "branch_q5": self.branch_value,
            "branch_q10": self.branch_q10_value,
        }[head]
        value = value_head(state_embedding)
        # Sorting before reduction makes the dueling centering term bitwise
        # invariant to an action-row permutation, not merely close in real arithmetic.
        centered_mean = advantage.sort(dim=1).values.mean(dim=1, keepdim=True)
        return cast("torch.Tensor", value + advantage - centered_mean)

    def forward(
        self,
        state: torch.Tensor,
        action_features: torch.Tensor,
        *,
        head: Literal["long_q", "branch_q5", "branch_q10"] = "long_q",
    ) -> torch.Tensor:
        """Return one Q value per action for the explicitly named head."""
        if head not in {"long_q", "branch_q5", "branch_q10"}:
            raise ValueError(f"Unknown Q head {head!r}.")
        if self.config.variant == "flat":
            flat = torch.cat((state, action_features.flatten(start_dim=1)), dim=1)
            return cast(
                "torch.Tensor",
                {
                    "long_q": self.flat_long,
                    "branch_q5": self.flat_branch,
                    "branch_q10": self.flat_branch_q10,
                }[head](flat),
            )
        return self._shared(state, action_features, head)

    def greedy_action(self, state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Select the deterministic long-Q action."""
        return cast("torch.Tensor", self(state, action_features, head="long_q").argmax(dim=1))

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable scalar parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


def build_offline_q_model(config: dict[str, object]) -> OfflineQNetwork:
    """Build a model from a plain, checkpoint-safe mapping."""
    return OfflineQNetwork(OfflineQModelConfig(**cast("dict[str, Any]", config)))
