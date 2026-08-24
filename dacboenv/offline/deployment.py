"""Canonical deployment-head and checkpoint-selection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeploymentHead = Literal["long_q", "branch_q5"]

_DEPLOYMENT_HEADS: dict[str, DeploymentHead] = {
    "branch_q5_only": "branch_q5",
    "offline_fqi": "long_q",
    "offline_cql": "long_q",
    "branch_pretrain_then_fqi": "long_q",
    "branch_pretrain_then_cql": "long_q",
    "joint_branch_cql": "long_q",
    "behavior_cloning": "long_q",
}


def deployment_head_for_mode(mode: str) -> DeploymentHead:
    """Return the only scientifically valid deployment head for a training mode."""
    try:
        return _DEPLOYMENT_HEADS[mode]
    except KeyError as error:
        raise ValueError(f"Unsupported offline algorithm mode {mode!r}.") from error


def deployment_selection_eligible(mode: str, update: int, branch_pretrain_updates: int) -> bool:
    """Return whether the deployed head has received a training update."""
    deployment_head_for_mode(mode)
    if update <= 0:
        return False
    if mode in {"branch_pretrain_then_fqi", "branch_pretrain_then_cql"}:
        return update > branch_pretrain_updates
    return True


@dataclass(slots=True)
class DeploymentSelectionState:
    """Resume-safe state for development-selected checkpointing."""

    deployment_head: DeploymentHead
    best_value: float = float("-inf")
    selected_update: int | None = None
    eligible_checkpoint_seen: bool = False
    patience_counter: int = 0

    @property
    def metric(self) -> str:
        """Return the canonical checkpoint-selection metric."""
        return "dev/deployment_selected_value"

    def to_dict(self) -> dict[str, object]:
        """Return checkpoint-safe state."""
        return {
            "deployment_head": self.deployment_head,
            "checkpoint_selection_head": self.deployment_head,
            "checkpoint_selection_metric": self.metric,
            "checkpoint_selection_value": self.best_value,
            "selected_update": self.selected_update,
            "deployment_selection_eligible": self.eligible_checkpoint_seen,
            "patience_counter": self.patience_counter,
        }

    def consider(self, *, value: float, update: int, eligible: bool) -> bool:
        """Update selection state and report whether a new best was selected."""
        if not eligible:
            return False
        self.eligible_checkpoint_seen = True
        if value > self.best_value:
            self.best_value = value
            self.selected_update = update
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False


__all__ = [
    "DeploymentHead",
    "DeploymentSelectionState",
    "deployment_head_for_mode",
    "deployment_selection_eligible",
]
