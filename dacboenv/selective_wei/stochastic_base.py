"""Reproducible once-per-episode and once-per-block base mixtures."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Literal

import numpy as np

from dacboenv.selective_wei.schemas import ALPHA_GRID, BaseDecision


class SeededBaseMixture:
    """Sample frozen probabilities without redrawing on repeated reads."""

    def __init__(self, mode: Literal["episode_static_mixture", "block_randomized_base"], seed: int = 0) -> None:
        if mode not in {"episode_static_mixture", "block_randomized_base"}:
            raise ValueError("Unknown stochastic base semantics.")
        self.mode = mode
        self.set_seed(seed)

    def set_seed(self, seed: int) -> None:
        """Restart the policy-only stream; do not touch optimizer randomness."""
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        """Clear the episode cache without altering the random stream."""
        self.key: tuple[str, int] | None = None
        self.decision: BaseDecision | None = None

    def select(self, base: BaseDecision, *, episode_id: str, block_index: int) -> BaseDecision:
        """Draw the actual base action and preserve it at this decision boundary."""
        key = (episode_id, 0 if self.mode == "episode_static_mixture" else block_index)
        if self.key == key and self.decision is not None:
            return self.decision
        probabilities = np.asarray(base.action_probabilities, dtype=np.float64)
        if (
            probabilities.shape != (5,)
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("Stochastic base requires a valid frozen five-action distribution.")
        action = int(self.rng.choice(5, p=probabilities))
        self.decision = replace(base, action=action, alpha=ALPHA_GRID[action])
        self.key = key
        return self.decision

    def state_dict(self) -> dict[str, Any]:
        """Serialize RNG and cached draw, not just the original seed."""
        return {
            "mode": self.mode,
            "rng": self.rng.bit_generator.state,
            "key": self.key,
            "decision": None if self.decision is None else asdict(self.decision),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore only the same stochastic semantics."""
        if state["mode"] != self.mode:
            raise ValueError("Cannot restore a different stochastic-base mode.")
        self.rng.bit_generator.state = state["rng"]
        self.key = None if state["key"] is None else tuple(state["key"])
        self.decision = None if state["decision"] is None else BaseDecision(**state["decision"])
