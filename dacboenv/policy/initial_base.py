"""CARP-S policy adapter for a once-selected initial-conditioned WEI base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dacboenv.policy.abstract_policy import AbstractPolicy
from dacboenv.selective_wei.initial_base_model import load_initial_base

ACTION_COUNT = 5
INTERACTION_FREQUENCY = 5


class InitialConditionedWEIPolicy(AbstractPolicy):
    """Freeze a0 from z0; subsequent calls never re-infer from the current GP."""

    def __init__(self, env: Any, base_bundle: str) -> None:
        super().__init__(env, base_bundle=base_bundle)
        self.base = load_initial_base(Path(base_bundle))
        if env.interaction_frequency != INTERACTION_FREQUENCY or getattr(env.action_space, "n", None) != ACTION_COUNT:
            raise ValueError("Initial-conditioned WEI requires Discrete(5) and f=5.")
        self.settings = self.base.metadata["feature_settings"]
        env.configure_initial_context(self.settings)

    def __call__(self, obs: Any) -> int:
        """Use the initial context cache, even after later surrogate updates."""
        del obs
        context = self._env.get_initial_context(self.settings)
        state = self._env.initial_anchor_state()
        if state["decision"] is None:
            decision = self._env.freeze_initial_base(self.base.select(context))
            return int(decision.action)
        if state["decision"]["base_semantic_hash"] != self.base.semantic_hash:
            raise ValueError("Episode anchor refers to another initial-base artifact.")
        return int(state["decision"]["action"])
