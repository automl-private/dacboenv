"""Verified row adapters for local residual calibration against frozen a0."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dacboenv.selective_wei.initial_base_model import InitialBaseModel, load_initial_base
from dacboenv.selective_wei.initial_context import InitialAnchorCache
from dacboenv.selective_wei.schemas import BaseDecision, Horizon


@dataclass
class FrozenInitialRowBase:
    """Read the original action; never predict a base from a mid-run feature row."""

    path: Path
    model: InitialBaseModel
    horizon: Horizon

    @classmethod
    def load(cls, path: Path, horizon: Horizon) -> FrozenInitialRowBase:
        """Load a verified terminal-target base for residual label interpretation."""
        return cls(path, load_initial_base(path), horizon)

    def row_decision(self, arrays: dict[str, Any], index: int) -> BaseDecision:
        """Verify anchor, declared action and original model before returning a0."""
        anchor = json.loads(str(arrays["initial_anchor_json"][index]))
        cache = InitialAnchorCache.from_dict(anchor["state"])
        if cache.context is None or cache.decision is None:
            raise ValueError("Local residual labels require a complete initial anchor.")
        if cache.decision.base_semantic_hash != self.model.semantic_hash:
            raise ValueError("Branch anchor refers to a different initial base.")
        if int(arrays["initial_base_action"][index]) != cache.decision.action:
            raise ValueError("Declared branch a0 conflicts with its verified anchor.")
        if self.model.select(cache.context).action != cache.decision.action:
            raise ValueError("Initial base no longer reproduces the anchored action.")
        metadata = self.model.metadata
        return BaseDecision(
            cache.decision.action,
            cache.decision.alpha,
            "initial_terminal_base",
            "initial_context",
            "exact",
            self.horizon,
            metadata["partition_manifest_hash"],
            metadata["train_dataset_hash"],
        )
