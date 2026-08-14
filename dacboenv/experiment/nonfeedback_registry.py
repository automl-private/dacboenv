"""Freeze deployable nonfeedback selectors selected by the v2 protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.headroom_v2 import enrich, fit_selector, select_nonfeedback, selector_action


def build_registry(branch_results: Path, output: Path) -> dict[str, Any]:
    """Refit the frozen training-selected class and persist executable mappings."""
    rows = enrich(pd.read_parquet(branch_results))
    entries: list[dict[str, Any]] = []
    for family in ("wei", "af_selection"):
        for horizon in (1, 5, 10):
            training = rows[(rows.split == "train") & (rows.action_space == family) & (rows.horizon == horizon)]
            if training.empty:
                raise ValueError(f"No training branches for {family}/H={horizon}.")
            selected_class, development_scores = select_nonfeedback(training)
            model = fit_selector(training, selected_class)
            if selected_class == "context_dimension_function_group_privileged":
                raise RuntimeError("Privileged BBOB function-group selectors cannot be deployable.")
            refit_hash = canonical_sha256(model)
            entries.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "selected_nonfeedback_class": selected_class,
                    "deployable": True,
                    "training_development_scores": development_scores,
                    "refit_model": model,
                    "refit_artifact_hash": refit_hash,
                    "prediction_artifact_hash": None,
                    "protocol_version": "headroom-predictability-v2",
                }
            )
    payload = {
        "schema_version": "nonfeedback-selector-registry-v1",
        "source_branch_results": str(branch_results.resolve()),
        "source_branch_sha256": file_sha256(branch_results),
        "entries": entries,
    }
    payload["registry_hash"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_registry(path: Path) -> dict[str, Any]:
    """Validate registry self-hash and deployable provenance."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    recorded = payload.pop("registry_hash", None)
    actual = canonical_sha256(payload)
    payload["registry_hash"] = recorded
    if recorded != actual:
        raise ValueError(f"Nonfeedback selector registry hash mismatch: {recorded!r} != {actual!r}.")
    entries = payload.get("entries", [])
    identities = {(entry["action_family"], int(entry["horizon"])) for entry in entries}
    if identities != {(family, horizon) for family in ("wei", "af_selection") for horizon in (1, 5, 10)}:
        raise ValueError("Nonfeedback selector registry is incomplete or duplicated.")
    if any(not entry.get("deployable") or "privileged" in entry["selected_nonfeedback_class"] for entry in entries):
        raise ValueError("Nonfeedback selector registry contains a nondeployable selector.")
    return payload


def registry_action(registry: dict[str, Any], family: str, horizon: int, metadata: dict[str, Any]) -> int:
    """Apply exactly the frozen selector for one learned-state comparison."""
    matches = [
        entry
        for entry in registry["entries"]
        if entry["action_family"] == family and int(entry["horizon"]) == int(horizon)
    ]
    if len(matches) != 1:
        raise ValueError(f"Selector registry does not uniquely define {family}/H={horizon}.")
    return selector_action(matches[0]["refit_model"], type("Metadata", (), metadata)())


__all__ = ["build_registry", "load_registry", "registry_action"]
