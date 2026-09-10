"""Task-balanced supervised initial-base models with terminal-loss semantics."""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.selective_wei.artifacts import atomic_json
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_context import InitialBaseDecision, InitialContext

SCALE_FLOOR = 1e-12


@dataclass
class InitialBaseModel:
    """Predict all five paired scaled terminal losses; choose their minimum."""

    estimator: Any
    groups: tuple[str, ...]
    schema_hash: str
    mean: np.ndarray
    scale: np.ndarray
    metadata: dict[str, Any]
    semantic_hash: str = "unsaved"

    def predict(self, contexts: Sequence[InitialContext]) -> np.ndarray:
        """Apply frozen training-only scaling and explicit availability masks."""
        if any(context.schema_hash != self.schema_hash for context in contexts):
            raise ValueError("Initial-base observation schema/diagnostic protocol mismatch.")
        rows = [context.predictor_input(self.groups) for context in contexts]
        values = np.asarray([row[0] for row in rows])
        masks = np.asarray([row[1] for row in rows], dtype=float)
        inputs = np.concatenate((((values - self.mean) / self.scale) * masks, masks), axis=1)
        prediction = np.asarray(self.estimator.predict(inputs), dtype=float)
        if prediction.shape != (len(contexts), 5) or not np.isfinite(prediction).all():
            raise ValueError("Nonfinite or incompatible initial-base predictions.")
        return prediction

    def select(self, context: InitialContext) -> InitialBaseDecision:
        """Select once from full-static terminal-loss predictions, not local Q."""
        if not context.has_controlled_budget or self.semantic_hash == "unsaved":
            raise ValueError("Initial base requires a saved artifact and remaining controlled budget.")
        action = int(np.argmin(self.predict([context])[0]))
        return InitialBaseDecision(context.digest, action, self.semantic_hash, int(self.metadata["seed"]))


def fit_initial_base(
    contexts: Sequence[InitialContext],
    losses: np.ndarray,
    task_ids: Sequence[str],
    *,
    groups: tuple[str, ...],
    kind: str = "ridge",
    seed: int = 0,
    source_split: str = "train",
) -> InitialBaseModel:
    """Fit a train-only task-balanced model on paired scaled terminal losses."""
    if source_split != "train" or not contexts or len(contexts) != len(task_ids):
        raise ValueError("Initial-base fitting requires explicit training-only contexts.")
    if len({context.schema_hash for context in contexts}) != 1:
        raise ValueError("Mixed initial feature protocols cannot be fitted together.")
    if not groups or not set(groups) <= {"M", "D", "G", "U"}:
        raise ValueError("Unknown initial feature ablation.")
    targets = np.asarray(losses, dtype=float)
    if targets.shape != (len(contexts), 5) or not np.isfinite(targets).all():
        raise ValueError("Complete five-action finite terminal targets are required.")
    tasks = np.asarray(task_ids).astype(str)
    weights = np.asarray([1 / np.sum(tasks == task) for task in tasks])
    weights /= weights.sum()
    rows = [context.predictor_input(groups) for context in contexts]
    values = np.asarray([row[0] for row in rows])
    masks = np.asarray([row[1] for row in rows], dtype=float)
    support = (weights[:, None] * masks).sum(axis=0)
    mean = (values * weights[:, None] * masks).sum(axis=0) / np.maximum(support, 1e-12)
    variance = (((values - mean) ** 2) * weights[:, None] * masks).sum(axis=0) / np.maximum(support, 1e-12)
    scale = np.sqrt(variance)
    scale[scale < SCALE_FLOOR] = 1.0
    inputs = np.concatenate((((values - mean) / scale) * masks, masks), axis=1)
    if kind == "ridge":
        estimator = Ridge(alpha=1.0)
    elif kind == "extra_trees":
        estimator = ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=1)
    else:
        raise ValueError("Initial base supports ridge and extra_trees, not offline long-Q algorithms.")
    estimator.fit(inputs, targets, sample_weight=weights * len(set(tasks)))
    return InitialBaseModel(
        estimator,
        groups,
        contexts[0].schema_hash,
        mean,
        scale,
        {
            "kind": kind,
            "seed": seed,
            "source_split": source_split,
            "task_ids": sorted(set(tasks)),
            "task_hash": canonical_hash(sorted(set(tasks))),
            "fit_context_hash": canonical_hash([context.digest for context in contexts]),
            "target_semantics": "full_static_terminal_loss",
            "target_scaling": "paired_alpha_half_over_D0_scale",
            "normalization": "train-task-balanced-masked-standardization",
        },
    )


def save_initial_base(model: InitialBaseModel, root: Path) -> dict[str, Any]:
    """Save a relocatable, hashed predictor and normalizer bundle atomically."""
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "predictor.pkl"
    if destination.exists() or (root / "base_bundle.json").exists():
        raise FileExistsError("Refusing to overwrite an initial-base artifact.")
    temporary = root / ".predictor.tmp"
    with temporary.open("wb") as stream:
        pickle.dump(model.estimator, stream, protocol=5)
    temporary.replace(destination)
    payload = {
        "schema_version": "initial-base-bundle-v1",
        "metadata": model.metadata,
        "groups": model.groups,
        "feature_schema_hash": model.schema_hash,
        "mean": model.mean.tolist(),
        "scale": model.scale.tolist(),
        "predictor_file": "predictor.pkl",
        "predictor_sha256": file_sha256(destination),
        "action_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "interaction_frequency": 5,
    }
    model.semantic_hash = canonical_hash(payload)
    payload["semantic_hash"] = model.semantic_hash
    atomic_json(root / "base_bundle.json", payload)
    return payload


def load_initial_base(path: Path) -> InitialBaseModel:
    """Verify trusted local model bytes before deserializing a relocated bundle."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.pop("semantic_hash")
    if canonical_hash(payload) != expected or payload["schema_version"] != "initial-base-bundle-v1":
        raise ValueError("Initial-base semantic artifact mismatch.")
    if (
        payload["metadata"]["target_semantics"] != "full_static_terminal_loss"
        or payload["metadata"]["source_split"] != "train"
        or payload["action_grid"] != [0.0, 0.25, 0.5, 0.75, 1.0]
        or payload["interaction_frequency"] != len(payload["action_grid"])
    ):
        raise ValueError("Initial-base target, action, split or f5 contract mismatch.")
    mean = np.asarray(payload["mean"], dtype=float)
    scale = np.asarray(payload["scale"], dtype=float)
    if mean.ndim != 1 or scale.shape != mean.shape or not np.isfinite(mean).all():
        raise ValueError("Initial-base normalizer shape or mean is invalid.")
    if not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Initial-base normalizer scale is invalid.")
    predictor_path = path.parent / payload["predictor_file"]
    if predictor_path.name != "predictor.pkl" or predictor_path.parent != path.parent:
        raise ValueError("Initial-base model must be bundle-local.")
    if file_sha256(predictor_path) != payload["predictor_sha256"]:
        raise ValueError("Initial-base model content hash mismatch.")
    with predictor_path.open("rb") as stream:
        estimator = pickle.load(stream)  # noqa: S301 - verified locally produced trusted artifact
    return InitialBaseModel(
        estimator,
        tuple(payload["groups"]),
        payload["feature_schema_hash"],
        np.asarray(payload["mean"]),
        np.asarray(payload["scale"]),
        payload["metadata"],
        expected,
    )
