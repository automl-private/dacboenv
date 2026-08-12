"""Leakage-safe same-state headroom decomposition and offline predictors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # noqa: TC002
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

PREDICTOR_OBSERVATION_KEYS = ("global_state", "action_features")
FORBIDDEN_PREDICTOR_TOKENS = ("reference", "optimum", "future", "branch_outcome")
DEFAULT_TIE_TOLERANCES = (0.0, 1e-4, 1e-3)
MATRIX_DIMENSIONS = 2
EARLY_MIDDLE_BOUNDARY = 0.375
MIDDLE_LATE_BOUNDARY = 0.625


def assert_predictor_columns_safe(columns: Iterable[str]) -> None:
    """Reject privileged or future-dependent predictor columns."""
    unsafe = sorted(
        column for column in columns if any(token in column.lower() for token in FORBIDDEN_PREDICTOR_TOKENS)
    )
    if unsafe:
        raise ValueError(f"Predictor inputs contain privileged/future information: {unsafe!r}.")


def parse_policy_observation(observation_json: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode exactly the two arrays saved from the deployable observation."""
    payload = json.loads(observation_json)
    if tuple(sorted(payload)) != tuple(sorted(PREDICTOR_OBSERVATION_KEYS)):
        raise ValueError(f"Expected only {PREDICTOR_OBSERVATION_KEYS!r}, got {tuple(sorted(payload))!r}.")
    arrays = []
    for key in PREDICTOR_OBSERVATION_KEYS:
        item = payload[key]
        arrays.append(np.asarray(item["values"], dtype=np.float32).reshape(item["shape"]))
    global_state, action_features = arrays
    if global_state.ndim != 1 or action_features.ndim != MATRIX_DIMENSIONS:
        raise ValueError("Expected one-dimensional global_state and two-dimensional action_features.")
    return global_state, action_features


def assert_disjoint_context_splits(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Ensure task/seed/history trajectories cannot cross scientific splits."""
    keys = ["task_id", "inner_seed", "history_generator", "history_seed"]
    missing = [key for key in keys if key not in train or key not in validation]
    if missing:
        raise ValueError(f"Split tables lack trajectory keys: {missing!r}.")
    train_keys = set(map(tuple, train[keys].itertuples(index=False, name=None)))
    validation_keys = set(map(tuple, validation[keys].itertuples(index=False, name=None)))
    overlap = train_keys & validation_keys
    if overlap:
        raise ValueError(f"Training and validation share complete trajectories: {sorted(overlap)!r}.")


def add_action_advantages(rows: pd.DataFrame) -> pd.DataFrame:
    """Add within-snapshot action advantages and tie-aware gap labels."""
    required = {"snapshot_id", "action", "horizon", "q_value"}
    if missing := sorted(required - set(rows)):
        raise ValueError(f"Branch table lacks columns {missing!r}.")
    output = rows.copy()
    grouping = ["snapshot_id", "horizon"]
    output["action_advantage"] = output["q_value"] - output.groupby(grouping)["q_value"].transform("mean")
    maxima = output.groupby(grouping)["q_value"].transform("max")
    output["gap_to_best"] = maxima - output["q_value"]
    gaps = output.groupby(grouping)["q_value"].transform(
        lambda values: np.sort(np.asarray(values))[-1] - np.sort(np.asarray(values))[-2] if len(values) > 1 else 0.0
    )
    output["top1_top2_gap"] = gaps
    return output


def tie_aware_accuracy(q_values: np.ndarray, predicted_actions: np.ndarray, tolerance: float) -> float:
    """Return credit when the selected action lies within tolerance of the best."""
    values = np.asarray(q_values, dtype=float)
    predictions = np.asarray(predicted_actions, dtype=int)
    if values.ndim != MATRIX_DIMENSIONS or predictions.shape != (values.shape[0],):
        raise ValueError("q_values/predicted_actions shapes are incompatible.")
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Tie tolerance must be finite and non-negative.")
    selected = values[np.arange(len(values)), predictions]
    return float(np.mean(np.max(values, axis=1) - selected <= tolerance))


@dataclass(frozen=True)
class SelectorValues:
    """Validation values for the prescribed executable selector hierarchy."""

    global_static: float
    contextual_static: float
    budget_only: float
    context_phase: float
    oracle: float

    @property
    def context_gain(self) -> float:
        """Return validation gain from static task-context routing."""
        return self.contextual_static - self.global_static

    @property
    def phase_gain(self) -> float:
        """Return validation gain from adding an open-loop phase schedule."""
        return self.context_phase - self.contextual_static

    @property
    def feedback_oracle_gain(self) -> float:
        """Return residual same-state feedback-oracle gain."""
        return self.oracle - self.context_phase


def _context_key(row: Mapping[str, Any]) -> str:
    if row["domain"] == "yahpo":
        return f"yahpo:{row['scenario']}"
    return f"bbob:d{int(row['dimension'])}:g{int(row['function_group'])}"


def _phase(fraction: float) -> str:
    if fraction < EARLY_MIDDLE_BOUNDARY:
        return "early"
    if fraction < MIDDLE_LATE_BOUNDARY:
        return "middle"
    return "late"


def _fit_action_map(
    train: pd.DataFrame,
    key_columns: Sequence[str],
    *,
    minimum_support: int,
    fallback_action: int,
) -> dict[tuple[Any, ...], int]:
    mapping: dict[tuple[Any, ...], int] = {}
    for key, group in train.groupby(list(key_columns), dropna=False, sort=True):
        normalized_key = key if isinstance(key, tuple) else (key,)
        if group["snapshot_id"].nunique() < minimum_support:
            mapping[normalized_key] = fallback_action
            continue
        means = group.groupby("action", sort=True)["q_value"].mean()
        mapping[normalized_key] = int(means[means == means.max()].index.min())
    return mapping


def evaluate_selector_decomposition(
    train_rows: pd.DataFrame,
    validation_rows: pd.DataFrame,
    *,
    minimum_support: int = 2,
) -> SelectorValues:
    """Fit every deployable selector on training rows and evaluate validation once."""
    assert_disjoint_context_splits(train_rows, validation_rows)
    train = train_rows.copy()
    validation = validation_rows.copy()
    for frame in (train, validation):
        frame["context_key"] = [_context_key(row) for row in frame.to_dict("records")]
        frame["phase"] = frame["budget_fraction"].map(_phase)

    action_means = train.groupby("action", sort=True)["q_value"].mean()
    global_action = int(action_means[action_means == action_means.max()].index.min())
    context_map = _fit_action_map(
        train, ["context_key"], minimum_support=minimum_support, fallback_action=global_action
    )
    phase_map = _fit_action_map(train, ["phase"], minimum_support=minimum_support, fallback_action=global_action)
    context_phase_map = _fit_action_map(
        train, ["context_key", "phase"], minimum_support=minimum_support, fallback_action=global_action
    )

    matrices = {
        snapshot_id: group.set_index("action")["q_value"]
        for snapshot_id, group in validation.groupby("snapshot_id", sort=True)
    }
    metadata = validation.drop_duplicates("snapshot_id").set_index("snapshot_id")

    def value(action_for: Any) -> float:
        selected = []
        for snapshot_id, matrix in matrices.items():
            action = int(action_for(metadata.loc[snapshot_id]))
            selected.append(float(matrix.loc[action]))
        return float(np.mean(selected))

    return SelectorValues(
        global_static=value(lambda _row: global_action),
        contextual_static=value(lambda row: context_map.get((row["context_key"],), global_action)),
        budget_only=value(lambda row: phase_map.get((row["phase"],), global_action)),
        context_phase=value(lambda row: context_phase_map.get((row["context_key"], row["phase"]), global_action)),
        oracle=float(np.mean([matrix.max() for matrix in matrices.values()])),
    )


class FlatObservationMLP(nn.Module):
    """Offline analogue of a flat Dict-observation actor representation."""

    def __init__(self, global_size: int, n_actions: int, action_feature_size: int) -> None:
        super().__init__()
        input_size = global_size + n_actions * action_feature_size
        self.network = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )

    def forward(self, global_state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Score all actions from one flattened observation."""
        return self.network(torch.cat((global_state, action_features.flatten(start_dim=1)), dim=1))


class SharedActionScorer(nn.Module):
    """Permutation-equivariant shared row encoder and scorer."""

    def __init__(self, global_size: int, action_feature_size: int) -> None:
        super().__init__()
        self.global_encoder = nn.Sequential(nn.Linear(global_size, 32), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_feature_size, 32), nn.ReLU())
        self.scorer = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, global_state: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Score action rows with shared weights."""
        global_embedding = self.global_encoder(global_state)
        action_embedding = self.action_encoder(action_features)
        expanded_global = global_embedding[:, None, :].expand(-1, action_features.shape[1], -1)
        return self.scorer(torch.cat((expanded_global, action_embedding), dim=2)).squeeze(-1)


def fit_ridge_action_model(
    train: pd.DataFrame,
    *,
    categorical: Sequence[str],
    numeric: Sequence[str],
) -> Pipeline:
    """Fit a fixed regularized diagnostic model on training data only."""
    predictors = tuple(categorical) + tuple(numeric) + ("action",)
    assert_predictor_columns_safe(predictors)
    transformer = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), [*list(categorical), "action"]),
            ("numeric", StandardScaler(), list(numeric)),
        ]
    )
    model = Pipeline([("features", transformer), ("ridge", Ridge(alpha=1.0))])
    return model.fit(train[list(predictors)], train["action_advantage"])


def grouped_bootstrap_mean(
    rows: pd.DataFrame,
    value_column: str,
    hierarchy: Sequence[str],
    *,
    n_resamples: int,
    seed: int,
) -> np.ndarray:
    """Hierarchically resample complete child groups without splitting branches."""
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive.")
    if missing := sorted((set(hierarchy) | {value_column}) - set(rows)):
        raise ValueError(f"Bootstrap table lacks columns {missing!r}.")
    rng = np.random.default_rng(seed)

    def build_tree(frame: pd.DataFrame, depth: int) -> Any:
        if depth == len(hierarchy):
            return frame[value_column].to_numpy(dtype=float)
        return [
            build_tree(group, depth + 1) for _key, group in frame.groupby(hierarchy[depth], sort=True, dropna=False)
        ]

    tree = build_tree(rows, 0)

    def sample_tree(node: Any, depth: int) -> list[float]:
        if depth == len(hierarchy):
            return np.asarray(node, dtype=float).tolist()
        choices = rng.integers(len(node), size=len(node))
        values: list[float] = []
        for choice in choices:
            values.extend(sample_tree(node[int(choice)], depth + 1))
        return values

    return np.asarray([np.mean(sample_tree(tree, 0)) for _ in range(n_resamples)], dtype=float)


def campaign_manifest_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash a deterministic exact snapshot inventory."""
    payload = json.dumps(list(rows), allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def write_parquet_atomic(rows: pd.DataFrame, path: Path) -> None:
    """Write a machine-readable table after validating finite numeric values."""
    numeric = rows.select_dtypes(include=[np.number])
    numeric_values = numeric.to_numpy(dtype=float)
    if np.isinf(numeric_values).any():
        raise ValueError("Refusing to write non-finite analysis values.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    rows.to_parquet(temporary, index=False)
    temporary.replace(path)


__all__ = [
    "DEFAULT_TIE_TOLERANCES",
    "FlatObservationMLP",
    "SelectorValues",
    "SharedActionScorer",
    "add_action_advantages",
    "assert_disjoint_context_splits",
    "assert_predictor_columns_safe",
    "campaign_manifest_hash",
    "evaluate_selector_decomposition",
    "fit_ridge_action_model",
    "grouped_bootstrap_mean",
    "parse_policy_observation",
    "tie_aware_accuracy",
    "write_parquet_atomic",
]
