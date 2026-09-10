"""Training-side gate tuning and probability calibration."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.selective_wei.context import canonical_hash

BINARY_CLASS_COUNT = 2

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GateConstraints:
    """Predeclared risk and support constraints for gate selection."""

    delta_harm: float = 0.1
    epsilon_harm: float = 1e-3
    minimum_override_count: int = 5
    minimum_override_tasks: int = 3
    minimum_coverage: float = 0.0
    maximum_coverage: float = 1.0
    minimum_tasks_per_context: int = 2


def task_balanced_mean(values: NDArray[np.floating], task_ids: NDArray[np.str_]) -> float:
    """Average states within task before averaging tasks."""
    tasks = np.asarray(task_ids).astype(str)
    array = np.asarray(values, dtype=np.float64)
    return float(np.mean([array[tasks == task].mean() for task in sorted(set(tasks.tolist()))]))


def risk_coverage_gain_row(
    selected_actions: NDArray[np.integer],
    base_actions: NDArray[np.integer],
    q_values: NDArray[np.floating],
    task_ids: NDArray[np.str_],
    contexts: NDArray[np.str_],
    *,
    parameters: dict[str, Any],
    constraints: GateConstraints,
) -> dict[str, Any]:
    """Evaluate one gate setting with task-balanced gain and explicit harm."""
    selected = np.asarray(selected_actions, dtype=np.int64)
    base = np.asarray(base_actions, dtype=np.int64)
    q = np.asarray(q_values, dtype=np.float64)
    rows = np.arange(len(selected))
    gain = q[rows, selected] - q[rows, base]
    override = selected != base
    harmful = override & (gain < -constraints.epsilon_harm)
    override_tasks = set(np.asarray(task_ids).astype(str)[override].tolist())
    context_support = {
        context: len(
            set(np.asarray(task_ids).astype(str)[override & (np.asarray(contexts).astype(str) == context)].tolist())
        )
        for context in sorted(set(np.asarray(contexts).astype(str)[override].tolist()))
    }
    coverage = float(override.mean())
    harm_fraction = float(harmful.sum() / override.sum()) if override.any() else None
    feasible = (
        harm_fraction is not None
        and harm_fraction <= constraints.delta_harm
        and int(override.sum()) >= constraints.minimum_override_count
        and len(override_tasks) >= constraints.minimum_override_tasks
        and constraints.minimum_coverage <= coverage <= constraints.maximum_coverage
        and all(value >= constraints.minimum_tasks_per_context for value in context_support.values())
    )
    return {
        "parameters": parameters,
        "parameter_hash": canonical_hash(parameters),
        "task_balanced_mean_gain": task_balanced_mean(gain, np.asarray(task_ids).astype(str)),
        "state_mean_gain": float(gain.mean()),
        "override_coverage": coverage,
        "override_count": int(override.sum()),
        "override_task_count": len(override_tasks),
        "harmful_override_fraction": harm_fraction,
        "mean_gain_when_overriding": float(gain[override].mean()) if override.any() else 0.0,
        "mean_harm_when_harmful": float(gain[harmful].mean()) if harmful.any() else 0.0,
        "context_task_support": context_support,
        "feasible": feasible,
    }


def select_frontier_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Include abstention in the saved frontier and maximize supported gain."""
    if not any(row.get("fallback") == "base_only_no_feasible_override_gate" for row in rows):
        rows.append(
            {
                "parameters": {},
                "parameter_hash": canonical_hash({}),
                "task_balanced_mean_gain": 0.0,
                "override_coverage": 0.0,
                "override_count": 0,
                "override_task_count": 0,
                "harmful_override_fraction": None,
                "feasible": True,
                "fallback": "base_only_no_feasible_override_gate",
            }
        )
    feasible = [row for row in rows if row["feasible"]]
    return sorted(
        feasible,
        key=lambda row: (
            -row["task_balanced_mean_gain"],
            row["harmful_override_fraction"] or 0.0,
            row["override_coverage"],
            row["parameter_hash"],
        ),
    )[0]


class ProbabilityCalibrator:
    """Logistic or isotonic benefit-probability calibration."""

    def __init__(self, method: Literal["logistic", "isotonic"]) -> None:
        self.method = method
        self.model: LogisticRegression | IsotonicRegression | None = None

    def fit(self, scores: NDArray[np.floating], labels: NDArray[np.bool_]) -> ProbabilityCalibrator:
        """Fit only from training-side gate-tuning examples."""
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        y = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(np.unique(y)) < BINARY_CLASS_COUNT:
            raise ValueError("Probability calibration requires both benefit classes.")
        if self.method == "logistic":
            self.model = LogisticRegression(C=1.0, class_weight="balanced", random_state=0).fit(x[:, None], y)
        else:
            self.model = IsotonicRegression(out_of_bounds="clip").fit(x, y)
        return self

    def predict(self, scores: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return calibrated benefit probabilities."""
        if self.model is None:
            raise RuntimeError("Probability calibrator has not been fit.")
        x = np.asarray(scores, dtype=np.float64).reshape(-1)
        if isinstance(self.model, LogisticRegression):
            return cast("NDArray[np.float64]", self.model.predict_proba(x[:, None])[:, 1])
        return np.asarray(self.model.predict(x), dtype=np.float64)


class OverrideWorthwhileClassifier:
    """G6 stage-one classifier with explicit class balancing."""

    def __init__(self, kind: Literal["shared_neural", "extra_trees", "hist_gradient_boosting"], seed: int) -> None:
        self.kind = kind
        if kind == "extra_trees":
            self.model: Any = ExtraTreesClassifier(
                n_estimators=256, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=seed
            )
        elif kind == "hist_gradient_boosting":
            self.model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=seed
            )
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True, random_state=seed, max_iter=300
            )

    def fit(
        self,
        features: NDArray[np.floating],
        true_residuals: NDArray[np.floating],
        *,
        epsilon_worthwhile: float,
    ) -> OverrideWorthwhileClassifier:
        """Fit worthwhile labels without assigning tied best-action labels."""
        residual = np.asarray(true_residuals, dtype=np.float64)
        labels = residual.max(axis=1) > epsilon_worthwhile
        if len(np.unique(labels)) < BINARY_CLASS_COUNT:
            raise ValueError("Override-worthwhile fit requires both classes.")
        weights = np.where(labels, len(labels) / (2 * labels.sum()), len(labels) / (2 * (~labels).sum()))
        if self.kind == "hist_gradient_boosting":
            self.model.fit(features, labels.astype(int), sample_weight=weights)
        else:
            self.model.fit(features, labels.astype(int), sample_weight=weights)
        return self

    def predict_probability(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return stage-one override-worthwhile probabilities."""
        return np.asarray(self.model.predict_proba(features)[:, 1], dtype=np.float64)


class ReliabilityClassifier:
    """Optional learned G9 probability that a prediction is reliable."""

    def __init__(self, seed: int = 0) -> None:
        self.model = ExtraTreesClassifier(
            n_estimators=256,
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
        )

    def fit(
        self,
        trust_features: NDArray[np.floating],
        absolute_selected_error: NDArray[np.floating],
        *,
        maximum_reliable_error: float,
    ) -> ReliabilityClassifier:
        """Fit on train-side errors using only deployable trust features."""
        labels = np.asarray(absolute_selected_error) <= maximum_reliable_error
        if len(np.unique(labels)) < BINARY_CLASS_COUNT:
            raise ValueError("Reliability fitting requires reliable and unreliable examples.")
        self.model.fit(np.asarray(trust_features, dtype=np.float64), labels.astype(int))
        return self

    def predict_probability(self, trust_features: NDArray[np.floating]) -> NDArray[np.float64]:
        """Return probability of a reliable delayed-value prediction."""
        return np.asarray(self.model.predict_proba(trust_features)[:, 1], dtype=np.float64)


def save_calibration_model(model: Any, path: Path, *, kind: str) -> dict[str, str]:
    """Persist one fitted training-side calibrator with an explicit hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(pickle.dumps(model))
    temporary.replace(path)
    return {"path": str(path), "sha256": file_sha256(path), "kind": kind}


def load_calibration_model(artifact: dict[str, str] | None) -> Any | None:
    """Load a hash-verified local calibrator, never inferring its kind."""
    if artifact is None:
        return None
    path = Path(artifact["path"])
    if not path.is_file() or file_sha256(path) != artifact["sha256"]:
        raise ValueError("Selective calibration-model artifact hash mismatch.")
    return pickle.loads(path.read_bytes())  # noqa: S301 - verified scientific artifact


def calibration_schema() -> dict[str, Any]:
    """Return a machine-readable calibration contract."""
    return {
        "schema_version": "dacbo-selective-calibration-v1",
        "selection_unit": "task",
        "objective": "task_balanced_mean_residual_gain",
        "constraints": asdict(GateConstraints()),
        "epsilon_grid": [0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        "kappa_grid": [0, 0.5, 1, 1.5, 2, 3],
        "p_min_grid": [0.5, 0.7, 0.8, 0.9, 0.95],
        "delta_harm_grid": [0.05, 0.1, 0.2],
        "epsilon_harm_grid": [1e-3, 5e-3, 1e-2],
    }
