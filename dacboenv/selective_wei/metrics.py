"""Value, selectivity, risk, prediction, and grouped uncertainty metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from dacboenv.selective_wei.calibration import task_balanced_mean

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _captured(selected: float, base: float, oracle: float, epsilon: float) -> float | None:
    denominator = oracle - base
    return None if abs(denominator) < epsilon else (selected - base) / denominator


def evaluate_selective_actions(
    q_values: NDArray[np.floating],
    selected_actions: NDArray[np.integer],
    base_actions: NDArray[np.integer],
    task_ids: NDArray[np.str_],
    *,
    equivalent_override: NDArray[np.bool_] | None = None,
    epsilon_harm: float = 1e-3,
    denominator_epsilon: float = 1e-12,
) -> dict[str, Any]:
    """Compute primary base-relative metrics with local denominators."""
    q = np.asarray(q_values, dtype=np.float64)
    selected = np.asarray(selected_actions, dtype=np.int64)
    base = np.asarray(base_actions, dtype=np.int64)
    tasks = np.asarray(task_ids).astype(str)
    rows = np.arange(len(q))
    selected_value = q[rows, selected]
    base_value = q[rows, base]
    oracle_value = q.max(axis=1)
    gain = selected_value - base_value
    override = selected != base
    harm = override & (gain < 0)
    material_harm = override & (gain < -epsilon_harm)
    equivalent = np.zeros(len(q), dtype=bool) if equivalent_override is None else np.asarray(equivalent_override)
    aggregate_selected = task_balanced_mean(selected_value, tasks)
    aggregate_base = task_balanced_mean(base_value, tasks)
    aggregate_oracle = task_balanced_mean(oracle_value, tasks)
    return {
        "selected_value": aggregate_selected,
        "base_value": aggregate_base,
        "selected_minus_base": task_balanced_mean(gain, tasks),
        "oracle_value": aggregate_oracle,
        "oracle_minus_base": aggregate_oracle - aggregate_base,
        "captured_residual_headroom": _captured(
            aggregate_selected, aggregate_base, aggregate_oracle, denominator_epsilon
        ),
        "override_coverage": float(override.mean()),
        "override_count": int(override.sum()),
        "override_task_count": len(set(tasks[override].tolist())),
        "base_action_retention_rate": float((~override).mean()),
        "equivalent_candidate_override_fraction": float(equivalent[override].mean()) if override.any() else 0.0,
        "harmful_override_fraction": float(harm.sum() / override.sum()) if override.any() else None,
        "materially_harmful_override_fraction": float(material_harm.sum() / override.sum()) if override.any() else None,
        "mean_harm_conditional": float(gain[harm].mean()) if harm.any() else 0.0,
        "maximum_observed_harm": float(gain.min(initial=0.0)),
        "conditional_benefit_when_overriding": float(gain[override].mean()) if override.any() else 0.0,
        "benefit_harm_ratio": (
            float(gain[override & (gain > 0)].sum() / max(abs(gain[harm].sum()), denominator_epsilon))
            if override.any()
            else None
        ),
    }


def grouped_bootstrap_gain(
    gains: NDArray[np.floating],
    task_ids: NDArray[np.str_],
    *,
    resamples: int = 2000,
    seed: int = 20260830,
) -> dict[str, float]:
    """Bootstrap tasks, preserving all within-task dependent states."""
    tasks = np.asarray(task_ids).astype(str)
    values = np.asarray(gains, dtype=np.float64)
    unique = np.asarray(sorted(set(tasks.tolist())))
    means = np.asarray([values[tasks == task].mean() for task in unique])
    rng = np.random.default_rng(seed)
    draws = means[rng.integers(len(means), size=(resamples, len(means)))].mean(axis=1)
    return {
        "mean": float(means.mean()),
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "resamples": float(resamples),
    }


def prediction_metrics(
    predicted_q: NDArray[np.floating],
    true_q: NDArray[np.floating],
    selected_actions: NDArray[np.integer],
    *,
    tie_tolerance: float = 1e-3,
    benefit_probabilities: NDArray[np.floating] | None = None,
    benefit_labels: NDArray[np.bool_] | None = None,
    lower_bounds: NDArray[np.floating] | None = None,
    true_residuals: NDArray[np.floating] | None = None,
) -> dict[str, float | None]:
    """Compute tie-aware ranking, calibration, and lower-bound diagnostics."""
    prediction = np.asarray(predicted_q, dtype=np.float64)
    truth = np.asarray(true_q, dtype=np.float64)
    selected = np.asarray(selected_actions, dtype=np.int64)
    rows = np.arange(len(truth))
    oracle = truth.max(axis=1)
    tie_correct = oracle - truth[rows, selected] <= tie_tolerance
    gaps = np.sort(truth, axis=1)[:, -1] - np.sort(truth, axis=1)[:, -2]
    pairwise: list[bool] = []
    for left in range(truth.shape[1]):
        for right in range(left + 1, truth.shape[1]):
            difference = truth[:, left] - truth[:, right]
            valid = np.abs(difference) > tie_tolerance
            pairwise.extend(np.sign(prediction[valid, left] - prediction[valid, right]) == np.sign(difference[valid]))
    rank_pred = np.argsort(np.argsort(prediction, axis=1), axis=1).astype(float)
    rank_true = np.argsort(np.argsort(truth, axis=1), axis=1).astype(float)
    rank_pred -= rank_pred.mean(axis=1, keepdims=True)
    rank_true -= rank_true.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(rank_pred, axis=1) * np.linalg.norm(rank_true, axis=1)
    spearman = np.divide(
        (rank_pred * rank_true).sum(axis=1),
        denominator,
        out=np.zeros(len(truth)),
        where=denominator > 0,
    )
    result: dict[str, float | None] = {
        "tie_aware_top1_accuracy": float(tie_correct.mean()),
        "gap_weighted_top1_accuracy": float(np.sum(tie_correct * gaps) / max(np.sum(gaps), 1e-12)),
        "pairwise_accuracy": float(np.mean(pairwise)) if pairwise else None,
        "spearman": float(spearman.mean()),
        "selected_action_regret": float((oracle - truth[rows, selected]).mean()),
    }
    if benefit_probabilities is not None and benefit_labels is not None:
        probability = np.asarray(benefit_probabilities, dtype=np.float64)
        labels = np.asarray(benefit_labels, dtype=np.float64)
        result["brier_score"] = float(np.mean((probability - labels) ** 2))
    else:
        result["brier_score"] = None
    if lower_bounds is not None and true_residuals is not None:
        lower = np.asarray(lower_bounds, dtype=np.float64)
        residual = np.asarray(true_residuals, dtype=np.float64)
        result["lower_bound_empirical_coverage"] = float(np.mean(lower <= residual))
        result["lower_bound_sharpness"] = float(np.mean(residual - lower))
    else:
        result["lower_bound_empirical_coverage"] = None
        result["lower_bound_sharpness"] = None
    return result


def stratified_value_metrics(
    q_values: NDArray[np.floating],
    selected_actions: NDArray[np.integer],
    base_actions: NDArray[np.integer],
    task_ids: NDArray[np.str_],
    strata: dict[str, NDArray[Any]],
) -> list[dict[str, Any]]:
    """Recompute local base/oracle denominators for each requested breakdown."""
    rows: list[dict[str, Any]] = []
    for stratum_name, values in strata.items():
        array = np.asarray(values)
        for level in np.unique(array):
            mask = array == level
            metrics = evaluate_selective_actions(
                np.asarray(q_values)[mask],
                np.asarray(selected_actions)[mask],
                np.asarray(base_actions)[mask],
                np.asarray(task_ids)[mask],
            )
            rows.append({"stratum": stratum_name, "level": str(level), "states": int(mask.sum()), **metrics})
    return rows


def feedback_classification(
    *,
    within_trajectory_variation: bool,
    selected_minus_base_lower: float,
    selected_minus_modal_lower: float,
    selected_minus_marginal_lower: float,
    captured_residual: float | None,
    harmful_override_fraction: float,
    maximum_acceptable_harm: float,
) -> str:
    """Apply conservative behavior/value labels, never switch-rate alone."""
    if not within_trajectory_variation:
        return "context_dynamic_only"
    if harmful_override_fraction > maximum_acceptable_harm:
        return "harmful_dynamic"
    if (
        selected_minus_base_lower > 0
        and selected_minus_modal_lower > 0
        and selected_minus_marginal_lower > 0
        and captured_residual is not None
        and captured_residual > 0
    ):
        return "feedback_dynamic_supported"
    if selected_minus_base_lower <= 0:
        return "inconclusive"
    return "selective_feedback_candidate"
