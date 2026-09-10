"""Residual uncertainty and candidate-equivalence utilities."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np

from dacboenv.env.observation import GLOBAL_STATE_INDEX
from dacboenv.selective_wei.schemas import (
    ACTION_COUNT,
    CalibrationArtifact,
    CandidateEquivalence,
    Horizon,
    ResidualPrediction,
    ValuePrediction,
)

SHORT_HORIZON = 5
MATRIX_NDIM = 2

if TYPE_CHECKING:
    from numpy.typing import NDArray


def residual_advantages(
    prediction: ValuePrediction,
    base_action: int,
    horizon: Horizon,
) -> ResidualPrediction:
    """Residualize each ensemble member before estimating uncertainty."""
    if not 0 <= base_action < ACTION_COUNT:
        raise ValueError("Base action is outside the WEI grid.")
    mean_q = prediction.q5_mean if horizon == SHORT_HORIZON else prediction.q10_mean
    member_q = prediction.q5_member_values if horizon == SHORT_HORIZON else prediction.q10_member_values
    if mean_q is None:
        raise ValueError(f"Predictor does not provide H={horizon} values.")
    if member_q is not None:
        members = np.asarray(member_q, dtype=np.float64) - np.asarray(member_q, dtype=np.float64)[:, [base_action]]
        mean = members.mean(axis=0)
        std = members.std(axis=0, ddof=1) if len(members) > 1 else np.zeros(ACTION_COUNT, dtype=np.float64)
    else:
        members = None
        mean = np.asarray(mean_q, dtype=np.float64) - float(np.asarray(mean_q)[base_action])
        std = None
    mean[base_action] = 0.0
    if members is not None:
        members[:, base_action] = 0.0
    if std is not None:
        std[base_action] = 0.0
    return ResidualPrediction(horizon=horizon, base_action=base_action, mean=mean, members=members, std=std)


def candidate_equivalence(
    action_features: NDArray[np.floating],
    *,
    tolerance: float = 1e-6,
    candidate_hashes: Sequence[str] | None = None,
    heuristic: bool = False,
) -> CandidateEquivalence:
    """Group immediate candidates, optionally using a compressed-feature heuristic.

    Neither kind proves equal delayed continuation values. Without hashes the
    default leaves actions distinct; feature equality is an explicit ablation.
    """
    features = np.asarray(action_features, dtype=np.float64)
    if features.shape[0] != ACTION_COUNT or features.ndim != MATRIX_NDIM:
        raise ValueError("Action features must have shape (5, features).")
    if candidate_hashes is not None:
        if len(candidate_hashes) != ACTION_COUNT:
            raise ValueError("Candidate hash list must have five entries.")
        keys: list[object] = list(candidate_hashes)
        source = "candidate_hash"
        mapping: dict[object, int] = {}
        groups: list[int] = []
        for key in keys:
            mapping.setdefault(key, len(mapping))
            groups.append(mapping[key])
    elif not heuristic:
        return CandidateEquivalence(groups=tuple(range(ACTION_COUNT)), tolerance=tolerance, source="not_established")
    else:
        consequences = features[:, 1:]
        if not np.isfinite(consequences).all():
            raise ValueError("Candidate consequence features must be finite.")
        if tolerance < 0.0:
            raise ValueError("Candidate-equivalence tolerance must be nonnegative.")
        representatives: list[NDArray[np.float64]] = []
        groups = []
        for row in consequences:
            matching_group = next(
                (
                    index
                    for index, representative in enumerate(representatives)
                    if np.allclose(row, representative, rtol=0.0, atol=tolerance)
                ),
                None,
            )
            if matching_group is None:
                matching_group = len(representatives)
                representatives.append(row.copy())
            groups.append(matching_group)
        source = "compressed_feature_heuristic"
    return CandidateEquivalence(groups=tuple(groups), tolerance=tolerance, source=source)


def empirical_probability(
    residual_members: NDArray[np.floating], threshold: float, *, greater: bool
) -> NDArray[np.float64]:
    """Estimate benefit or harm probability from paired residual members."""
    members = np.asarray(residual_members, dtype=np.float64)
    if members.ndim != MATRIX_NDIM or members.shape[1] != ACTION_COUNT:
        raise ValueError("Residual members must have shape (members, 5).")
    values = members > threshold if greater else members < threshold
    return np.asarray(values.mean(axis=0, dtype=np.float64), dtype=np.float64)


def finite_sample_conformal_quantile(values: Sequence[float], delta: float) -> float:
    """Return the ceil((n+1)(1-delta)) one-sided finite-sample quantile."""
    residuals = np.sort(np.asarray(values, dtype=np.float64))
    if not 0.0 < delta < 1.0:
        raise ValueError("Conformal delta must lie strictly between zero and one.")
    if residuals.ndim != 1 or not np.isfinite(residuals).all():
        raise ValueError("Conformal scores must be a finite one-dimensional array.")
    rank = math.ceil((len(residuals) + 1) * (1.0 - delta))
    if rank > len(residuals):
        return float("inf")
    return float(residuals[rank - 1])


def calibrate_task_max_conformal(
    predicted: NDArray[np.floating],
    truth: NDArray[np.floating],
    task_ids: Sequence[str],
    *,
    base_actions: Sequence[int],
    delta: float,
    horizon: Horizon,
    task_hash: str,
    minimum_tasks: int = 5,
    mode: str = "task_max_simultaneous",
) -> CalibrationArtifact:
    """Fit a simultaneous one-sided error bound with task as the unit."""
    predictions = np.asarray(predicted, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64)
    tasks = np.asarray(task_ids).astype(str)
    bases = np.asarray(base_actions, dtype=np.int64)
    if predictions.shape != targets.shape or predictions.shape != (len(tasks), ACTION_COUNT):
        raise ValueError("Conformal prediction/truth arrays must have shape (states, 5).")
    if mode not in {"task_max_simultaneous", "state_level", "domain_mondrian"}:
        raise ValueError(f"Unsupported conformal mode {mode!r}.")
    calibration_tasks = sorted(set(tasks.tolist()))
    errors: list[float] = []
    if mode == "state_level":
        for index in range(len(tasks)):
            valid_actions = np.arange(ACTION_COUNT) != bases[index]
            errors.append(float(np.max((predictions[index] - targets[index])[valid_actions])))
    else:
        for task in calibration_tasks:
            mask = tasks == task
            residuals = predictions[mask] - targets[mask]
            valid = np.ones_like(residuals, dtype=bool)
            valid[np.arange(mask.sum()), bases[mask]] = False
            errors.append(float(residuals[valid].max(initial=-np.inf)))
    if len(calibration_tasks) < minimum_tasks:
        return CalibrationArtifact(
            calibration_id="conformal-unavailable",
            mode=mode,
            horizon=horizon,
            task_ids=tuple(calibration_tasks),
            task_hash=task_hash,
            nominal_delta=delta,
            residual_quantile=float("inf"),
            realized_coverage=None,
            fallback_mode="insufficient_tasks",
            minimum_tasks=minimum_tasks,
        )
    quantile = finite_sample_conformal_quantile(errors, delta)
    return CalibrationArtifact(
        calibration_id=f"conformal-{mode}-h{horizon}",
        mode=mode,
        horizon=horizon,
        task_ids=tuple(calibration_tasks),
        task_hash=task_hash,
        nominal_delta=delta,
        residual_quantile=quantile,
        realized_coverage=float(np.mean(np.asarray(errors) <= quantile)) if np.isfinite(quantile) else None,
        fallback_mode=None if np.isfinite(quantile) else "unsupported_order_statistic",
        minimum_tasks=minimum_tasks,
    )


def conformal_lower_bounds(residual_mean: NDArray[np.floating], artifact: CalibrationArtifact) -> NDArray[np.float64]:
    """Apply a frozen one-sided task-level conformal correction."""
    result = np.asarray(residual_mean, dtype=np.float64) - artifact.residual_quantile
    if not np.isfinite(artifact.residual_quantile):
        result[:] = -np.inf
    return result


def mondrian_or_global(
    artifacts: Mapping[str, CalibrationArtifact],
    stratum: str,
    global_artifact: CalibrationArtifact,
) -> CalibrationArtifact:
    """Use a supported Mondrian artifact or fail conservatively to global."""
    artifact = artifacts.get(stratum)
    if artifact is None or artifact.fallback_mode is not None:
        return global_artifact
    return artifact


def hard_trust_metadata(
    global_state: NDArray[np.floating],
    action_features: NDArray[np.floating],
    equivalence: CandidateEquivalence,
    parameters: Mapping[str, object],
    *,
    gp_available: bool | None = None,
) -> dict[str, object]:
    """Evaluate a frozen trust rule using only deployable reliability features."""
    state = np.asarray(global_state, dtype=np.float64)
    features = np.asarray(action_features, dtype=np.float64)
    consequence_spread = float(np.max(np.ptp(features[:, 1:], axis=0), initial=0.0))
    duplicate_fraction = 1.0 - len(set(equivalence.groups)) / ACTION_COUNT
    calibration_error = float(state[GLOBAL_STATE_INDEX["calibration_error"]])
    reasons: list[str] = []
    minimum_spread = parameters.get("minimum_action_feature_spread", 0.0)
    maximum_duplicates = parameters.get("maximum_duplicate_fraction", 1.0)
    maximum_calibration_value = parameters.get("maximum_calibration_error", float("inf"))
    if not isinstance(minimum_spread, (int, float)) or not isinstance(maximum_duplicates, (int, float)):
        raise TypeError("Trust thresholds must be numeric.")
    if not isinstance(maximum_calibration_value, (int, float)):
        raise TypeError("Maximum calibration error must be numeric.")
    if consequence_spread < float(minimum_spread):
        reasons.append("candidate_spread")
    if duplicate_fraction > float(maximum_duplicates):
        reasons.append("duplicate_fraction")
    maximum_calibration = float(maximum_calibration_value)
    if calibration_error >= 0.0 and calibration_error > maximum_calibration:
        reasons.append("calibration_error")
    if bool(parameters.get("require_gp_available", False)) and gp_available is not True:
        reasons.append("gp_unavailable")
    return {
        "trust_probability": float(not reasons),
        "trust_reason": "trusted" if not reasons else "+".join(reasons),
        "candidate_consequence_spread": consequence_spread,
        "candidate_duplicate_fraction": duplicate_fraction,
        "budget_density": float(state[GLOBAL_STATE_INDEX["rho_B"]]),
        "calibration_error": calibration_error,
        "gp_available": gp_available,
    }


def trust_feature_vector(metadata: Mapping[str, object]) -> NDArray[np.float64]:
    """Encode fixed deployable trust metadata for the optional G9 classifier."""
    gp = metadata.get("gp_available")
    values = [
        metadata["budget_density"],
        metadata["calibration_error"],
        metadata["candidate_consequence_spread"],
        metadata["candidate_duplicate_fraction"],
    ]
    if any(not isinstance(value, (int, float)) for value in values):
        raise TypeError("Trust feature metadata must be numeric.")
    numeric_values = cast("list[int | float]", values)
    return np.asarray(
        [
            *(float(value) for value in numeric_values),
            -1.0 if gp is None else float(bool(gp)),
        ],
        dtype=np.float64,
    )
