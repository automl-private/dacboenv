"""Typed scientific contracts for selective WEI control."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

ACTION_COUNT = 5
MATRIX_NDIM = 2
ALPHA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
Horizon = Literal[5, 10]


@dataclass(frozen=True, slots=True)
class StaticContext:
    """Deployment-time context without privileged landscape identity."""

    domain: Literal["bbob", "yahpo", "optbench"]
    dimension: int | None = None
    scenario: str | None = None
    phase_bin: int | None = None

    def validate(self) -> None:
        """Reject incomplete or internally inconsistent contexts."""
        if self.domain not in {"bbob", "yahpo", "optbench"}:
            raise ValueError("Unsupported deployment domain.")
        if self.domain in {"bbob", "optbench"} and (self.dimension is None or self.dimension <= 0):
            raise ValueError("Numeric benchmark deployment context requires a positive dimension.")
        if self.domain == "yahpo" and not self.scenario:
            raise ValueError("YAHPO deployment context requires a scenario.")


@dataclass(frozen=True, slots=True)
class BaseDecision:
    """One frozen contextual base-policy decision with provenance."""

    action: int
    alpha: float
    selector_id: str
    matched_context: str
    fallback_level: Literal["exact", "domain", "global", "emergency"]
    horizon: Horizon
    fit_manifest_hash: str
    fit_data_hash: str
    action_probabilities: tuple[float, ...] | None = None
    expected_base_value: float | None = None
    unique_task_support: int | None = None


@dataclass(frozen=True, slots=True)
class ValuePrediction:
    """Delayed fixed-action values and optional ensemble members."""

    q5_mean: NDArray[np.float64]
    q10_mean: NDArray[np.float64] | None = None
    q5_member_values: NDArray[np.float64] | None = None
    q10_member_values: NDArray[np.float64] | None = None
    q5_std: NDArray[np.float64] | None = None
    q10_std: NDArray[np.float64] | None = None
    model_ids: tuple[str, ...] = ()
    model_hashes: tuple[str, ...] = ()
    normalizer_hash: str = ""
    feature_schema_hash: str = ""
    prediction_status: str = "ok"

    def __post_init__(self) -> None:
        """Validate action axes and finite successful predictions."""
        for name in ("q5_mean", "q10_mean", "q5_std", "q10_std"):
            value = getattr(self, name)
            if value is not None and np.asarray(value).shape != (ACTION_COUNT,):
                raise ValueError(f"{name} must have shape (5,).")
        for name in ("q5_member_values", "q10_member_values"):
            value = getattr(self, name)
            if value is not None and (
                np.asarray(value).ndim != MATRIX_NDIM or np.asarray(value).shape[1] != ACTION_COUNT
            ):
                raise ValueError(f"{name} must have shape (members, 5).")
        if self.prediction_status == "ok":
            values = [value for value in (self.q5_mean, self.q10_mean) if value is not None]
            if any(not np.isfinite(np.asarray(value)).all() for value in values):
                raise ValueError("Successful delayed-value predictions must be finite.")


@dataclass(frozen=True, slots=True)
class ResidualPrediction:
    """Base-relative delayed advantages, residualized member by member."""

    horizon: Horizon
    base_action: int
    mean: NDArray[np.float64]
    members: NDArray[np.float64] | None
    std: NDArray[np.float64] | None


@dataclass(frozen=True, slots=True)
class CandidateEquivalence:
    """Equivalence classes derived from deployable candidate consequences."""

    groups: tuple[int, ...]
    tolerance: float
    source: str

    def equivalent(self, left: int, right: int) -> bool:
        """Return whether two action rows lead to equivalent candidates."""
        return self.groups[left] == self.groups[right]


@dataclass(slots=True)
class GateInputs:
    """Complete deployable inputs to an override gate."""

    base: BaseDecision
    prediction: ValuePrediction
    residual5: ResidualPrediction
    residual10: ResidualPrediction | None
    equivalence: CandidateEquivalence
    current_action: int | None = None
    blocks_since_switch: int = 10**9
    lower_bounds5: NDArray[np.float64] | None = None
    lower_bounds10: NDArray[np.float64] | None = None
    probability_benefit5: NDArray[np.float64] | None = None
    probability_harm5: NDArray[np.float64] | None = None
    override_probability: float | None = None
    model_trusted: bool = True
    trust_reason: str = "trusted"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OverrideDecision:
    """Auditable result of one selective override decision."""

    base_action: int
    selected_action: int
    override: bool
    reason: str
    horizon: Horizon
    predicted_advantages: tuple[float, ...]
    lower_bounds: tuple[float, ...] | None
    probability_of_benefit: tuple[float, ...] | None
    probability_of_harm: tuple[float, ...] | None
    predicted_margin: float
    candidate_distinct: bool
    model_trusted: bool
    gate_parameters: dict[str, Any]
    gate_id: str
    equivalence_class: int
    fallback_status: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationArtifact:
    """Frozen task-grouped uncertainty calibration result."""

    calibration_id: str
    mode: str
    horizon: Horizon
    task_ids: tuple[str, ...]
    task_hash: str
    nominal_delta: float
    residual_quantile: float
    realized_coverage: float | None
    fallback_mode: str | None
    minimum_tasks: int

    def to_dict(self) -> dict[str, Any]:
        """Encode unavailable bounds without nonstandard JSON infinity."""
        payload = asdict(self)
        finite = bool(np.isfinite(self.residual_quantile))
        payload["residual_quantile"] = self.residual_quantile if finite else None
        payload["bound_status"] = "finite" if finite else "unbounded"
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CalibrationArtifact:
        """Restore strict JSON and previously serialized internal infinities."""
        data = dict(payload)
        status = data.pop("bound_status", None)
        if data["residual_quantile"] is None:
            if status != "unbounded":
                raise ValueError("Null conformal quantile requires unbounded status.")
            data["residual_quantile"] = float("inf")
        elif status not in (None, "finite"):
            raise ValueError("Inconsistent conformal bound status.")
        data["task_ids"] = tuple(data["task_ids"])
        return cls(**data)
