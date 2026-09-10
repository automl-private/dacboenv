"""Composable, fail-safe selective override gates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from dacboenv.selective_wei.schemas import ACTION_COUNT, GateInputs, Horizon, OverrideDecision
from dacboenv.selective_wei.uncertainty import empirical_probability

SHORT_HORIZON = 5


def _tuple(values: np.ndarray | None) -> tuple[float, ...] | None:
    return None if values is None else tuple(float(value) for value in values)


def _best_distinct(scores: np.ndarray, inputs: GateInputs, tie_tolerance: float) -> int:
    candidates = [
        action
        for action in range(ACTION_COUNT)
        if action != inputs.base.action and not inputs.equivalence.equivalent(action, inputs.base.action)
    ]
    if not candidates:
        return inputs.base.action
    best = max(float(scores[action]) for action in candidates)
    tied = [action for action in candidates if best - float(scores[action]) <= tie_tolerance]
    return min(tied)


class OverrideGate(ABC):
    """A gate whose public decision always fails closed to the base."""

    gate_id = "abstract"
    horizon: Horizon = 5

    def decide(self, inputs: GateInputs) -> OverrideDecision:
        """Return a selective decision or an auditable base fallback."""
        try:
            if inputs.prediction.prediction_status != "ok":
                raise ValueError(f"prediction_status={inputs.prediction.prediction_status}")
            if not inputs.model_trusted:
                return self._fallback(inputs, f"untrusted:{inputs.trust_reason}", "trust_blocked")
            if not np.isfinite(inputs.residual5.mean).all():
                raise ValueError("nonfinite residual prediction")
            return self._decide(inputs)
        except Exception as error:  # noqa: BLE001 - deployment must fail safely
            return self._fallback(inputs, f"fallback:{type(error).__name__}:{error}", "internal_fallback")

    @abstractmethod
    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        """Implement the scientific gate after common validation."""

    def parameters(self) -> dict[str, Any]:
        """Return JSON-compatible frozen gate parameters."""
        return {}

    def _decision(
        self,
        inputs: GateInputs,
        action: int,
        reason: str,
        *,
        lower_bounds: np.ndarray | None = None,
        p_benefit: np.ndarray | None = None,
        p_harm: np.ndarray | None = None,
    ) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None:
            raise ValueError(f"Gate requires H={self.horizon} values.")
        distinct = action != inputs.base.action and not inputs.equivalence.equivalent(action, inputs.base.action)
        selected = action if distinct else inputs.base.action
        override = selected != inputs.base.action
        return OverrideDecision(
            base_action=inputs.base.action,
            selected_action=selected,
            override=override,
            reason=reason if override else ("equivalent_candidate" if action != inputs.base.action else reason),
            horizon=self.horizon,
            predicted_advantages=tuple(float(value) for value in residual.mean),
            lower_bounds=_tuple(lower_bounds),
            probability_of_benefit=_tuple(p_benefit),
            probability_of_harm=_tuple(p_harm),
            predicted_margin=float(residual.mean[selected]),
            candidate_distinct=distinct,
            model_trusted=inputs.model_trusted,
            gate_parameters=self.parameters(),
            gate_id=self.gate_id,
            equivalence_class=inputs.equivalence.groups[selected],
        )

    def _fallback(self, inputs: GateInputs, reason: str, status: str) -> OverrideDecision:
        residual = inputs.residual5.mean if inputs.residual5 is not None else np.zeros(ACTION_COUNT)
        values = np.asarray(residual, dtype=np.float64)
        if values.shape != (ACTION_COUNT,) or not np.isfinite(values).all():
            values = np.zeros(ACTION_COUNT)
        return OverrideDecision(
            base_action=inputs.base.action,
            selected_action=inputs.base.action,
            override=False,
            reason=reason,
            horizon=self.horizon,
            predicted_advantages=tuple(float(value) for value in values),
            lower_bounds=None,
            probability_of_benefit=None,
            probability_of_harm=None,
            predicted_margin=0.0,
            candidate_distinct=False,
            model_trusted=inputs.model_trusted,
            gate_parameters=self.parameters(),
            gate_id=self.gate_id,
            equivalence_class=inputs.equivalence.groups[inputs.base.action],
            fallback_status=status,
        )


@dataclass(slots=True)
class BaseOnlyGate(OverrideGate):
    """G0: exact contextual base comparator."""

    gate_id = "G0_base_only"
    horizon: Horizon = 5

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        return self._decision(inputs, inputs.base.action, "base_only")


@dataclass(slots=True)
class PointThresholdGate(OverrideGate):
    """G1: override on a positive point residual above epsilon."""

    epsilon: float = 1e-3
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G1_point_threshold"

    def parameters(self) -> dict[str, Any]:
        """Return the point-threshold parameters."""
        return {"epsilon": self.epsilon, "tie_tolerance": self.tie_tolerance, "horizon": self.horizon}

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None:
            raise ValueError("Missing requested horizon.")
        action = _best_distinct(residual.mean, inputs, self.tie_tolerance)
        if action == inputs.base.action or residual.mean[action] <= self.epsilon:
            return self._decision(inputs, inputs.base.action, "advantage_below_threshold")
        return self._decision(inputs, action, "point_advantage")


@dataclass(slots=True)
class HysteresisGate(OverrideGate):
    """G2: point threshold plus switch margin, dwell, and distinctness."""

    epsilon_base: float = 1e-3
    epsilon_switch: float = 1e-3
    minimum_dwell_blocks: int = 1
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G2_hysteresis"

    def parameters(self) -> dict[str, Any]:
        """Return hysteresis and dwell parameters."""
        return {
            "epsilon_base": self.epsilon_base,
            "epsilon_switch": self.epsilon_switch,
            "minimum_dwell_blocks": self.minimum_dwell_blocks,
            "tie_tolerance": self.tie_tolerance,
            "horizon": self.horizon,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None:
            raise ValueError("Missing requested horizon.")
        action = _best_distinct(residual.mean, inputs, self.tie_tolerance)
        if action == inputs.base.action or residual.mean[action] <= self.epsilon_base:
            return self._decision(inputs, inputs.base.action, "advantage_below_threshold")
        current = inputs.base.action if inputs.current_action is None else inputs.current_action
        q = inputs.prediction.q5_mean if self.horizon == SHORT_HORIZON else inputs.prediction.q10_mean
        if q is None:
            raise ValueError("Missing requested Q prediction.")
        if inputs.blocks_since_switch < self.minimum_dwell_blocks:
            return self._decision(inputs, current, "minimum_dwell")
        if inputs.equivalence.equivalent(action, current) or q[action] - q[current] <= self.epsilon_switch:
            return self._decision(inputs, current, "switch_margin_or_equivalence")
        return self._decision(inputs, action, "hysteresis_override")


@dataclass(slots=True)
class EnsembleLCBGate(OverrideGate):
    """G3: primary ensemble lower-confidence-bound gate."""

    kappa: float = 1.0
    epsilon: float = 1e-3
    minimum_ensemble_size: int = 3
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G3_ensemble_lcb"

    def parameters(self) -> dict[str, Any]:
        """Return ensemble LCB parameters."""
        return {
            "kappa": self.kappa,
            "epsilon": self.epsilon,
            "minimum_ensemble_size": self.minimum_ensemble_size,
            "tie_tolerance": self.tie_tolerance,
            "horizon": self.horizon,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None or residual.members is None or len(residual.members) < self.minimum_ensemble_size:
            raise ValueError("insufficient ensemble")
        std = residual.std if residual.std is not None else residual.members.std(axis=0, ddof=1)
        lower = residual.mean - self.kappa * std
        lower[inputs.base.action] = 0.0
        action = _best_distinct(lower, inputs, self.tie_tolerance)
        if action == inputs.base.action or lower[action] <= self.epsilon:
            return self._decision(inputs, inputs.base.action, "lcb_below_threshold", lower_bounds=lower)
        return self._decision(inputs, action, "credible_ensemble_advantage", lower_bounds=lower)


@dataclass(slots=True)
class ProbabilitySuperiorityGate(OverrideGate):
    """G4: calibrated or empirical probability-of-benefit gate."""

    p_min: float = 0.9
    epsilon_benefit: float = 1e-3
    epsilon_mean: float | None = 0.0
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G4_probability_superiority"

    def parameters(self) -> dict[str, Any]:
        """Return probability-of-benefit parameters."""
        return {
            "p_min": self.p_min,
            "epsilon_benefit": self.epsilon_benefit,
            "epsilon_mean": self.epsilon_mean,
            "horizon": self.horizon,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None:
            raise ValueError("Missing requested horizon.")
        probability = inputs.probability_benefit5
        if probability is None:
            if residual.members is None:
                raise ValueError("Benefit probability unavailable.")
            probability = empirical_probability(residual.members, self.epsilon_benefit, greater=True)
        action = _best_distinct(probability + residual.mean * 1e-12, inputs, self.tie_tolerance)
        mean_ok = self.epsilon_mean is None or residual.mean[action] > self.epsilon_mean
        if action == inputs.base.action or probability[action] < self.p_min or not mean_ok:
            return self._decision(
                inputs, inputs.base.action, "benefit_probability_below_threshold", p_benefit=probability
            )
        return self._decision(inputs, action, "probability_of_superiority", p_benefit=probability)


@dataclass(slots=True)
class ConformalLCBGate(OverrideGate):
    """G5: task-grouped simultaneous one-sided conformal lower bound."""

    epsilon: float = 1e-3
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G5_conformal_lcb"

    def parameters(self) -> dict[str, Any]:
        """Return conformal threshold parameters."""
        return {"epsilon": self.epsilon, "tie_tolerance": self.tie_tolerance, "horizon": self.horizon}

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        lower = inputs.lower_bounds5 if self.horizon == SHORT_HORIZON else inputs.lower_bounds10
        if lower is None or not np.isfinite(lower).all():
            raise ValueError("resolved conformal lower bounds unavailable")
        action = _best_distinct(lower, inputs, self.tie_tolerance)
        if action == inputs.base.action or lower[action] <= self.epsilon:
            return self._decision(inputs, inputs.base.action, "conformal_bound_below_threshold", lower_bounds=lower)
        return self._decision(inputs, action, "conformal_positive_lower_bound", lower_bounds=lower)


@dataclass(slots=True)
class TwoStageGate(OverrideGate):
    """G6: override-worthwhile classifier followed by an action ranker."""

    p_override_min: float = 0.8
    epsilon_worthwhile: float = 1e-3
    tie_tolerance: float = 1e-12
    horizon: Horizon = 5
    gate_id = "G6_two_stage"

    def parameters(self) -> dict[str, Any]:
        """Return two-stage classifier threshold parameters."""
        return {
            "p_override_min": self.p_override_min,
            "epsilon_worthwhile": self.epsilon_worthwhile,
            "horizon": self.horizon,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        if inputs.override_probability is None or not np.isfinite(inputs.override_probability):
            raise ValueError("Override-worthwhile classifier output unavailable.")
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None:
            raise ValueError("Missing requested horizon.")
        action = _best_distinct(residual.mean, inputs, self.tie_tolerance)
        if inputs.override_probability < self.p_override_min or residual.mean[action] <= self.epsilon_worthwhile:
            return self._decision(inputs, inputs.base.action, "override_not_worthwhile")
        return self._decision(inputs, action, "two_stage_override")


@dataclass(slots=True)
class SafeRegretGate(OverrideGate):
    """G7: require benefit while bounding estimated material harm."""

    delta_harm: float = 0.1
    epsilon_harm: float = 1e-3
    epsilon_benefit: float = 1e-3
    p_min: float = 0.7
    benefit_mode: Literal["mean", "lcb", "probability"] = "lcb"
    kappa: float = 1.0
    horizon: Horizon = 5
    gate_id = "G7_safe_regret"

    def parameters(self) -> dict[str, Any]:
        """Return benefit and harm constraints."""
        return {
            "delta_harm": self.delta_harm,
            "epsilon_harm": self.epsilon_harm,
            "epsilon_benefit": self.epsilon_benefit,
            "p_min": self.p_min,
            "benefit_mode": self.benefit_mode,
            "kappa": self.kappa,
            "horizon": self.horizon,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        residual = inputs.residual5 if self.horizon == SHORT_HORIZON else inputs.residual10
        if residual is None or residual.members is None:
            raise ValueError("Safe-regret gate requires residual ensemble members.")
        p_harm = inputs.probability_harm5
        if p_harm is None:
            p_harm = empirical_probability(residual.members, -self.epsilon_harm, greater=False)
        p_benefit = inputs.probability_benefit5
        if p_benefit is None:
            p_benefit = empirical_probability(residual.members, self.epsilon_benefit, greater=True)
        if self.benefit_mode == "mean":
            score = residual.mean
            qualified = score > self.epsilon_benefit
            lower = None
        elif self.benefit_mode == "probability":
            score = p_benefit
            qualified = score >= self.p_min
            lower = None
        else:
            if residual.std is None:
                raise ValueError("Residual standard deviation unavailable.")
            lower = residual.mean - self.kappa * residual.std
            score = lower
            qualified = score > self.epsilon_benefit
        score = np.where((p_harm <= self.delta_harm) & qualified, score, -np.inf)
        action = _best_distinct(score, inputs, 1e-12)
        if action == inputs.base.action or not np.isfinite(score[action]):
            return self._decision(
                inputs,
                inputs.base.action,
                "harm_or_benefit_constraint_failed",
                lower_bounds=lower,
                p_benefit=p_benefit,
                p_harm=p_harm,
            )
        return self._decision(
            inputs, action, "safe_regret_override", lower_bounds=lower, p_benefit=p_benefit, p_harm=p_harm
        )


@dataclass(slots=True)
class MultiHorizonGate(OverrideGate):
    """G8: enforce H5/H10 compatibility before an optional commitment."""

    mode: Literal["h5_only", "h10_only", "robust_both", "long_benefit_short_nonharm", "adaptive_commitment"] = (
        "robust_both"
    )
    epsilon5: float = 1e-3
    epsilon10: float = 1e-3
    epsilon5_harm: float = 1e-3
    horizon: Horizon = 5
    gate_id = "G8_multi_horizon"
    _committed_action: int | None = None
    _remaining_blocks: int = 0

    def reset(self) -> None:
        """Clear episode-local commitment state."""
        self._committed_action = None
        self._remaining_blocks = 0

    def episode_state(self) -> dict[str, int | None]:
        """Return the deterministic commitment state."""
        return {
            "committed_action": self._committed_action,
            "remaining_commitment_blocks": self._remaining_blocks,
        }

    def restore_episode_state(self, *, committed_action: int | None, remaining_blocks: int) -> None:
        """Restore a validated commitment state."""
        if committed_action is not None and not 0 <= committed_action < ACTION_COUNT:
            raise ValueError("Invalid committed action.")
        if remaining_blocks < 0:
            raise ValueError("Remaining commitment blocks must be nonnegative.")
        self._committed_action = committed_action
        self._remaining_blocks = remaining_blocks

    def parameters(self) -> dict[str, Any]:
        """Return multi-horizon compatibility parameters."""
        return {
            "mode": self.mode,
            "epsilon5": self.epsilon5,
            "epsilon10": self.epsilon10,
            "epsilon5_harm": self.epsilon5_harm,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        if self.mode == "adaptive_commitment" and self._remaining_blocks > 0 and self._committed_action is not None:
            self._remaining_blocks -= 1
            return self._decision(inputs, self._committed_action, "active_two_block_commitment")
        r5, r10 = inputs.residual5, inputs.residual10
        if self.mode == "h5_only":
            score, valid = r5.mean, r5.mean > self.epsilon5
        elif self.mode == "h10_only":
            if r10 is None:
                raise ValueError("H10 prediction unavailable.")
            score, valid = r10.mean, r10.mean > self.epsilon10
        else:
            if r10 is None:
                raise ValueError("Multi-horizon gate requires H10 prediction.")
            lower5 = inputs.lower_bounds5 if inputs.lower_bounds5 is not None else r5.mean
            lower10 = inputs.lower_bounds10 if inputs.lower_bounds10 is not None else r10.mean
            score = lower10
            if self.mode == "robust_both":
                valid = (lower5 > self.epsilon5) & (lower10 > self.epsilon10)
            else:
                valid = (lower10 > self.epsilon10) & (lower5 >= -self.epsilon5_harm)
        qualified = np.where(valid, score, -np.inf)
        action = _best_distinct(qualified, inputs, 1e-12)
        if action == inputs.base.action or not np.isfinite(qualified[action]):
            return self._decision(inputs, inputs.base.action, "multi_horizon_condition_failed")
        if self.mode == "adaptive_commitment" and r10 is not None:
            lower5 = inputs.lower_bounds5 if inputs.lower_bounds5 is not None else r5.mean
            lower10 = inputs.lower_bounds10 if inputs.lower_bounds10 is not None else r10.mean
            if lower10[action] > self.epsilon10 and lower5[action] >= -self.epsilon5_harm:
                self._committed_action = action
                self._remaining_blocks = 1
        return self._decision(inputs, action, "multi_horizon_override")


@dataclass(slots=True)
class TrustWrapperGate(OverrideGate):
    """G9: block an inner conservative gate when deployable trust is low."""

    inner: OverrideGate
    minimum_trust_probability: float = 0.8
    trust_metadata_key: str = "trust_probability"
    minimum_action_feature_spread: float = 0.0
    maximum_duplicate_fraction: float = 1.0
    maximum_calibration_error: float = float("inf")
    require_gp_available: bool = False
    horizon: Horizon = 5
    gate_id = "G9_gp_trust_wrapper"

    def parameters(self) -> dict[str, Any]:
        """Return trust-wrapper and inner-gate identity."""
        return {
            "minimum_trust_probability": self.minimum_trust_probability,
            "trust_metadata_key": self.trust_metadata_key,
            "minimum_action_feature_spread": self.minimum_action_feature_spread,
            "maximum_duplicate_fraction": self.maximum_duplicate_fraction,
            "maximum_calibration_error": self.maximum_calibration_error,
            "require_gp_available": self.require_gp_available,
            "inner_gate": self.inner.gate_id,
        }

    def _decide(self, inputs: GateInputs) -> OverrideDecision:
        trust = inputs.metadata.get(self.trust_metadata_key)
        if trust is None or not np.isfinite(float(trust)) or float(trust) < self.minimum_trust_probability:
            return self._fallback(inputs, "trust_probability_below_threshold", "trust_blocked")
        return self.inner.decide(inputs)


def build_gate(gate_id: str, parameters: dict[str, Any] | None = None) -> OverrideGate:
    """Construct one registered gate from resolved Hydra parameters."""
    kwargs = dict(parameters or {})
    registry: dict[str, type[OverrideGate]] = {
        "G0_base_only": BaseOnlyGate,
        "G1_point_threshold": PointThresholdGate,
        "G2_hysteresis": HysteresisGate,
        "G3_ensemble_lcb": EnsembleLCBGate,
        "G4_probability_superiority": ProbabilitySuperiorityGate,
        "G5_conformal_lcb": ConformalLCBGate,
        "G6_two_stage": TwoStageGate,
        "G7_safe_regret": SafeRegretGate,
        "G8_multi_horizon": MultiHorizonGate,
    }
    if gate_id == "G9_gp_trust_wrapper":
        inner_config = kwargs.pop("inner", {"gate_id": "G3_ensemble_lcb", "parameters": {}})
        return TrustWrapperGate(inner=build_gate(inner_config["gate_id"], inner_config.get("parameters")), **kwargs)
    try:
        return registry[gate_id](**kwargs)  # type: ignore[call-arg]
    except KeyError as error:
        raise ValueError(f"Unknown selective gate {gate_id!r}.") from error


GATE_REGISTRY = {
    "G0": "base only",
    "G1": "point residual threshold",
    "G2": "hysteresis and candidate distinctness",
    "G3": "ensemble residual lower confidence bound",
    "G4": "probability of superiority",
    "G5": "task-grouped conformal lower bound",
    "G6": "override-worthwhile classifier plus ranker",
    "G7": "harm-probability constrained residual gate",
    "G8": "multi-horizon compatibility",
    "G9": "deployable model-trust wrapper",
}
