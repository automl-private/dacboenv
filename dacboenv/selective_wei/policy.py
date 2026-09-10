"""CARP-S deployment policy for selective residual WEI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from gymnasium.spaces import (
    Dict as DictSpace,
    Discrete,
)

from dacboenv.env.observation import GLOBAL_STATE_INDEX
from dacboenv.policy.abstract_policy import AbstractPolicy
from dacboenv.policy.initial_base import InitialConditionedWEIPolicy
from dacboenv.selective_wei.artifacts import load_policy_bundle
from dacboenv.selective_wei.base_selector import load_base_selector
from dacboenv.selective_wei.calibration import load_calibration_model
from dacboenv.selective_wei.context import context_from_task_id
from dacboenv.selective_wei.gates import MultiHorizonGate, build_gate
from dacboenv.selective_wei.predictors import load_predictor
from dacboenv.selective_wei.schemas import ACTION_COUNT, BaseDecision, CalibrationArtifact, GateInputs
from dacboenv.selective_wei.stochastic_base import SeededBaseMixture
from dacboenv.selective_wei.uncertainty import (
    candidate_equivalence,
    conformal_lower_bounds,
    empirical_probability,
    hard_trust_metadata,
    residual_advantages,
    trust_feature_vector,
)

INTERACTION_FREQUENCY = 5
SHORT_HORIZON = 5
AVAILABILITY_THRESHOLD = 0.5
TIE_TOLERANCE = 1e-12
DECISION_KEY_SIZE = 2

if TYPE_CHECKING:
    from dacboenv.dacboenv import DACBOEnv
    from dacboenv.env.observations.types import ObsType


class SelectiveWEIPolicy(AbstractPolicy):
    """Choose a contextual base unless a frozen gate supports an override."""

    def __init__(
        self,
        env: DACBOEnv,
        policy_bundle: str,
        policy_bundle_hash: str,
        decision_log: str | None = None,
    ) -> None:
        super().__init__(
            env,
            policy_bundle=policy_bundle,
            policy_bundle_hash=policy_bundle_hash,
            decision_log=decision_log,
        )
        if not isinstance(env.action_space, Discrete) or int(env.action_space.n) != ACTION_COUNT:
            raise TypeError("SelectiveResidualWEI requires Discrete(5).")
        if int(env.interaction_frequency) != INTERACTION_FREQUENCY:
            raise ValueError("SelectiveResidualWEI requires interaction frequency f=5.")
        if not isinstance(env.observation_space, DictSpace) or not {
            "global_state",
            "action_features",
        } <= set(env.observation_space.spaces):
            raise ValueError("SelectiveResidualWEI requires the compact structured observation.")
        bundle = load_policy_bundle(Path(policy_bundle).resolve(), expected_hash=policy_bundle_hash)
        self.bundle = bundle
        self.initial_base: InitialConditionedWEIPolicy | None = None
        self.base_selector = None
        if bundle.get("base_target_semantics") == "full_static_terminal_loss":
            self.initial_base = InitialConditionedWEIPolicy(env, bundle["base_selector_registry"])
            if bundle.get("calibration_base_semantic_hash") != self.initial_base.base.semantic_hash:
                raise ValueError("Selective gate was not calibrated against this exact frozen initial base.")
        else:
            self.base_selector = load_base_selector(
                Path(bundle["base_selector_registry"]), expected_hash=bundle.get("base_selector_registry_hash")
            )
        if self.base_selector is not None and self.base_selector.registry.kind == "context_stochastic":
            raise ValueError(
                "Selective gates require a frozen realized stochastic anchor; "
                "legacy B4 mixtures are base-only controls."
            )
        self.predictor = load_predictor(Path(bundle["predictor_manifest"]))
        self.gate = build_gate(str(bundle["gate_id"]), dict(bundle["gate_parameters"]))
        self.calibration = self._load_calibration(bundle)
        self.override_classifier = load_calibration_model(bundle.get("override_classifier_artifact"))
        self.probability_calibrator = load_calibration_model(bundle.get("probability_calibration_artifact"))
        self.reliability_classifier = load_calibration_model(bundle.get("reliability_calibration_artifact"))
        self.decision_log = None if decision_log is None else Path(decision_log).resolve()
        self.current_action: int | None = None
        self.blocks_since_switch = 10**9
        self._decision_key: tuple[int, int] | None = None
        self._cached_action: int | None = None

    def _select_base(self, task_id: str) -> BaseDecision:
        """Return either the legacy contextual base or the immutable episode a0."""
        if self.initial_base is None:
            assert self.base_selector is not None
            return self.base_selector.select(context_from_task_id(task_id))
        action = self.initial_base(None)
        metadata = self.initial_base.base.metadata
        return BaseDecision(
            action,
            action / 4.0,
            "initial_terminal_base",
            "initial_context",
            "exact",
            self.bundle["horizon"],
            metadata["partition_manifest_hash"],
            metadata["train_dataset_hash"],
        )

    @staticmethod
    def _load_calibration(bundle: dict[str, Any]) -> CalibrationArtifact | None:
        path = bundle.get("uncertainty_calibration_artifact")
        if not path:
            return None
        return CalibrationArtifact.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def _log(self, record: dict[str, Any]) -> None:
        if self.decision_log is None:
            return
        if self.initial_base is not None:
            anchor = self._env.initial_anchor_state()
            record["initial_anchor_digest"] = anchor["digest"]
            record["initial_base_semantic_hash"] = self.initial_base.base.semantic_hash
        self.decision_log.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, sort_keys=True, allow_nan=False) + "\n"
        descriptor = os.open(self.decision_log, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o640)
        try:
            os.write(descriptor, line.encode("utf-8"))
        finally:
            os.close(descriptor)

    def __call__(self, obs: ObsType) -> int:
        """Make one deterministic f5 decision, falling back on any failure."""
        boundary = getattr(self._env, "policy_decision_key", None)
        key = boundary() if callable(boundary) else None
        if key is not None and key == self._decision_key and self._cached_action is not None:
            return self._cached_action
        if key is not None and self._decision_key is not None and key[0] != self._decision_key[0]:
            self.reset_policy_state()
        selected = self._decide(obs)
        self._decision_key = key
        self._cached_action = selected
        return selected

    def _decide(self, obs: ObsType) -> int:
        """Advance a gate exactly once per completed f5 boundary."""
        task_id = str(getattr(self._env, "current_task_id", ""))
        fallback_status: str | None = None
        try:
            base = self._select_base(task_id)
            observation = {
                "global_state": np.asarray(obs["global_state"], dtype=np.float32),
                "action_features": np.asarray(obs["action_features"], dtype=np.float32),
            }
            prediction = self.predictor.predict(observation)
            residual5 = residual_advantages(prediction, base.action, 5)
            residual10 = residual_advantages(prediction, base.action, 10) if prediction.q10_mean is not None else None
            equivalence = candidate_equivalence(
                observation["action_features"],
                tolerance=float(self.bundle.get("candidate_equivalence_tolerance", 1e-6)),
                heuristic=bool(self.bundle.get("candidate_feature_heuristic", False)),
            )
            lower5 = lower10 = None
            if self.calibration is not None:
                if self.calibration.horizon == SHORT_HORIZON:
                    lower5 = conformal_lower_bounds(residual5.mean, self.calibration)
                    lower5[base.action] = 0.0
                elif residual10 is not None:
                    lower10 = conformal_lower_bounds(residual10.mean, self.calibration)
                    lower10[base.action] = 0.0
            p_benefit = (
                None
                if residual5.members is None
                else empirical_probability(
                    residual5.members, float(self.bundle.get("epsilon_benefit", 1e-3)), greater=True
                )
            )
            if p_benefit is not None and self.probability_calibrator is not None:
                p_benefit = self.probability_calibrator.predict(p_benefit)
                p_benefit[base.action] = 0.0
            p_harm = (
                None
                if residual5.members is None
                else empirical_probability(
                    residual5.members, -float(self.bundle.get("epsilon_harm", 1e-3)), greater=False
                )
            )
            override_probability = (
                None
                if self.override_classifier is None
                else float(
                    self.override_classifier.predict_probability(
                        np.concatenate((observation["global_state"], observation["action_features"].reshape(-1)))[
                            None, :
                        ]
                    )[0]
                )
            )
            inputs = GateInputs(
                base=base,
                prediction=prediction,
                residual5=residual5,
                residual10=residual10,
                equivalence=equivalence,
                current_action=self.current_action,
                blocks_since_switch=self.blocks_since_switch,
                lower_bounds5=lower5,
                lower_bounds10=lower10,
                probability_benefit5=p_benefit,
                probability_harm5=p_harm,
                override_probability=override_probability,
                model_trusted=True,
                metadata=self._trust_metadata(obs, equivalence),
            )
            decision = self.gate.decide(inputs)
            selected = decision.selected_action
            if self.current_action != selected:
                self.blocks_since_switch = 0
            else:
                self.blocks_since_switch += 1
            self.current_action = selected
            fallback_status = decision.fallback_status
            record = {
                "schema_version": "dacbo-selective-decision-v1",
                "task_id": task_id,
                "seed": int(getattr(self._env, "current_seed", -1)),
                "bo_budget_fraction": float(observation["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]]),
                "base_action": decision.base_action,
                "base_alpha": float(decision.base_action) / 4.0,
                "current_action": inputs.current_action,
                "selected_action": selected,
                "selected_alpha": float(selected) / 4.0,
                "override": decision.override,
                "gate_reason": decision.reason,
                "predicted_q5": prediction.q5_mean.tolist(),
                "predicted_q10": None if prediction.q10_mean is None else prediction.q10_mean.tolist(),
                "predicted_residual5": residual5.mean.tolist(),
                "predicted_residual10": None if residual10 is None else residual10.mean.tolist(),
                "residual_uncertainty5": None if residual5.std is None else residual5.std.tolist(),
                "residual_uncertainty10": (
                    None if residual10 is None or residual10.std is None else residual10.std.tolist()
                ),
                "lower_bounds": decision.lower_bounds,
                "probability_of_benefit": decision.probability_of_benefit,
                "probability_of_harm": decision.probability_of_harm,
                "candidate_equivalence": equivalence.groups,
                "selected_equivalence_class": decision.equivalence_class,
                "trust_status": decision.model_trusted,
                "fallback_status": fallback_status,
                "gate_id": decision.gate_id,
                "gate_parameters": decision.gate_parameters,
                "policy_artifact_hash": self.bundle["policy_artifact_hash"],
                "predictor_model_hashes": self.bundle["predictor_model_hashes"],
                "base_selector_registry_sha256": self.bundle["base_selector_registry_sha256"],
            }
            self._log(record)
            return int(selected)
        except Exception as error:  # noqa: BLE001 - policy must fail safely
            try:
                base = self._select_base(task_id)
            except Exception:
                if self.initial_base is not None:
                    raise  # A missing/corrupt initial anchor is not a successful learned evaluation.
                base_action = int(self.bundle.get("emergency_global_action", 2))
            else:
                base_action = base.action
            self._log(
                {
                    "schema_version": "dacbo-selective-decision-v1",
                    "task_id": task_id,
                    "seed": int(getattr(self._env, "current_seed", -1)),
                    "bo_budget_fraction": float(
                        np.asarray(obs["global_state"])[GLOBAL_STATE_INDEX["budget_percentage"]]
                    ),
                    "base_action": base_action,
                    "base_alpha": float(base_action) / 4.0,
                    "current_action": self.current_action,
                    "selected_action": base_action,
                    "selected_alpha": float(base_action) / 4.0,
                    "override": False,
                    "gate_reason": f"safe_fallback:{type(error).__name__}:{error}",
                    "predicted_q5": None,
                    "predicted_q10": None,
                    "predicted_residual5": None,
                    "predicted_residual10": None,
                    "residual_uncertainty5": None,
                    "residual_uncertainty10": None,
                    "lower_bounds": None,
                    "probability_of_benefit": None,
                    "probability_of_harm": None,
                    "candidate_equivalence": None,
                    "selected_equivalence_class": None,
                    "trust_status": False,
                    "fallback_status": "policy_exception",
                    "gate_id": self.bundle["gate_id"],
                    "gate_parameters": self.bundle["gate_parameters"],
                    "policy_artifact_hash": self.bundle["policy_artifact_hash"],
                    "predictor_model_hashes": self.bundle["predictor_model_hashes"],
                    "base_selector_registry_sha256": self.bundle["base_selector_registry_sha256"],
                }
            )
            self.current_action = base_action
            return base_action

    def _trust_metadata(self, obs: ObsType, equivalence: Any) -> dict[str, Any]:
        """Expose only optional deployable reliability summaries."""
        gp_available = None
        if "gp_hp_summary" in obs:
            summary = np.asarray(obs["gp_hp_summary"], dtype=np.float64)
            gp_available = bool(summary[0] > AVAILABILITY_THRESHOLD) if summary.size else False
        metadata = hard_trust_metadata(
            np.asarray(obs["global_state"]),
            np.asarray(obs["action_features"]),
            equivalence,
            self.bundle.get("gate_parameters", {}),
            gp_available=gp_available,
        )
        if self.reliability_classifier is not None:
            metadata["trust_probability"] = float(
                self.reliability_classifier.predict_probability(trust_feature_vector(metadata)[None, :])[0]
            )
            metadata["trust_reason"] = "learned_reliability_probability"
        if "gp_hp_change" in obs:
            change = np.asarray(obs["gp_hp_change"], dtype=np.float64)
            metadata["gp_change_available"] = bool(change[0] > AVAILABILITY_THRESHOLD) if change.size else False
        return metadata

    def set_seed(self, seed: int | None) -> None:
        """Ignore seeds because primary selective deployment is deterministic."""

    def reset_policy_state(self) -> None:
        """Reset hysteresis and optional commitment state between episodes."""
        self.current_action = None
        self.blocks_since_switch = 10**9
        self._decision_key = None
        self._cached_action = None
        if isinstance(self.gate, MultiHorizonGate):
            self.gate.reset()

    def get_policy_state(self) -> dict[str, Any]:
        """Return JSON-safe episode state for deterministic pause/resume."""
        state: dict[str, Any] = {
            "current_action": self.current_action,
            "blocks_since_switch": self.blocks_since_switch,
            "committed_action": None,
            "remaining_commitment_blocks": 0,
            "decision_key": self._decision_key,
            "cached_action": self._cached_action,
        }
        if isinstance(self.gate, MultiHorizonGate):
            state.update(self.gate.episode_state())
        return state

    def set_policy_state(self, state: dict[str, Any]) -> None:
        """Restore validated episode state without changing scientific artifacts."""
        current = state.get("current_action")
        committed = state.get("committed_action")
        if current is not None and not 0 <= current < ACTION_COUNT:
            raise ValueError("Invalid selective current action in restored state.")
        if committed is not None and not 0 <= committed < ACTION_COUNT:
            raise ValueError("Invalid selective committed action in restored state.")
        blocks = int(state.get("blocks_since_switch") or 0)
        remaining = int(state.get("remaining_commitment_blocks") or 0)
        if blocks < 0 or remaining < 0:
            raise ValueError("Selective policy state counters must be nonnegative.")
        self.current_action = current
        self.blocks_since_switch = blocks
        key = state.get("decision_key")
        cached = state.get("cached_action")
        if key is not None and (len(key) != DECISION_KEY_SIZE or any(int(value) < 0 for value in key)):
            raise ValueError("Invalid restored policy boundary key.")
        if cached is not None and cached not in range(ACTION_COUNT):
            raise ValueError("Invalid restored cached action.")
        self._decision_key = None if key is None else (int(key[0]), int(key[1]))
        self._cached_action = cached
        if isinstance(self.gate, MultiHorizonGate):
            self.gate.restore_episode_state(committed_action=committed, remaining_blocks=remaining)


class SelectiveBaseOnlyPolicy(AbstractPolicy):
    """Deploy any frozen B0--B5 selector without a value predictor."""

    def __init__(
        self,
        env: DACBOEnv,
        registry: str,
        registry_hash: str,
        stochastic_mode: Literal["episode_static_mixture", "block_randomized_base"] = "episode_static_mixture",
    ) -> None:
        super().__init__(env, registry=registry, registry_hash=registry_hash, stochastic_mode=stochastic_mode)
        self.selector = load_base_selector(Path(registry).resolve(), expected_hash=registry_hash)
        self.mixture = (
            SeededBaseMixture(stochastic_mode) if self.selector.registry.kind == "context_stochastic" else None
        )

    def __call__(self, obs: ObsType) -> int:
        """Route by deployment context and optional explicit phase ablation."""
        phase = None
        if self.selector.registry.kind == "context_phase":
            budget_fraction = float(np.asarray(obs["global_state"])[GLOBAL_STATE_INDEX["budget_percentage"]])
            phase = min(max(int(budget_fraction * 4), 0), 3)
        context = context_from_task_id(str(getattr(self._env, "current_task_id", "")), phase_bin=phase)
        base = self.selector.select(context)
        if self.mixture is not None:
            episode = f"{self._env.current_task_id}:{getattr(self._env, 'current_seed', -1)}"
            base = self.mixture.select(base, episode_id=episode, block_index=int(self._env.get_n_finished_trials()))
        return base.action

    def set_seed(self, seed: int | None) -> None:
        """Seed only the optional policy mixture; deterministic routing is unchanged."""
        if self.mixture is not None:
            self.mixture.set_seed(0 if seed is None else seed)

    def reset_policy_state(self) -> None:
        """Clear the once-drawn base between episodes."""
        if self.mixture is not None:
            self.mixture.reset()


class UnconstrainedDelayedValuePolicy(AbstractPolicy):
    """Diagnostic argmax branch predictor without selective abstention."""

    def __init__(self, env: DACBOEnv, predictor_manifest: str, horizon: int = 5) -> None:
        super().__init__(env, predictor_manifest=predictor_manifest, horizon=horizon)
        if horizon not in {5, 10}:
            raise ValueError("Unconstrained branch predictor horizon must be 5 or 10.")
        self.predictor = load_predictor(Path(predictor_manifest).resolve())
        self.horizon = horizon

    def __call__(self, obs: ObsType) -> int:
        """Return deterministic argmax fixed-action branch value."""
        prediction = self.predictor.predict(
            {
                "global_state": np.asarray(obs["global_state"], dtype=np.float32),
                "action_features": np.asarray(obs["action_features"], dtype=np.float32),
            }
        )
        values = prediction.q5_mean if self.horizon == SHORT_HORIZON else prediction.q10_mean
        if values is None or not np.isfinite(values).all():
            raise ValueError("Unconstrained predictor values are unavailable or nonfinite.")
        equivalence = candidate_equivalence(np.asarray(obs["action_features"], dtype=np.float32))
        maximum = float(np.max(values))
        actions = [index for index, value in enumerate(values) if maximum - float(value) <= TIE_TOLERANCE]
        # Stable first action within an equivalent tied group.
        return min(actions, key=lambda action: (equivalence.groups[action], action))

    def set_seed(self, seed: int | None) -> None:
        """Ignore seeds because argmax inference is deterministic."""


__all__ = ["SelectiveBaseOnlyPolicy", "SelectiveWEIPolicy", "UnconstrainedDelayedValuePolicy"]
