"""Evaluate a frozen selective policy once on an explicit branch dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np

from dacboenv.offline.branch_dataset import BranchDataset
from dacboenv.selective_wei.artifacts import atomic_json, load_policy_bundle
from dacboenv.selective_wei.base_selector import load_base_selector
from dacboenv.selective_wei.calibration import load_calibration_model
from dacboenv.selective_wei.context import context_from_task_id
from dacboenv.selective_wei.gates import build_gate
from dacboenv.selective_wei.metrics import (
    evaluate_selective_actions,
    grouped_bootstrap_gain,
    prediction_metrics,
    stratified_value_metrics,
)
from dacboenv.selective_wei.predictors import load_predictor
from dacboenv.selective_wei.provenance import validate_selective_branch_provenance
from dacboenv.selective_wei.schemas import CalibrationArtifact, GateInputs
from dacboenv.selective_wei.uncertainty import (
    candidate_equivalence,
    conformal_lower_bounds,
    empirical_probability,
    hard_trust_metadata,
    residual_advantages,
    trust_feature_vector,
)

SHORT_HORIZON = 5
SMALL_GAP_THRESHOLD = 1e-3
LARGE_GAP_THRESHOLD = 1e-2

if TYPE_CHECKING:
    from omegaconf import DictConfig


def evaluate(config: DictConfig) -> dict[str, object]:  # noqa: PLR0915
    """Score local selected/base/oracle values without refitting anything."""
    bundle = load_policy_bundle(Path(str(config.policy_bundle)).resolve())
    dataset = BranchDataset(Path(str(config.branch_dataset)).resolve())
    validate_selective_branch_provenance(dataset.metadata, role="dev")
    selector = load_base_selector(
        Path(bundle["base_selector_registry"]), expected_hash=bundle.get("base_selector_registry_hash")
    )
    predictor = load_predictor(Path(bundle["predictor_manifest"]))
    gate = build_gate(bundle["gate_id"], bundle["gate_parameters"])
    override_classifier = load_calibration_model(bundle.get("override_classifier_artifact"))
    probability_calibrator = load_calibration_model(bundle.get("probability_calibration_artifact"))
    reliability_classifier = load_calibration_model(bundle.get("reliability_calibration_artifact"))
    calibration = None
    if bundle.get("uncertainty_calibration_artifact"):
        calibration = CalibrationArtifact.from_dict(
            json.loads(Path(bundle["uncertainty_calibration_artifact"]).read_text(encoding="utf-8"))
        )
    selected, bases, equivalent, predicted, duplicate_counts = [], [], [], [], []
    for index in range(len(dataset)):
        task = str(dataset.arrays["task_id"][index])
        base = selector.select(context_from_task_id(task))
        observation = {
            "global_state": dataset.arrays["global_state"][index],
            "action_features": dataset.arrays["action_features"][index],
        }
        prediction = predictor.predict(observation)
        r5 = residual_advantages(prediction, base.action, 5)
        r10 = residual_advantages(prediction, base.action, 10) if prediction.q10_mean is not None else None
        lower5 = None
        if calibration is not None and calibration.horizon == SHORT_HORIZON:
            lower5 = conformal_lower_bounds(r5.mean, calibration)
            lower5[base.action] = 0.0
        p_benefit = None if r5.members is None else empirical_probability(r5.members, 1e-3, greater=True)
        if p_benefit is not None and probability_calibrator is not None:
            p_benefit = probability_calibrator.predict(p_benefit)
            p_benefit[base.action] = 0.0
        p_harm = None if r5.members is None else empirical_probability(r5.members, -1e-3, greater=False)
        equivalence = candidate_equivalence(observation["action_features"])
        trust_metadata = hard_trust_metadata(
            observation["global_state"], observation["action_features"], equivalence, bundle["gate_parameters"]
        )
        if reliability_classifier is not None:
            trust_metadata["trust_probability"] = float(
                reliability_classifier.predict_probability(trust_feature_vector(trust_metadata)[None, :])[0]
            )
            trust_metadata["trust_reason"] = "learned_reliability_probability"
        decision = gate.decide(
            GateInputs(
                base=base,
                prediction=prediction,
                residual5=r5,
                residual10=r10,
                equivalence=equivalence,
                lower_bounds5=lower5,
                probability_benefit5=p_benefit,
                probability_harm5=p_harm,
                override_probability=(
                    1.0
                    if override_classifier is None
                    else float(
                        override_classifier.predict_probability(
                            np.concatenate(
                                (
                                    np.asarray(observation["global_state"]).reshape(-1),
                                    np.asarray(observation["action_features"]).reshape(-1),
                                )
                            )[None, :]
                        )[0]
                    )
                ),
                metadata=trust_metadata,
            )
        )
        selected.append(decision.selected_action)
        bases.append(base.action)
        predicted_values = prediction.q5_mean if int(bundle["horizon"]) == SHORT_HORIZON else prediction.q10_mean
        if predicted_values is None:
            raise ValueError("Frozen predictor lacks the policy deployment horizon.")
        predicted.append(predicted_values)
        duplicate_counts.append(5 - len(set(equivalence.groups)))
        equivalent.append(decision.override and equivalence.equivalent(decision.selected_action, base.action))
    selected_array, base_array = np.asarray(selected), np.asarray(bases)
    horizon = int(bundle["horizon"])
    q = dataset.arrays["q5" if horizon == SHORT_HORIZON else "q10"]
    result = evaluate_selective_actions(
        q,
        selected_array,
        base_array,
        dataset.arrays["task_id"].astype(str),
        equivalent_override=np.asarray(equivalent),
    )
    rows = np.arange(len(dataset))
    gain = q[rows, selected_array] - q[rows, base_array]
    result["paired_task_bootstrap"] = grouped_bootstrap_gain(
        gain,
        dataset.arrays["task_id"].astype(str),
        resamples=int(config.bootstrap_resamples),
        seed=int(config.seed),
    )
    result["prediction_metrics"] = prediction_metrics(np.stack(predicted), q, selected_array)
    gap = np.sort(q, axis=1)[:, -1] - np.sort(q, axis=1)[:, -2]
    result["breakdowns"] = stratified_value_metrics(
        q,
        selected_array,
        base_array,
        dataset.arrays["task_id"].astype(str),
        {
            "domain": dataset.arrays["domain_id"],
            "scenario": dataset.arrays["scenario_id"],
            "budget_phase": dataset.arrays["phase_bin"],
            "selected_action": selected_array,
            "base_action": base_array,
            "gap_bin": np.where(
                gap > LARGE_GAP_THRESHOLD,
                "gt_1e-2",
                np.where(gap > SMALL_GAP_THRESHOLD, "gt_1e-3", "small_or_tie"),
            ),
            "candidate_duplicate_count": np.asarray(duplicate_counts),
        },
    )
    result["policy_artifact_hash"] = bundle["policy_artifact_hash"]
    result["holdout_accessed"] = False
    output = Path(str(config.output)).resolve()
    atomic_json(output, result)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="selective_branch_eval")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Evaluate one explicit dev branch dataset."""
    print(json.dumps(evaluate(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
