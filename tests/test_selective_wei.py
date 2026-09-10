"""Scientific contracts for selective residual WEI control."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dacboenv.experiment.collect_selective_wei_policies import collect as collect_policies
from dacboenv.experiment.fit_selective_wei import fit
from dacboenv.experiment.prepare_selective_wei_carps_eval import prepare as prepare_carps
from dacboenv.experiment.selective_branch_campaign import prepare as prepare_campaign
from dacboenv.offline.schema import MIDRUN_BRANCH_SCHEMA_VERSION
from dacboenv.selective_wei.artifacts import finalize_policy_bundle, load_policy_bundle
from dacboenv.selective_wei.base_selector import fit_base_selector, save_base_selector
from dacboenv.selective_wei.calibration import (
    GateConstraints,
    OverrideWorthwhileClassifier,
    ProbabilityCalibrator,
    ReliabilityClassifier,
    risk_coverage_gain_row,
    select_frontier_row,
)
from dacboenv.selective_wei.context import context_from_task_id, exact_context_key, selective_train_partition
from dacboenv.selective_wei.gates import (
    BaseOnlyGate,
    ConformalLCBGate,
    EnsembleLCBGate,
    HysteresisGate,
    MultiHorizonGate,
    PointThresholdGate,
    ProbabilitySuperiorityGate,
    SafeRegretGate,
    TrustWrapperGate,
    TwoStageGate,
)
from dacboenv.selective_wei.initial_base_model import fit_initial_base, save_initial_base
from dacboenv.selective_wei.initial_context import InitialAnchorCache, InitialContext
from dacboenv.selective_wei.metrics import evaluate_selective_actions, grouped_bootstrap_gain
from dacboenv.selective_wei.policy import SelectiveWEIPolicy
from dacboenv.selective_wei.predictors import EnsembleDelayedValuePredictor, fit_sklearn_predictor, save_predictor
from dacboenv.selective_wei.provenance import validate_selective_branch_provenance
from dacboenv.selective_wei.schemas import BaseDecision, GateInputs, ValuePrediction
from dacboenv.selective_wei.uncertainty import (
    calibrate_task_max_conformal,
    candidate_equivalence,
    finite_sample_conformal_quantile,
    hard_trust_metadata,
    mondrian_or_global,
    residual_advantages,
    trust_feature_vector,
)
from gymnasium.spaces import (
    Box,
    Dict as DictSpace,
    Discrete,
)
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf

SCENARIOS = ("lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost")


def _tasks(repeats: int = 3) -> list[str]:
    tasks = [f"bbob/{dimension}/{function}/1" for dimension in (2, 8) for function in range(1, repeats + 1)]
    tasks.extend(f"yahpo/so/{scenario}/{100 + index}/None" for scenario in SCENARIOS for index in range(repeats))
    return tasks


def _dev_tasks() -> list[str]:
    return ["bbob/2/10/1", "bbob/8/10/1", *(f"yahpo/so/{scenario}/999/None" for scenario in SCENARIOS)]


def _write_branch(path: Path, tasks: list[str], split: str) -> None:
    rows = [(task, seed, phase) for task in tasks for seed in (0, 1) for phase in range(4)]
    n = len(rows)
    state = np.zeros((n, 13), dtype=np.float32)
    features = np.zeros((n, 5, 4), dtype=np.float32)
    features[:, :, 0] = np.asarray([0, 0.25, 0.5, 0.75, 1])
    features[:, :, 1] = np.arange(5)
    q5 = np.zeros((n, 5), dtype=np.float64)
    q10 = np.zeros((n, 5), dtype=np.float64)
    for index, (task, seed, phase) in enumerate(rows):
        state[index, 0] = phase / 4
        state[index, 1] = seed
        preferred = 3 if task.startswith("bbob/8/") else 1 if task.startswith("yahpo/") else 2
        q5[index] = -0.01 * np.abs(np.arange(5) - preferred) + 0.001 * phase
        q10[index] = -0.02 * np.abs(np.arange(5) - preferred) + 0.002 * phase
    task_values = [row[0] for row in rows]
    data_split = np.asarray([split] * n)
    environment = np.asarray(["train" if split == "train" else "validation"] * n)
    metadata = {
        "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
        "component": "midrun_same_state_q5_q10",
        "split": split,
        "context_split": split,
        "environment_context_split": environment[0],
        "data_role": "offline_training" if split == "train" else "offline_development",
        "manifest_hash": f"manifest-{split}",
    }
    scenario_ids = []
    for task in task_values:
        scenario_ids.append(0 if task.startswith("bbob/") else 1 + SCENARIOS.index(task.split("/")[2]))
    strings = lambda values: np.asarray(values, dtype=f"U{max(map(len, values))}")
    arrays: dict[str, Any] = {
        "global_state": state,
        "action_features": features,
        "action_alpha": np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32),
        "q5": q5,
        "q10": q10,
        "valid_action_mask": np.ones((n, 5), dtype=bool),
        "tie_mask_q5": q5.max(axis=1, keepdims=True) - q5 <= 1e-3,
        "tie_mask_q10": q10.max(axis=1, keepdims=True) - q10 <= 1e-3,
        "top1_top2_gap_q5": np.sort(q5, axis=1)[:, -1] - np.sort(q5, axis=1)[:, -2],
        "top1_top2_gap_q10": np.sort(q10, axis=1)[:, -1] - np.sort(q10, axis=1)[:, -2],
        "task_id": strings(task_values),
        "domain_id": np.asarray([int(task.startswith("yahpo/")) for task in task_values], dtype=np.int8),
        "scenario_id": np.asarray(scenario_ids, dtype=np.int8),
        "phase_bin": np.asarray([row[2] for row in rows], dtype=np.int8),
        "seed": np.asarray([row[1] for row in rows], dtype=np.int32),
        "source_policy_id": strings(["uniform_random"] * n),
        "source_state_digest": strings([f"state-{index}" for index in range(n)]),
        "source_replay_digest": strings([f"replay-{index}" for index in range(n)]),
        "candidate_duplicate_groups": strings(["[0,1,2,3,4]"] * n),
        "branch_protocol_hash": strings(["protocol"] * n),
        "reference_metadata_json": strings(["{}"] * n),
        "data_context_split": data_split,
        "environment_context_split": environment,
        "dataset_metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
    }
    np.savez_compressed(path, **arrays)


def _selector() -> Any:
    tasks = np.asarray(["bbob/2/1/1", "bbob/2/2/1", "yahpo/so/lcbench/1/None"])
    q = np.asarray([[0, 1, 0, 0, 0], [0, 2, 0, 0, 0], [0, 0, 3, 0, 0]], dtype=float)
    return fit_base_selector(
        q_values=q,
        task_ids=tasks,
        phase_bins=np.zeros(3, dtype=int),
        kind="context_mean",
        horizon=5,
        source_split="train",
        fit_manifest_hash="manifest",
        fit_data_hash="data",
        code_revision="revision",
    )


def _inputs() -> GateInputs:
    base = BaseDecision(2, 0.5, "B1", "global", "global", 5, "m", "d")
    members = np.asarray([[0, 0.1, 0.2, 0.35, 0.1], [0, 0.1, 0.2, 0.30, 0.1], [0, 0.1, 0.2, 0.40, 0.1]])
    prediction = ValuePrediction(q5_mean=members.mean(axis=0), q5_member_values=members)
    residual = residual_advantages(prediction, 2, 5)
    features = np.column_stack((np.arange(5) / 4, np.arange(5), np.zeros((5, 2))))
    return GateInputs(
        base=base,
        prediction=prediction,
        residual5=residual,
        residual10=None,
        equivalence=candidate_equivalence(features),
        lower_bounds5=residual.mean - 0.02,
        probability_benefit5=np.asarray([0, 0, 0, 1, 0]),
        probability_harm5=np.zeros(5),
        override_probability=0.95,
        metadata={"trust_probability": 0.95},
    )


def test_contextual_bases_use_dimension_scenario_and_safe_fallback() -> None:
    selector = _selector()
    key = exact_context_key(context_from_task_id("bbob/2/24/99"))
    assert key == "bbob:dimension:2"
    assert "24" not in key
    assert selector.select(context_from_task_id("bbob/2/24/99")).action == 1
    assert selector.select(context_from_task_id("yahpo/so/lcbench/77/None")).action == 2
    assert selector.select(context_from_task_id("bbob/8/24/99")).fallback_level == "domain"


def test_base_variants_and_stochastic_probabilities_are_task_balanced() -> None:
    tasks = np.asarray(["bbob/2/1/1"] * 20 + ["bbob/2/2/1"])
    q = np.vstack((np.tile([0, 1, 0, 0, 0], (20, 1)), [[0, 0, 2, 0, 0]]))
    selector = fit_base_selector(
        q_values=q,
        task_ids=tasks,
        phase_bins=np.zeros(len(tasks), dtype=int),
        kind="context_stochastic",
        horizon=5,
        source_split="train",
        fit_manifest_hash="m",
        fit_data_hash="d",
        code_revision="r",
    )
    decision = selector.select(context_from_task_id("bbob/2/9/1"))
    assert decision.action_probabilities is not None
    assert np.isclose(sum(decision.action_probabilities), 1)
    assert decision.action == 2  # task-level mean: action 2 has value 1 versus action 1 value .5


def test_memberwise_residualization_preserves_covariance_and_base_zero() -> None:
    members = np.asarray([[1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [-5, -4, -3, -2, -1]], dtype=float)
    prediction = ValuePrediction(q5_mean=members.mean(axis=0), q5_member_values=members)
    residual = residual_advantages(prediction, 2, 5)
    assert np.array_equal(residual.members[:, 2], np.zeros(3))
    assert residual.mean[2] == residual.std[2] == 0
    assert np.allclose(residual.std, np.zeros(5))  # common shifts cancel exactly


def test_candidate_equivalence_excludes_alpha_and_gates_abstain() -> None:
    features = np.asarray([[0, 1, 2, 3], [0.25, 1, 2, 3], [0.5, 4, 5, 6], [0.75, 7, 8, 9], [1, 10, 11, 12]])
    equivalence = candidate_equivalence(features, heuristic=True)
    assert equivalence.equivalent(0, 1)
    inputs = _inputs()
    inputs.equivalence = equivalence
    inputs.base = BaseDecision(0, 0, "B", "global", "global", 5, "m", "d")
    prediction = ValuePrediction(q5_mean=np.asarray([0, 10, 0, 0, 0], dtype=float))
    inputs.prediction = prediction
    inputs.residual5 = residual_advantages(prediction, 0, 5)
    assert PointThresholdGate(epsilon=0).decide(inputs).selected_action == 0
    boundary = features.copy()
    boundary[1, 1:] = boundary[0, 1:] + 0.99e-6
    assert candidate_equivalence(boundary, tolerance=1e-6, heuristic=True).equivalent(0, 1)


def test_g0_g1_g2_g3_g5_g7_g8_g9_and_fail_safe() -> None:
    inputs = _inputs()
    assert not BaseOnlyGate().decide(inputs).override
    assert PointThresholdGate(epsilon=0.05).decide(inputs).selected_action == 3
    assert HysteresisGate(epsilon_base=0.05, epsilon_switch=0.05).decide(inputs).selected_action == 3
    assert EnsembleLCBGate(kappa=1, epsilon=0.05).decide(inputs).selected_action == 3
    assert ConformalLCBGate(epsilon=0.05).decide(inputs).selected_action == 3
    assert ProbabilitySuperiorityGate(p_min=0.9).decide(inputs).selected_action == 3
    assert TwoStageGate(p_override_min=0.9).decide(inputs).selected_action == 3
    assert SafeRegretGate(delta_harm=0.1, epsilon_benefit=0.05).decide(inputs).selected_action == 3
    h10_members = inputs.prediction.q5_member_values + np.asarray([0, 0, 0, 0.1, 0])
    inputs.prediction = ValuePrediction(
        q5_mean=inputs.prediction.q5_mean,
        q10_mean=h10_members.mean(axis=0),
        q5_member_values=inputs.prediction.q5_member_values,
        q10_member_values=h10_members,
    )
    inputs.residual10 = residual_advantages(inputs.prediction, 2, 10)
    inputs.lower_bounds10 = inputs.residual10.mean - 0.01
    assert MultiHorizonGate(mode="robust_both").decide(inputs).selected_action == 3
    assert TrustWrapperGate(inner=EnsembleLCBGate(), minimum_trust_probability=0.9).decide(inputs).override
    inputs.prediction = ValuePrediction(q5_mean=np.zeros(5), prediction_status="missing_model")
    fallback = EnsembleLCBGate().decide(inputs)
    assert not fallback.override
    assert fallback.selected_action == inputs.base.action
    assert fallback.fallback_status


def test_hysteresis_dwell_and_adaptive_commitment_reset() -> None:
    inputs = _inputs()
    inputs.current_action = 1
    inputs.blocks_since_switch = 0
    assert HysteresisGate(minimum_dwell_blocks=2).decide(inputs).selected_action == 1
    h10 = inputs.prediction.q5_member_values + np.asarray([0, 0, 0, 0.1, 0])
    inputs.prediction = ValuePrediction(
        q5_mean=inputs.prediction.q5_mean,
        q10_mean=h10.mean(axis=0),
        q5_member_values=inputs.prediction.q5_member_values,
        q10_member_values=h10,
    )
    inputs.residual10 = residual_advantages(inputs.prediction, 2, 10)
    gate = MultiHorizonGate(mode="adaptive_commitment")
    assert gate.decide(inputs).selected_action == 3
    assert gate._remaining_blocks == 1
    gate.reset()
    assert gate._remaining_blocks == 0
    assert gate._committed_action is None


def test_task_grouped_conformal_quantile_and_insufficient_fallback() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    assert finite_sample_conformal_quantile(values, 0.4) == 0.3  # ceil(5*.6)=3, zero-based index 2
    predicted = np.asarray([[0, 0.2, 0, 0, 0], [0, 0.3, 0, 0, 0], [0, 0.1, 0, 0, 0]])
    truth = np.zeros_like(predicted)
    artifact = calibrate_task_max_conformal(
        predicted,
        truth,
        ["a", "b", "c"],
        base_actions=[0, 0, 0],
        delta=0.2,
        horizon=5,
        task_hash="tasks",
        minimum_tasks=5,
    )
    assert artifact.fallback_mode == "insufficient_tasks"
    assert np.isinf(artifact.residual_quantile)
    global_artifact = calibrate_task_max_conformal(
        np.tile(predicted, (2, 1)),
        np.zeros((6, 5)),
        ["a", "b", "c", "d", "e", "f"],
        base_actions=[0] * 6,
        delta=0.2,
        horizon=5,
        task_hash="global",
        minimum_tasks=5,
    )
    assert mondrian_or_global({}, "yahpo", global_artifact) is global_artifact


def test_probability_two_stage_and_learned_trust_models() -> None:
    scores = np.linspace(0, 1, 20)
    labels = scores > 0.5
    calibrator = ProbabilityCalibrator("logistic").fit(scores, labels)
    assert calibrator.predict(np.asarray([0.1, 0.9]))[0] < calibrator.predict(np.asarray([0.1, 0.9]))[1]
    features = np.column_stack((scores, scores**2))
    residuals = np.column_stack((np.zeros(20), scores - 0.5, np.zeros((20, 3))))
    classifier = OverrideWorthwhileClassifier("extra_trees", 0).fit(features, residuals, epsilon_worthwhile=0.0)
    assert classifier.predict_probability(features).shape == (20,)
    reliability = ReliabilityClassifier(0).fit(
        features,
        np.where(labels, 0.001, 0.1),
        maximum_reliable_error=0.01,
    )
    assert reliability.predict_probability(features).shape == (20,)
    action_features = np.column_stack((np.arange(5) / 4, np.arange(5), np.zeros((5, 2))))
    metadata = hard_trust_metadata(
        np.zeros(13), action_features, candidate_equivalence(action_features), {"maximum_duplicate_fraction": 1}
    )
    assert trust_feature_vector(metadata).shape == (5,)


def test_task_partition_is_stratified_disjoint_deterministic_and_dev_rejected() -> None:
    tasks = _tasks()
    first = selective_train_partition(tasks)
    assert first == selective_train_partition(tasks)
    parts = [set(first["partitions"][key]) for key in ("model_fit", "gate_tune", "uncertainty_calibration")]
    assert set.union(*parts) == set(tasks)
    assert not any(a & b for i, a in enumerate(parts) for b in parts[i + 1 :])
    with pytest.raises(ValueError, match="learned-policy"):
        validate_selective_branch_provenance(
            {"context_split": "train", "campaign_role": "d1_f5_learned_headroom", "task_ids": []},
            role="model_fit",
        )


def test_risk_frontier_metrics_and_grouped_bootstrap() -> None:
    q = np.asarray([[0, 1, 0, 0, 0], [0, -1, 0, 0, 0], [0, 2, 0, 0, 0]], dtype=float)
    selected, base = np.asarray([1, 1, 1]), np.zeros(3, dtype=int)
    tasks = np.asarray(["a", "b", "c"])
    row = risk_coverage_gain_row(
        selected,
        base,
        q,
        tasks,
        np.asarray(["x", "x", "x"]),
        parameters={"epsilon": 0},
        constraints=GateConstraints(delta_harm=0.5, minimum_override_count=1, minimum_override_tasks=1),
    )
    assert row["override_coverage"] == 1
    assert row["harmful_override_fraction"] == pytest.approx(1 / 3)
    assert select_frontier_row([row])["parameters"] == {"epsilon": 0}
    metrics = evaluate_selective_actions(q, selected, base, tasks)
    assert metrics["selected_minus_base"] == pytest.approx(2 / 3)
    assert grouped_bootstrap_gain(np.asarray([1, -1, 2]), tasks, resamples=20)["resamples"] == 20


def _attach_synthetic_initial_anchors(branch: Path, output: Path) -> Path:
    """Add engineering-only immutable initial contexts to an existing fixture."""
    context = InitialContext(
        "synthetic", ("dimension",), (2.0,), (True,), ("M",), 2, 100, "design", "costs", "model", "data", "protocol", 0
    )
    fitting = selective_train_partition(_tasks())["partitions"]["model_fit"]
    model = fit_initial_base(
        [context] * len(fitting), np.tile([1.0, 1.0, 0.0, 1.0, 1.0], (len(fitting), 1)), fitting, groups=("M",)
    )
    model.metadata.update(feature_settings={}, partition_manifest_hash="synthetic", train_dataset_hash="synthetic")
    save_initial_base(model, output)
    cache = InitialAnchorCache()
    cache.freeze(context)
    cache.select(model.select(context))
    for split in ("train", "dev"):
        path = branch / f"branches_{split}.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        count = len(arrays["q5"])
        arrays["initial_anchor_json"] = np.asarray([json.dumps({"state": cache.to_dict(), "settings": {}})] * count)
        arrays["initial_base_action"] = np.full(count, 2, dtype=np.int8)
        arrays["initial_base_semantic_hash"] = np.asarray([model.semantic_hash] * count)
        np.savez_compressed(path, **arrays)
    return output / "base_bundle.json"


@pytest.mark.parametrize("initial_base", [False, True])
def test_selective_fit_synthetic_smoke_and_policy_artifacts(tmp_path: Path, initial_base: bool) -> None:
    branch = tmp_path / "branches"
    branch.mkdir()
    _write_branch(branch / "branches_train.npz", _tasks(), "train")
    _write_branch(branch / "branches_dev.npz", _dev_tasks(), "dev")
    initial_path = None
    if initial_base:
        initial_path = _attach_synthetic_initial_anchors(branch, tmp_path / "initial_base")
    output = tmp_path / "run"
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        config = compose(
            config_name="selective_fit",
            overrides=[
                "+selective_policy=selective_b1_g3",
                "selective_predictor=neural_smoke",
                "selective_predictor.updates=2",
                f"selective_data.branch_root={branch}",
                f"selective_output.root={output}",
                "selective_evaluation.bootstrap_resamples=20",
                "selective_calibration.constraints.minimum_override_count=1",
                "selective_calibration.constraints.minimum_override_tasks=1",
            ],
        )
    config.selective_data.initial_base_bundle = None if initial_path is None else str(initial_path)
    completion = fit(config)
    assert completion["status"] == "complete"
    bundle = load_policy_bundle(output / "policy_bundle.json")
    if initial_base:
        assert bundle["base_target_semantics"] == "full_static_terminal_loss"
        assert bundle["calibration_base_semantic_hash"] == bundle["base_selector_registry_hash"]
    assert bundle["dev_used_for_fitting"] is False
    assert bundle["holdout_accessed"] is False
    assert (output / "gate_tuning_frontier.json").is_file()
    assert set(bundle["base_comparator_registries"]) == {
        "B0_global_static",
        "B1_context_mean",
        "B2_context_lcb",
        "B5_context_phase_ablation",
    }
    export_root = tmp_path / "export"
    inventory = collect_policies(OmegaConf.create({"run_root": str(output), "output_root": str(export_root)}))
    actual_dev = [
        "bbob/2/1/1",
        "bbob/8/1/1",
        "yahpo/so/lcbench/167168/None",
        "yahpo/so/rbv2_glmnet/375/None",
        "yahpo/so/rbv2_ranger/16/None",
        "yahpo/so/rbv2_rpart/14/None",
        "yahpo/so/rbv2_super/1053/None",
        "yahpo/so/rbv2_xgboost/12/None",
    ]
    inventory["policies"][0]["dev_task_ids"] = actual_dev
    inventory_path = export_root / "selective_policy_inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    plan = prepare_carps(
        OmegaConf.create({"policy_inventory": str(inventory_path), "output_root": str(tmp_path / "eval")})
    )
    assert plan["scientific_result_root"] != plan["hydra_sweep_root"]
    launcher = Path(plan["launcher"]).read_text(encoding="utf-8")
    assert '"$@" &' in launcher
    assert "wait" in launcher
    assert "singularity" not in launcher.lower()


def test_policy_fail_safe_logs_and_selective_campaign_manifests(tmp_path: Path) -> None:
    selector = _selector()
    registry = tmp_path / "base.json"
    registry_hash = save_base_selector(selector, registry)
    state = np.zeros((8, 13), dtype=np.float32)
    actions = np.zeros((8, 5, 4), dtype=np.float32)
    actions[:, :, 0] = np.arange(5) / 4
    q = np.tile(np.arange(5, dtype=float), (8, 1))
    members = [
        fit_sklearn_predictor(state, actions, q, q, kind="extra_trees", seed=seed, model_id=f"tree-{seed}")
        for seed in range(3)
    ]
    predictor = EnsembleDelayedValuePredictor(members, ensemble_kind="independent_seed")
    predictor_meta = save_predictor(predictor, tmp_path / "predictor")
    predictor_manifest = Path(predictor_meta["manifest_path"])
    payload = {
        "base_selector_registry": str(registry),
        "base_selector_registry_sha256": __import__("hashlib").sha256(registry.read_bytes()).hexdigest(),
        "base_selector_registry_hash": registry_hash,
        "predictor_manifest": str(predictor_manifest),
        "predictor_manifest_sha256": __import__("hashlib").sha256(predictor_manifest.read_bytes()).hexdigest(),
        "predictor_model_hashes": [member.model_hash for member in members],
        "normalizer_hash": "",
        "gate_id": "G3_ensemble_lcb",
        "gate_parameters": {"kappa": 0.0, "epsilon": 0.0, "minimum_ensemble_size": 3, "horizon": 5},
        "train_partition_hash": "partition",
        "gate_tune_task_hash": "tune",
        "uncertainty_calibration_task_hash": "calibration",
        "feature_schema_hash": "features",
        "action_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "horizon": 5,
        "interaction_frequency": 5,
        "code_revision": "revision",
    }
    bundle_path = tmp_path / "bundle.json"
    bundle = finalize_policy_bundle(payload, bundle_path)

    class Env:
        action_space = Discrete(5)
        interaction_frequency = 5
        observation_space = DictSpace(
            {
                "global_state": Box(-np.inf, np.inf, (13,), np.float32),
                "action_features": Box(-np.inf, np.inf, (5, 4), np.float32),
            }
        )
        current_task_id = "bbob/2/1/1"
        current_seed = 0

    log = tmp_path / "decisions.jsonl"
    policy = SelectiveWEIPolicy(Env(), str(bundle_path), bundle["policy_artifact_hash"], str(log))  # type: ignore[arg-type]
    action = policy({"global_state": np.zeros(13), "action_features": actions[0]})
    assert 0 <= action < 5
    assert json.loads(log.read_text().splitlines()[0])["policy_artifact_hash"]

    final = tmp_path / "final"
    final.mkdir()
    manifest = {
        "manifest_hash": "final",
        "task_splits": {"train": ["bbob/2/1/1"], "dev": ["yahpo/so/lcbench/1/None"], "holdout": ["bbob/8/24/2"]},
    }
    (final / "final_offline_dataset_manifest.json").write_text(json.dumps(manifest))
    config = OmegaConf.create(
        {
            "final_dataset_root": str(final),
            "output_root": str(tmp_path / "campaign"),
            "campaign": "base",
            "base_selector_registry": str(registry),
            "base_selector_registry_hash": registry_hash,
            "policy_bundle": None,
        }
    )
    campaign = prepare_campaign(config)
    assert campaign["job_count"] == 30
    assert all("holdout" not in row["data_context_split"] for row in campaign["jobs"])


def test_hydra_aliases_and_otus_scripts_have_no_concurrency_cap() -> None:
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        for index in range(10):
            config = compose(
                config_name="selective_fit",
                overrides=[
                    f"+selective_policy=selective_b1_g{index}",
                    "selective_data.branch_root=/tmp/branch",
                    "selective_output.root=/tmp/run",
                ],
            )
            assert config.selective_gate.id.startswith(f"G{index}_")
    scripts = list(Path("scripts/otus").glob("*selective*.sh"))
    assert scripts
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        assert "%" not in "\n".join(line for line in text.splitlines() if "--array=" in line)
        subprocess.run(["/usr/bin/bash", "-n", str(script)], check=True)  # noqa: S603
