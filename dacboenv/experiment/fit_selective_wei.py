"""Fit contextual bases, delayed branch predictors, and selective gates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.offline.branch_dataset import BranchDataset
from dacboenv.offline.normalization import ObservationNormalizer
from dacboenv.selective_wei.anchored_base import FrozenInitialRowBase
from dacboenv.selective_wei.artifacts import atomic_json, finalize_policy_bundle, load_policy_bundle
from dacboenv.selective_wei.base_selector import FittedBaseSelector, fit_base_selector, save_base_selector
from dacboenv.selective_wei.calibration import (
    GateConstraints,
    OverrideWorthwhileClassifier,
    ProbabilityCalibrator,
    ReliabilityClassifier,
    risk_coverage_gain_row,
    save_calibration_model,
    select_frontier_row,
)
from dacboenv.selective_wei.context import (
    canonical_hash,
    context_from_task_id,
    exact_context_key,
    selective_train_partition,
)
from dacboenv.selective_wei.gates import OverrideGate, build_gate
from dacboenv.selective_wei.metrics import evaluate_selective_actions, grouped_bootstrap_gain
from dacboenv.selective_wei.predictors import (
    DelayedValuePredictor,
    EnsembleDelayedValuePredictor,
    fit_neural_branch_predictor,
    fit_sklearn_predictor,
    load_predictor,
    save_predictor,
)
from dacboenv.selective_wei.provenance import validate_selective_branch_provenance, validate_task_partitions
from dacboenv.selective_wei.schemas import BaseDecision, CalibrationArtifact, GateInputs, Horizon
from dacboenv.selective_wei.uncertainty import (
    calibrate_task_max_conformal,
    candidate_equivalence,
    conformal_lower_bounds,
    empirical_probability,
    hard_trust_metadata,
    residual_advantages,
    trust_feature_vector,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

SHORT_HORIZON = 5
BaseSource = FittedBaseSelector | FrozenInitialRowBase


def _normalizer(path: str | None) -> ObservationNormalizer | None:
    if not path:
        return None
    return ObservationNormalizer.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _indices_for_tasks(dataset: BranchDataset, tasks: list[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(dataset.arrays["task_id"].astype(str), np.asarray(tasks))).astype(np.int64)


def _base_kind(identifier: str) -> str:
    return {
        "B0_global_static": "global_static",
        "B1_context_mean": "context_mean",
        "B2_context_lcb": "context_lcb",
        "B3_context_cvar": "context_cvar",
        "B4_context_stochastic": "context_stochastic",
        "B5_context_phase_ablation": "context_phase",
    }[identifier]


def _fit_base(
    dataset: BranchDataset,
    indices: np.ndarray,
    config: DictConfig,
    *,
    horizon: Horizon,
    source_revision: str,
) -> BaseSource:
    initial_path = config.selective_data.get("initial_base_bundle")
    if initial_path:
        frozen = FrozenInitialRowBase.load(Path(str(initial_path)).resolve(), horizon)
        fitting_tasks = set(dataset.arrays["task_id"][indices].astype(str))
        if not set(frozen.model.metadata["task_ids"]) <= fitting_tasks:
            raise ValueError("Initial-base fitting tasks must stay within model_fit, excluding tune/calibration/dev.")
        return frozen
    target = dataset.arrays["q5" if horizon == SHORT_HORIZON else "q10"][indices]
    return fit_base_selector(
        q_values=target,
        task_ids=dataset.arrays["task_id"][indices].astype(str),
        phase_bins=dataset.arrays["phase_bin"][indices],
        kind=_base_kind(str(config.selective_base.id)),  # type: ignore[arg-type]
        horizon=horizon,
        source_split="train",
        fit_manifest_hash=str(dataset.metadata["manifest_hash"]),
        fit_data_hash=file_sha256(dataset.path),
        code_revision=source_revision,
        kappa=float(config.selective_base.get("kappa", 1.0)),
        cvar_level=float(config.selective_base.get("cvar_level", 0.2)),
        stochastic_temperature=float(config.selective_base.get("temperature", 0.05)),
    )


def _member_indices(
    dataset: BranchDataset,
    indices: np.ndarray,
    *,
    ensemble_kind: str,
    seed: int,
) -> NDArray[np.int64]:
    if ensemble_kind != "task_bootstrap":
        return np.asarray(indices, dtype=np.int64)
    tasks = np.asarray(sorted(set(dataset.arrays["task_id"][indices].astype(str).tolist())))
    generator = np.random.default_rng(seed)
    sampled = generator.choice(tasks, size=len(tasks), replace=True)
    result = np.concatenate([indices[dataset.arrays["task_id"][indices].astype(str) == task] for task in sampled])
    return np.asarray(result, dtype=np.int64)


def _fit_predictor(
    dataset: BranchDataset,
    indices: np.ndarray,
    config: DictConfig,
    normalizer: ObservationNormalizer | None,
) -> DelayedValuePredictor:
    kind = str(config.selective_predictor.kind)
    if kind == "existing":
        return load_predictor(Path(str(config.selective_predictor.manifest)).resolve())
    count = int(config.selective_predictor.get("ensemble_size", 1))
    ensemble_kind = str(config.selective_predictor.get("ensemble_kind", "independent_seed"))
    members: list[DelayedValuePredictor] = []
    for member in range(count):
        seed = int(config.seed) + 1009 * member
        selected = _member_indices(dataset, indices, ensemble_kind=ensemble_kind, seed=seed)
        state = dataset.arrays["global_state"][selected]
        features = dataset.arrays["action_features"][selected]
        q5 = dataset.arrays["q5"][selected]
        q10 = dataset.arrays.get("q10")
        q10_selected = None if q10 is None else q10[selected]
        model_id = f"{kind}-seed{seed}"
        if kind in {"extra_trees", "hist_gradient_boosting"}:
            fitted: DelayedValuePredictor = fit_sklearn_predictor(
                state,
                features,
                q5,
                q10_selected,
                kind=kind,  # type: ignore[arg-type]
                seed=seed,
                model_id=model_id,
            )
        elif kind == "shared_neural":
            fitted = fit_neural_branch_predictor(
                state,
                features,
                q5,
                q10_selected,
                seed=seed,
                updates=int(config.selective_predictor.updates),
                learning_rate=float(config.selective_predictor.learning_rate),
                model_id=model_id,
                normalizer=normalizer,
            )
        else:
            raise ValueError(f"Unsupported selective predictor kind {kind!r}.")
        members.append(fitted)
    return members[0] if len(members) == 1 else EnsembleDelayedValuePredictor(members, ensemble_kind=ensemble_kind)


def _row_base(selector: BaseSource, dataset: BranchDataset, index: int) -> BaseDecision:
    if isinstance(selector, FrozenInitialRowBase):
        return selector.row_decision(dataset.arrays, index)
    return selector.select(context_from_task_id(str(dataset.arrays["task_id"][index])))


def _base_actions(selector: BaseSource, dataset: BranchDataset, indices: np.ndarray) -> NDArray[np.int64]:
    result: NDArray[np.int64] = np.asarray(
        [_row_base(selector, dataset, int(index)).action for index in indices],
        dtype=np.int64,
    )
    return result


def _predictions(
    predictor: DelayedValuePredictor,
    dataset: BranchDataset,
    indices: np.ndarray,
) -> list[Any]:
    return [
        predictor.predict(
            {
                "global_state": dataset.arrays["global_state"][index],
                "action_features": dataset.arrays["action_features"][index],
            }
        )
        for index in indices
    ]


def _gate_parameters(config: DictConfig) -> list[dict[str, Any]]:  # noqa: PLR0911
    gate_id = str(config.selective_gate.id)
    parameters = OmegaConf.to_container(config.selective_gate.get("parameters", {}), resolve=True)
    base = dict(parameters) if isinstance(parameters, dict) else {}
    epsilon_grid = [float(value) for value in config.selective_calibration.epsilon_grid]
    kappa_grid = [float(value) for value in config.selective_calibration.kappa_grid]
    p_grid = [float(value) for value in config.selective_calibration.p_min_grid]
    if gate_id in {"G1_point_threshold", "G5_conformal_lcb"}:
        return [{**base, "epsilon": value} for value in epsilon_grid]
    if gate_id == "G2_hysteresis":
        return [{**base, "epsilon_base": value, "epsilon_switch": value} for value in epsilon_grid]
    if gate_id == "G3_ensemble_lcb":
        return [{**base, "epsilon": epsilon, "kappa": kappa} for epsilon in epsilon_grid for kappa in kappa_grid]
    if gate_id == "G4_probability_superiority":
        return [{**base, "p_min": value} for value in p_grid]
    if gate_id == "G6_two_stage":
        return [{**base, "p_override_min": value} for value in p_grid]
    if gate_id == "G7_safe_regret":
        return [
            {**base, "delta_harm": delta, "epsilon_harm": harm, "epsilon_benefit": benefit}
            for delta in config.selective_calibration.delta_harm_grid
            for harm in config.selective_calibration.epsilon_harm_grid
            for benefit in epsilon_grid
        ]
    return [base]


def _gate_inputs(
    prediction: Any,
    selector: BaseSource,
    task_id: str,
    global_state: np.ndarray,
    action_features: np.ndarray,
    parameters: dict[str, Any],
    calibration: CalibrationArtifact | None,
    benefit_probability: np.ndarray | None = None,
    row_base: BaseDecision | None = None,
) -> GateInputs:
    if row_base is None:
        if isinstance(selector, FrozenInitialRowBase):
            raise ValueError("Initial-base gate input requires the verified row anchor.")
        row_base = selector.select(context_from_task_id(task_id))
    base = row_base
    r5 = residual_advantages(prediction, base.action, 5)
    r10 = residual_advantages(prediction, base.action, 10) if prediction.q10_mean is not None else None
    lower5 = lower10 = None
    if calibration is not None:
        if calibration.horizon == SHORT_HORIZON:
            lower5 = conformal_lower_bounds(r5.mean, calibration)
            lower5[base.action] = 0.0
        elif r10 is not None:
            lower10 = conformal_lower_bounds(r10.mean, calibration)
            lower10[base.action] = 0.0
    p_benefit = (
        benefit_probability
        if benefit_probability is not None
        else (
            None
            if r5.members is None
            else empirical_probability(r5.members, float(parameters.get("epsilon_benefit", 1e-3)), greater=True)
        )
    )
    p_harm = (
        None
        if r5.members is None
        else empirical_probability(r5.members, -float(parameters.get("epsilon_harm", 1e-3)), greater=False)
    )
    equivalence = candidate_equivalence(
        action_features, heuristic=bool(parameters.get("candidate_feature_heuristic", False))
    )
    trust = hard_trust_metadata(global_state, action_features, equivalence, parameters)
    return GateInputs(
        base=base,
        prediction=prediction,
        residual5=r5,
        residual10=r10,
        equivalence=equivalence,
        lower_bounds5=lower5
        if lower5 is not None
        else (r5.mean if parameters.pop("_tuning_conformal", False) else None),
        lower_bounds10=lower10,
        probability_benefit5=p_benefit,
        probability_harm5=p_harm,
        override_probability=float(parameters.get("_override_probability", 1.0)),
        metadata={
            **trust,
            "trust_probability": float(parameters.get("_trust_probability", trust["trust_probability"])),
        },
    )


def _selected_actions(
    gate: OverrideGate,
    predictor_values: list[Any],
    selector: BaseSource,
    dataset: BranchDataset,
    indices: np.ndarray,
    parameters: dict[str, Any],
    calibration: CalibrationArtifact | None = None,
    override_probabilities: np.ndarray | None = None,
    benefit_probabilities: np.ndarray | None = None,
    trust_probabilities: np.ndarray | None = None,
) -> NDArray[np.int64]:
    if gate.gate_id == "G2_hysteresis" or parameters.get("mode") == "adaptive_commitment":
        raise ValueError(
            "Stateful gate tuning requires verified ordered source-history replay; "
            "isolated branch rows are unsupported."
        )
    actions = []
    for position, (prediction, index) in enumerate(zip(predictor_values, indices, strict=True)):
        local = dict(parameters)
        if gate.gate_id == "G5_conformal_lcb" and calibration is None:
            local["_tuning_conformal"] = True
        if override_probabilities is not None:
            local["_override_probability"] = float(override_probabilities[position])
        if trust_probabilities is not None:
            local["_trust_probability"] = float(trust_probabilities[position])
        inputs = _gate_inputs(
            prediction,
            selector,
            str(dataset.arrays["task_id"][index]),
            dataset.arrays["global_state"][index],
            dataset.arrays["action_features"][index],
            local,
            calibration,
            None if benefit_probabilities is None else benefit_probabilities[position],
            row_base=_row_base(selector, dataset, int(index)),
        )
        actions.append(gate.decide(inputs).selected_action)
    result: NDArray[np.int64] = np.asarray(actions, dtype=np.int64)
    return result


def _classifier_features(dataset: BranchDataset, indices: np.ndarray) -> NDArray[np.float64]:
    """Build deployable state plus candidate-row classifier features."""
    result: NDArray[np.float64] = np.concatenate(
        (
            np.asarray(dataset.arrays["global_state"][indices], dtype=np.float64),
            np.asarray(dataset.arrays["action_features"][indices], dtype=np.float64).reshape(len(indices), -1),
        ),
        axis=1,
    )
    return result


def _fit_override_classifier(
    dataset: BranchDataset,
    indices: np.ndarray,
    selector: BaseSource,
    config: DictConfig,
    horizon: Horizon,
) -> OverrideWorthwhileClassifier:
    """Fit G6 stage one without assigning tied best-action labels."""
    truth = dataset.arrays["q5" if horizon == SHORT_HORIZON else "q10"][indices]
    bases = _base_actions(selector, dataset, indices)
    residuals = truth - truth[np.arange(len(indices)), bases, None]
    classifier_kind = cast(
        "Literal['shared_neural', 'extra_trees', 'hist_gradient_boosting']",
        str(config.selective_gate.get("classifier", "extra_trees")),
    )
    classifier = OverrideWorthwhileClassifier(classifier_kind, int(config.seed))
    return classifier.fit(
        _classifier_features(dataset, indices),
        residuals,
        epsilon_worthwhile=float(config.selective_gate.parameters.get("epsilon_worthwhile", 1e-3)),
    )


def _calibrate_conformal(
    predictor: DelayedValuePredictor,
    selector: BaseSource,
    dataset: BranchDataset,
    indices: np.ndarray,
    config: DictConfig,
    horizon: Horizon,
) -> CalibrationArtifact:
    predictions = _predictions(predictor, dataset, indices)
    means = np.stack([value.q5_mean if horizon == SHORT_HORIZON else value.q10_mean for value in predictions])
    truth = dataset.arrays["q5" if horizon == SHORT_HORIZON else "q10"][indices]
    bases = _base_actions(selector, dataset, indices)
    rows = np.arange(len(indices))
    predicted_residual = means - means[rows, bases, None]
    true_residual = truth - truth[rows, bases, None]
    tasks = dataset.arrays["task_id"][indices].astype(str)
    return calibrate_task_max_conformal(
        predicted_residual,
        true_residual,
        tasks,
        base_actions=bases.tolist(),
        delta=float(config.selective_calibration.conformal_delta),
        horizon=horizon,
        task_hash=canonical_hash(sorted(set(tasks.tolist()))),
        minimum_tasks=int(config.selective_calibration.minimum_calibration_tasks),
        mode=str(config.selective_calibration.conformal_mode),
    )


def _fit_probability_calibrator(
    predictor: DelayedValuePredictor,
    selector: BaseSource,
    dataset: BranchDataset,
    indices: np.ndarray,
    config: DictConfig,
) -> ProbabilityCalibrator:
    """Calibrate empirical ensemble benefit rates on held-aside train tasks."""
    predictions = _predictions(predictor, dataset, indices)
    bases = _base_actions(selector, dataset, indices)
    truth = np.asarray(dataset.arrays["q5"][indices], dtype=np.float64)
    true_residual = truth - truth[np.arange(len(indices)), bases, None]
    scores: list[float] = []
    labels: list[bool] = []
    threshold = float(config.selective_gate.parameters.get("epsilon_benefit", 1e-3))
    for prediction, base_action, residual_truth in zip(predictions, bases, true_residual, strict=True):
        residual = residual_advantages(prediction, int(base_action), 5)
        if residual.members is None:
            raise ValueError("G4 probability calibration requires an ensemble predictor.")
        empirical = empirical_probability(residual.members, threshold, greater=True)
        for action in range(5):
            if action != base_action:
                scores.append(float(empirical[action]))
                labels.append(bool(residual_truth[action] > threshold))
    probability_method = cast("Literal['logistic', 'isotonic']", str(config.selective_calibration.probability_method))
    return ProbabilityCalibrator(probability_method).fit(np.asarray(scores), np.asarray(labels))


def _calibrated_probabilities(
    calibrator: ProbabilityCalibrator | None,
    predictions: list[Any],
    selector: BaseSource,
    dataset: BranchDataset,
    indices: np.ndarray,
    *,
    epsilon_benefit: float,
) -> np.ndarray | None:
    """Apply one frozen probability map to member-wise residual rates."""
    if calibrator is None:
        return None
    output: NDArray[np.float64] = np.zeros((len(indices), 5), dtype=np.float64)
    for position, (prediction, index) in enumerate(zip(predictions, indices, strict=True)):
        base = _row_base(selector, dataset, int(index)).action
        residual = residual_advantages(prediction, base, 5)
        if residual.members is None:
            raise ValueError("Calibrated probability inference requires ensemble members.")
        empirical = empirical_probability(residual.members, epsilon_benefit, greater=True)
        output[position] = np.asarray(calibrator.predict(empirical), dtype=np.float64)
        output[position, base] = 0.0
    return cast("NDArray[np.float64]", output)  # type: ignore[no-any-return]


def _trust_features(dataset: BranchDataset, indices: np.ndarray, parameters: dict[str, Any]) -> NDArray[np.float64]:
    """Build fixed deployable trust features without future labels."""
    rows: list[NDArray[np.float64]] = []
    for index in indices:
        equivalence = candidate_equivalence(dataset.arrays["action_features"][index])
        metadata = hard_trust_metadata(
            dataset.arrays["global_state"][index],
            dataset.arrays["action_features"][index],
            equivalence,
            parameters,
        )
        rows.append(trust_feature_vector(metadata))
    result: NDArray[np.float64] = np.stack(rows)
    return result


def _fit_reliability_classifier(
    predictor: DelayedValuePredictor,
    selector: BaseSource,
    dataset: BranchDataset,
    indices: np.ndarray,
    config: DictConfig,
    horizon: Horizon,
) -> ReliabilityClassifier:
    """Fit optional G9 reliability on held-aside training tasks only."""
    predictions = _predictions(predictor, dataset, indices)
    bases = _base_actions(selector, dataset, indices)
    truth = np.asarray(dataset.arrays["q5" if horizon == SHORT_HORIZON else "q10"][indices])
    errors = []
    for position, (prediction, base) in enumerate(zip(predictions, bases, strict=True)):
        residual = residual_advantages(prediction, int(base), horizon)
        alternative = int(np.argmax(np.where(np.arange(5) == base, -np.inf, residual.mean)))
        true_residual = truth[position] - truth[position, base]
        errors.append(abs(float(residual.mean[alternative] - true_residual[alternative])))
    return ReliabilityClassifier(seed=int(config.seed)).fit(
        _trust_features(dataset, indices, dict(config.selective_gate.parameters)),
        np.asarray(errors),
        maximum_reliable_error=float(config.selective_gate.maximum_reliable_error),
    )


def fit(config: DictConfig) -> dict[str, Any]:
    """Claim a scientific run; resume only verified completed fits."""
    output = Path(str(config.selective_output.root)).resolve()
    if config.get("expected_source_revision") and str(config.expected_source_revision) != current_source_revision():
        raise ValueError("Queued selective-fit source changed after launch preparation.")
    branch = Path(str(config.selective_data.branch_root)).resolve()
    identity = {
        "schema": "selective-fit-claim-v2",
        "config": {
            key: OmegaConf.to_container(config[key], resolve=True)
            for key in (
                "selective_base",
                "selective_predictor",
                "selective_gate",
                "selective_calibration",
                "selective_policy",
                "selective_evaluation",
            )
        },
        "seed": int(config.seed),
        "partition_seed": int(config.selective_data.partition_seed),
        "train_sha256": file_sha256(branch / "branches_train.npz"),
        "dev_sha256": file_sha256(branch / "branches_dev.npz"),
        "normalizer_sha256": (
            file_sha256(Path(str(config.selective_data.normalizer))) if config.selective_data.normalizer else None
        ),
        "source_revision": current_source_revision(),
        "initial_base_sha256": (
            file_sha256(Path(str(config.selective_data.initial_base_bundle)))
            if config.selective_data.get("initial_base_bundle")
            else None
        ),
    }
    claim = output / "scientific_run_claim.json"
    if bool(config.selective_output.get("resume", False)):
        if not claim.is_file() or json.loads(claim.read_text()) != identity:
            raise ValueError("Incompatible selective resume: configuration, source, or data changed.")
        completion = output / "selective_training_complete.json"
        if not completion.is_file():
            raise RuntimeError(
                "Incomplete selective fit cannot resume without optimizer/RNG checkpoints; use a new root."
            )
        load_policy_bundle(output / "policy_bundle.json")
        return cast("dict[str, Any]", json.loads(completion.read_text()))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite selective run {output}.")
    output.mkdir(parents=True, exist_ok=True)
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(identity, stream, sort_keys=True, allow_nan=False)
    return _fit_claimed(config)


def _fit_claimed(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Run the task-disjoint selective fitting, tuning, calibration, and dev protocol."""
    branch_root = Path(str(config.selective_data.branch_root)).resolve()
    output = Path(str(config.selective_output.root)).resolve()
    train = BranchDataset(branch_root / "branches_train.npz")
    dev = BranchDataset(branch_root / "branches_dev.npz")
    validate_selective_branch_provenance(train.metadata, role="model_fit")
    validate_selective_branch_provenance(dev.metadata, role="dev")
    train_tasks = sorted(set(train.arrays["task_id"].astype(str).tolist()))
    dev_tasks = sorted(set(dev.arrays["task_id"].astype(str).tolist()))
    partition = selective_train_partition(train_tasks, seed=int(config.selective_data.partition_seed))
    partition_hash = validate_task_partitions(partition["partitions"], train_tasks, dev_tasks)  # type: ignore[arg-type]
    atomic_json(output / "selective_train_partition_manifest.json", partition)
    fit_indices = _indices_for_tasks(train, partition["partitions"]["model_fit"])  # type: ignore[index]
    tune_indices = _indices_for_tasks(train, partition["partitions"]["gate_tune"])  # type: ignore[index]
    calibration_indices = _indices_for_tasks(train, partition["partitions"]["uncertainty_calibration"])  # type: ignore[index]
    fit_tune_indices = np.concatenate((fit_indices, tune_indices))
    horizon: Horizon = int(config.selective_policy.horizon)  # type: ignore[assignment]
    source_revision = current_source_revision()
    normalizer = _normalizer(str(config.selective_data.normalizer or "") or None)

    provisional_base = _fit_base(train, fit_indices, config, horizon=horizon, source_revision=source_revision)
    provisional_predictor = _fit_predictor(train, fit_indices, config, normalizer)
    tune_predictions = _predictions(provisional_predictor, train, tune_indices)
    provisional_classifier = (
        _fit_override_classifier(train, fit_indices, provisional_base, config, horizon)
        if str(config.selective_gate.id) == "G6_two_stage"
        else None
    )
    tune_override_probability = (
        None
        if provisional_classifier is None
        else provisional_classifier.predict_probability(_classifier_features(train, tune_indices))
    )
    tune_base = _base_actions(provisional_base, train, tune_indices)
    target = train.arrays["q5" if horizon == SHORT_HORIZON else "q10"][tune_indices]
    contexts = np.asarray(
        [exact_context_key(context_from_task_id(str(train.arrays["task_id"][index]))) for index in tune_indices]
    )
    constraints_payload = OmegaConf.to_container(config.selective_calibration.constraints, resolve=True)
    if not isinstance(constraints_payload, dict):
        raise TypeError("Selective gate constraints must resolve to a mapping.")
    constraints = GateConstraints(**cast("dict[str, Any]", constraints_payload))
    frontier: list[dict[str, Any]] = []
    gate_id = str(config.selective_gate.id)
    for parameters in _gate_parameters(config):
        gate = build_gate(gate_id, parameters)
        selected = _selected_actions(
            gate,
            tune_predictions,
            provisional_base,
            train,
            tune_indices,
            parameters,
            calibration=None,
            override_probabilities=tune_override_probability,
        )
        frontier.append(
            risk_coverage_gain_row(
                selected,
                tune_base,
                target,
                train.arrays["task_id"][tune_indices].astype(str),
                contexts,
                parameters=parameters,
                constraints=constraints,
            )
        )
    selected_frontier = select_frontier_row(frontier)
    selected_parameters = dict(selected_frontier["parameters"])
    if selected_frontier.get("fallback"):
        gate_id = "G0_base_only"
        selected_parameters = {"horizon": horizon}
    atomic_json(
        output / "gate_tuning_frontier.json",
        {"rows": frontier, "selected": selected_frontier, "selection_data": "train/gate_tune"},
    )

    # Refitting after tuning would invalidate its empirical risk frontier.
    final_base = provisional_base
    final_predictor = provisional_predictor
    final_classifier = provisional_classifier if gate_id == "G6_two_stage" else None
    calibration = None
    if gate_id == "G5_conformal_lcb":
        calibration = _calibrate_conformal(final_predictor, final_base, train, calibration_indices, config, horizon)
        if calibration.fallback_mode is not None:
            gate_id, selected_parameters = "G0_base_only", {"horizon": horizon}
    probability_calibrator = None
    if gate_id == "G4_probability_superiority":
        probability_calibrator = _fit_probability_calibrator(
            final_predictor, final_base, train, calibration_indices, config
        )
    reliability_classifier = None
    if gate_id == "G9_gp_trust_wrapper" and str(config.selective_gate.get("trust_mode", "hard")) == "learned":
        reliability_classifier = _fit_reliability_classifier(
            final_predictor, final_base, train, calibration_indices, config, horizon
        )
    base_path = output / "base_selector_registry.json"
    if isinstance(final_base, FrozenInitialRowBase):
        destination = output / "initial_base"
        destination.mkdir()
        shutil.copy2(final_base.path, destination / "base_bundle.json")
        shutil.copy2(final_base.path.parent / "predictor.pkl", destination / "predictor.pkl")
        base_path = destination / "base_bundle.json"
        base_registry_hash = final_base.model.semantic_hash
    else:
        base_registry_hash = save_base_selector(final_base, base_path)
    comparator_registries: dict[str, dict[str, str]] = {}
    original_base = str(config.selective_base.id)
    for comparator_id in ("B0_global_static", "B1_context_mean", "B2_context_lcb", "B5_context_phase_ablation"):
        comparison_config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        if not isinstance(comparison_config, DictConfig):
            raise TypeError("Resolved selective comparator config must be a mapping.")
        comparison_config.selective_base.id = comparator_id
        if "initial_base_bundle" in comparison_config.selective_data:
            comparison_config.selective_data.initial_base_bundle = None
        comparison = _fit_base(
            train, fit_tune_indices, comparison_config, horizon=horizon, source_revision=source_revision
        )
        comparator_path = output / "base_comparators" / f"{comparator_id}.json"
        assert isinstance(comparison, FittedBaseSelector)
        comparator_hash = save_base_selector(comparison, comparator_path)
        comparator_registries[comparator_id] = {
            "path": str(comparator_path),
            "registry_hash": comparator_hash,
            "file_sha256": file_sha256(comparator_path),
        }
    if str(config.selective_base.id) != original_base:
        raise RuntimeError("Comparator fitting mutated the resolved primary base configuration.")
    predictor_metadata = save_predictor(final_predictor, output / "predictor")
    override_classifier_artifact = None
    if final_classifier is not None:
        override_classifier_artifact = save_calibration_model(
            final_classifier,
            output / "override_worthwhile_classifier.pkl",
            kind=f"override_worthwhile_{final_classifier.kind}",
        )
        override_classifier_artifact["feature_schema"] = "global_state_plus_flat_action_features"
    probability_calibration_artifact = (
        None
        if probability_calibrator is None
        else save_calibration_model(
            probability_calibrator,
            output / "probability_calibrator.pkl",
            kind=f"benefit_probability_{probability_calibrator.method}",
        )
    )
    reliability_calibration_artifact = (
        None
        if reliability_classifier is None
        else save_calibration_model(
            reliability_classifier,
            output / "reliability_classifier.pkl",
            kind="deployable_prediction_reliability_extra_trees",
        )
    )
    calibration_path = None
    if calibration is not None:
        calibration_path = output / "uncertainty_calibration.json"
        atomic_json(calibration_path, calibration.to_dict())

    training_side_predictions = _predictions(final_predictor, train, fit_tune_indices)
    training_side_gate = build_gate(gate_id, selected_parameters)
    training_side_selected = _selected_actions(
        training_side_gate,
        training_side_predictions,
        final_base,
        train,
        fit_tune_indices,
        selected_parameters,
        calibration,
        override_probabilities=(
            None
            if final_classifier is None
            else final_classifier.predict_probability(_classifier_features(train, fit_tune_indices))
        ),
        benefit_probabilities=_calibrated_probabilities(
            probability_calibrator,
            training_side_predictions,
            final_base,
            train,
            fit_tune_indices,
            epsilon_benefit=float(selected_parameters.get("epsilon_benefit", 1e-3)),
        ),
        trust_probabilities=(
            None
            if reliability_classifier is None
            else reliability_classifier.predict_probability(
                _trust_features(train, fit_tune_indices, selected_parameters)
            )
        ),
    )
    action_counts = np.bincount(training_side_selected, minlength=5)
    action_probabilities = action_counts / action_counts.sum()
    projection_controls = {
        "source": "offline_train_model_fit_plus_gate_tune_only",
        "source_task_hash": canonical_hash(sorted(set(train.arrays["task_id"][fit_tune_indices].astype(str).tolist()))),
        "modal_action": int(np.flatnonzero(action_counts == action_counts.max())[0]),
        "marginal_action_probabilities": action_probabilities.tolist(),
        "action_counts": action_counts.tolist(),
    }

    dev_indices: NDArray[np.int64] = np.arange(len(dev), dtype=np.int64)
    dev_predictions = _predictions(final_predictor, dev, dev_indices)
    final_gate = build_gate(gate_id, selected_parameters)
    dev_selected = _selected_actions(
        final_gate,
        dev_predictions,
        final_base,
        dev,
        dev_indices,
        selected_parameters,
        calibration,
        override_probabilities=(
            None
            if final_classifier is None
            else final_classifier.predict_probability(_classifier_features(dev, dev_indices))
        ),
        benefit_probabilities=_calibrated_probabilities(
            probability_calibrator,
            dev_predictions,
            final_base,
            dev,
            dev_indices,
            epsilon_benefit=float(selected_parameters.get("epsilon_benefit", 1e-3)),
        ),
        trust_probabilities=(
            None
            if reliability_classifier is None
            else reliability_classifier.predict_probability(_trust_features(dev, dev_indices, selected_parameters))
        ),
    )
    dev_base = _base_actions(final_base, dev, dev_indices)
    dev_target = dev.arrays["q5" if horizon == SHORT_HORIZON else "q10"]
    metrics = evaluate_selective_actions(
        dev_target,
        dev_selected,
        dev_base,
        dev.arrays["task_id"].astype(str),
        epsilon_harm=float(config.selective_calibration.constraints.epsilon_harm),
    )
    rows = np.arange(len(dev))
    gain = dev_target[rows, dev_selected] - dev_target[rows, dev_base]
    metrics["paired_task_bootstrap"] = grouped_bootstrap_gain(
        gain,
        dev.arrays["task_id"].astype(str),
        resamples=int(config.selective_evaluation.bootstrap_resamples),
        seed=int(config.seed) + 7001,
    )
    atomic_json(output / "dev_branch_metrics.json", metrics)

    predictor_manifest = Path(predictor_metadata["manifest_path"])
    model_hashes = [item["model_hash"] for item in predictor_metadata.get("members", [predictor_metadata])]
    payload = {
        "policy_id": f"selective-{config.selective_policy.id}-seed{config.seed}",
        "base_selector_registry": str(base_path),
        "base_selector_registry_sha256": file_sha256(base_path),
        "base_selector_registry_hash": base_registry_hash,
        "predictor_manifest": str(predictor_manifest),
        "predictor_manifest_sha256": file_sha256(predictor_manifest),
        "predictor_model_hashes": model_hashes,
        "normalizer_hash": predictor_metadata.get("normalizer_hash", ""),
        "gate_id": gate_id,
        "gate_parameters": selected_parameters,
        "uncertainty_calibration_artifact": None if calibration_path is None else str(calibration_path),
        "uncertainty_calibration_sha256": None if calibration_path is None else file_sha256(calibration_path),
        "train_partition_hash": partition_hash,
        "gate_tune_task_hash": canonical_hash(partition["partitions"]["gate_tune"]),  # type: ignore[index]
        "uncertainty_calibration_task_hash": canonical_hash(
            partition["partitions"]["uncertainty_calibration"]  # type: ignore[index]
        ),
        "feature_schema_hash": predictor_metadata.get("feature_schema_hash", canonical_sha256([13, [5, 4]])),
        "action_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "horizon": horizon,
        "interaction_frequency": 5,
        "candidate_equivalence_tolerance": float(config.selective_policy.candidate_equivalence_tolerance),
        "code_revision": source_revision,
        "branch_train_sha256": file_sha256(train.path),
        "branch_dev_sha256": file_sha256(dev.path),
        "branch_protocol_hash": str(train.metadata["manifest_hash"]),
        "base_comparator_registries": comparator_registries,
        "projection_controls": projection_controls,
        "override_classifier_artifact": override_classifier_artifact,
        "probability_calibration_artifact": probability_calibration_artifact,
        "reliability_calibration_artifact": reliability_calibration_artifact,
        "dev_task_ids": dev_tasks,
        "emergency_global_action": (
            2 if isinstance(final_base, FrozenInitialRowBase) else final_base.registry.selected_actions["global"]
        ),
        "checkpoint_selection_metric": "dev/selected_value_over_exact_frozen_base",
        "checkpoint_selection_value": metrics["selected_minus_base"],
        "dev_used_for_fitting": False,
        "holdout_accessed": False,
    }
    if isinstance(final_base, FrozenInitialRowBase):
        payload.update(
            base_target_semantics="full_static_terminal_loss",
            calibration_base_semantic_hash=final_base.model.semantic_hash,
            initial_feature_settings=final_base.model.metadata["feature_settings"],
        )
    bundle = finalize_policy_bundle(payload, output / "policy_bundle.json")
    resolved = OmegaConf.to_container(config, resolve=False)
    atomic_json(output / "resolved_config.json", resolved)
    completion = {
        "schema_version": "dacbo-selective-fit-complete-v1",
        "status": "complete",
        "policy_id": bundle["policy_id"],
        "policy_bundle": str(output / "policy_bundle.json"),
        "policy_artifact_hash": bundle["policy_artifact_hash"],
        "selected_gate": gate_id,
        "selected_gate_parameters": selected_parameters,
        "selected_value_over_base": metrics["selected_minus_base"],
        "dev_used_once": True,
        "holdout_accessed": False,
    }
    atomic_json(output / "selective_training_complete.json", completion)
    return completion


@hydra.main(version_base=None, config_path="../configs", config_name="selective_fit")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Fit one resolved selective policy."""
    print(json.dumps(fit(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
