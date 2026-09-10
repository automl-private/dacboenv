"""Initial terminal-base fitting, relocation and real deployment tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.real_env import real_structured_bbob_smoke_env
from dacboenv.experiment.snapshot_branch import BOSnapshot, replay_snapshot
from dacboenv.optimizer import DACBOEnvOptimizer
from dacboenv.policy.initial_base import InitialConditionedWEIPolicy
from dacboenv.selective_wei.artifacts import finalize_policy_bundle, load_policy_bundle
from dacboenv.selective_wei.initial_base_model import fit_initial_base, load_initial_base, save_initial_base
from dacboenv.selective_wei.initial_features import InitialFeatureSettings
from dacboenv.selective_wei.policy import SelectiveWEIPolicy
from dacboenv.selective_wei.predictors import EnsembleDelayedValuePredictor, fit_sklearn_predictor, save_predictor
from dacboenv.utils import carps_optimizer
from omegaconf import OmegaConf


def test_initial_base_relocation_and_real_frozen_episode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A fitted base stays fixed over two real f5 blocks and verified replay."""
    original = carps_optimizer.get_task_config

    def tiny_task(task_id: str) -> Any:
        config = OmegaConf.create(OmegaConf.to_container(original(task_id), resolve=False))
        config.task.optimization_resources.n_trials = 12
        return config

    monkeypatch.setattr(carps_optimizer, "get_task_config", tiny_task)
    env = real_structured_bbob_smoke_env("bbob/2/3/0", 201, interaction_frequency=5)
    clone = real_structured_bbob_smoke_env("bbob/2/3/0", 201, interaction_frequency=5)
    settings = asdict(InitialFeatureSettings(probe_count=8))
    try:
        observation, _ = env.reset()
        initial = env.get_initial_context(settings)
        contexts = [replace(initial, episode_id=f"synthetic-{i}") for i in range(4)]
        model = fit_initial_base(
            contexts,
            np.tile([1.0, 1.0, 1.0, 0.0, 1.0], (4, 1)),
            ["synthetic-train-a", "synthetic-train-a", "synthetic-train-b", "synthetic-train-b"],
            groups=("M", "D", "G", "U"),
            kind="ridge",
        )
        model.metadata["feature_settings"] = settings
        model.metadata.update(partition_manifest_hash="synthetic", train_dataset_hash="synthetic")
        save_initial_base(model, tmp_path / "original")
        shutil.copytree(tmp_path / "original", tmp_path / "relocated")
        loaded = load_initial_base(tmp_path / "relocated" / "base_bundle.json")
        assert loaded.semantic_hash == model.semantic_hash
        assert np.array_equal(loaded.predict(contexts), model.predict(contexts))
        policy = InitialConditionedWEIPolicy(env, str(tmp_path / "relocated" / "base_bundle.json"))
        assert policy(observation) == 3
        observation, _, _, _, _ = env.step(3)
        state = env.initial_anchor_state()
        assert policy(observation) == 3
        clone.reset()
        clone.get_initial_context(settings)
        clone.step(3)
        clone.restore_initial_anchor(state)
        assert clone.initial_anchor_state() == state
        snapshot = BOSnapshot(
            "bbob/2/3/0",
            201,
            action_history=(3,),
            interaction_frequency=5,
            initial_anchor_json=json.dumps(env.portable_initial_anchor()),
        )
        replayed = replay_snapshot(
            snapshot, lambda task, seed: real_structured_bbob_smoke_env(task, seed, interaction_frequency=5)
        )
        try:
            assert replayed.initial_anchor_state() == state
        finally:
            replayed.close()
        observation, reward, terminated, truncated, _ = env.step(policy(observation))
        assert terminated or truncated
        assert np.isfinite(reward)
        assert env.get_n_finished_trials() == 12
        assert env.get_initial_context(settings) is initial
        observation, _ = env.reset()
        _exercise_real_selective(env, observation, tmp_path, loaded.semantic_hash)
        _exercise_outer_carps(env, initial, settings, tmp_path)
    finally:
        env.close()
        clone.close()


def _exercise_outer_carps(reference_env: Any, initial: Any, settings: dict[str, Any], root: Path) -> None:
    """Run CARP-S's real ask/objective/tell loop, including its initial design."""
    external = real_structured_bbob_smoke_env("bbob/2/3/0", 201, interaction_frequency=5)
    external._evaluation_mode = True
    external.configure_initial_context(settings)
    optimizer = DACBOEnvOptimizer(
        task=reference_env._carps_solver.task,
        dacboenv=external,
        seed=201,
        policy_class=InitialConditionedWEIPolicy,
        policy_kwargs={"base_bundle": str(root / "relocated" / "base_bundle.json")},
    )
    try:
        optimizer.setup_optimizer()
        assert external.get_n_finished_trials() == 0
        assert external.portable_initial_anchor() is None
        optimizer.run()
        assert external.get_n_finished_trials() == 12
        assert optimizer.trial_counter == 12
        anchor = external.get_initial_context(settings)
        assert anchor.values == initial.values
        assert external.initial_anchor_state()["decision"]["action"] == 3
    finally:
        external.close()


def _exercise_real_selective(env: Any, observation: Any, root: Path, base_hash: str) -> None:
    """Force one engineering override followed by safe return to the exact a0."""
    states = np.tile(observation["global_state"], (8, 1))
    features = np.tile(observation["action_features"], (8, 1, 1))
    targets = np.tile([0.0, 0.0, 0.0, 0.0, 1.0], (8, 1))
    members = [
        fit_sklearn_predictor(
            states, features, targets, targets, kind="extra_trees", seed=seed, model_id=f"engineering-{seed}"
        )
        for seed in range(3)
    ]
    predictor = EnsembleDelayedValuePredictor(members, ensemble_kind="independent_seed")
    metadata = save_predictor(predictor, root / "local_predictor")
    manifest = Path(metadata["manifest_path"])
    base_path = root / "relocated" / "base_bundle.json"
    payload = {
        "base_selector_registry": str(base_path),
        "base_selector_registry_sha256": file_sha256(base_path),
        "base_target_semantics": "full_static_terminal_loss",
        "calibration_base_semantic_hash": base_hash,
        "predictor_manifest": str(manifest),
        "predictor_manifest_sha256": file_sha256(manifest),
        "predictor_model_hashes": [member.model_hash for member in members],
        "normalizer_hash": "",
        "gate_id": "G3_ensemble_lcb",
        "gate_parameters": {"horizon": 5, "kappa": 1.0, "epsilon": 0.0, "minimum_ensemble_size": 3},
        "train_partition_hash": "synthetic",
        "gate_tune_task_hash": "synthetic",
        "uncertainty_calibration_task_hash": "synthetic",
        "feature_schema_hash": "structured",
        "action_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "horizon": 5,
        "interaction_frequency": 5,
        "code_revision": "engineering-fixture",
    }
    bundle_path = root / "selective.json"
    bundle = finalize_policy_bundle(payload, bundle_path)
    log = root / "selective_decisions.jsonl"
    policy = SelectiveWEIPolicy(env, str(bundle_path), bundle["policy_artifact_hash"], str(log))
    assert policy(observation) == 4
    assert policy(observation) == 4
    assert len(log.read_text().splitlines()) == 1
    observation, reward, _, _, _ = env.step(4)
    assert np.isfinite(reward)

    def unavailable(_observation: Any) -> Any:
        raise RuntimeError("Injected engineering predictor failure")

    policy.predictor.predict = unavailable
    assert policy(observation) == 3
    assert env.initial_anchor_state()["decision"]["action"] == 3
    _, reward, terminated, truncated, _ = env.step(3)
    assert np.isfinite(reward)
    assert terminated or truncated
    assert len(log.read_text().splitlines()) == 2
    relocated = root / "complete_bundle_relocated"
    shutil.copytree(root, relocated, ignore=shutil.ignore_patterns("complete_bundle_relocated"))
    moved_bundle = load_policy_bundle(relocated / "selective.json")
    assert moved_bundle["policy_artifact_hash"] == bundle["policy_artifact_hash"]
    assert Path(moved_bundle["predictor_manifest"]).is_relative_to(relocated)
    restored = SelectiveWEIPolicy(env, str(relocated / "selective.json"), bundle["policy_artifact_hash"])
    restored.set_policy_state(json.loads(json.dumps(policy.get_policy_state())))
    assert restored.get_policy_state() == policy.get_policy_state()
