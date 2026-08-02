"""Dummy-SB3 coverage for learned snapshot-headroom controls."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from dacboenv.experiment.analyze_snapshot_policy import analyze_snapshot_policy
from dacboenv.experiment.collect_snapshots import (
    configured_structured_action_space,
    observation_digest,
    write_snapshots,
)
from dacboenv.experiment.run_snapshot_branches import BRANCH_SCHEMA_VERSION
from dacboenv.experiment.snapshot_branch import BOSnapshot, snapshot_record_digest
from gymnasium import spaces
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_REPOSITORY = Path(__file__).parents[1]
_VALIDATION_MANIFEST = _REPOSITORY / "dacboenv/configs/instance_sets/bbob_validation.yaml"
_VALIDATION_ID = "bbob-validation-v1"
_VALIDATION_HASH = "36ed3fb56ddc141069b1efad21f4f2ee51d98fed5a0ebaf8c1cdc0d3fcfec196"
_VALIDATION_CONTEXTS = (("bbob/4/2/1", 1349011988), ("bbob/4/7/1", 2024774586))


@pytest.mark.parametrize(
    ("action_space_id", "expected"),
    [
        ("WEI-discrete-f1", "wei"),
        ("LCB-quantile-discrete-f1", "lcb_quantile"),
        ("UCB-quantile-discrete-f1", "ucb_quantile"),
        ("AF-select-f1", "af_selection"),
    ],
)
def test_stage_a_action_space_ids_resolve_fail_closed(action_space_id: str, expected: str) -> None:
    assert configured_structured_action_space(OmegaConf.create({"action_space_id": action_space_id})) == expected


class _PolicyEnv(gym.Env[dict[str, np.ndarray], int]):
    action_space = spaces.Discrete(3)
    observation_space = spaces.Dict(
        {
            "global_state": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
            "action_features": spaces.Box(-10.0, 10.0, shape=(3, 2), dtype=np.float32),
        }
    )

    def __init__(self, task_id: str = _VALIDATION_CONTEXTS[0][0], inner_seed: int = _VALIDATION_CONTEXTS[0][1]) -> None:
        self.current_task_id = task_id
        self.current_seed = inner_seed
        self.action_space_name = "wei"
        self.interaction_frequency = 1
        self.finished = 0
        self._smac_instance = SimpleNamespace(_scenario=SimpleNamespace(n_trials=10))

    def _observation(self) -> dict[str, np.ndarray]:
        task_marker = 0.0 if self.current_task_id == _VALIDATION_CONTEXTS[0][0] else 1.0
        return {
            "global_state": np.asarray([task_marker, self.finished / 10.0], dtype=np.float32),
            "action_features": np.asarray(((1, 0), (0, 1), (-1, 0)), dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.finished = 0
        return self._observation(), {"task_id": self.current_task_id, "inner_seed": self.current_seed}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:  # noqa: ARG002
        self.finished += 1
        return self._observation(), 0.0, self.finished >= 2, False, {}

    def get_observation(self) -> dict[str, np.ndarray]:
        return self._observation()

    def get_n_finished_trials(self) -> int:
        return self.finished

    @staticmethod
    def get_incumbent_cost() -> float:
        return 10.0


def _factory(task_id: str, inner_seed: int, action_space_name: str) -> _PolicyEnv:
    if action_space_name != "wei":
        raise ValueError(action_space_name)
    return _PolicyEnv(task_id, inner_seed)


class _LateSpacePolicyEnv(gym.Env[dict[str, np.ndarray], int]):
    """Policy environment whose spaces appear only after the first reset."""

    def __init__(self, task_id: str, inner_seed: int) -> None:
        self.current_task_id = task_id
        self.current_seed = inner_seed
        self.action_space_name = "wei"
        self.interaction_frequency = 1
        self.finished = 0
        self._smac_instance = SimpleNamespace(_scenario=SimpleNamespace(n_trials=10))

    def _observation(self) -> dict[str, np.ndarray]:
        task_marker = 0.0 if self.current_task_id == _VALIDATION_CONTEXTS[0][0] else 1.0
        return {
            "global_state": np.asarray([task_marker, self.finished / 10.0], dtype=np.float32),
            "action_features": np.asarray(((1, 0), (0, 1), (-1, 0)), dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.action_space = _PolicyEnv.action_space
        self.observation_space = _PolicyEnv.observation_space
        self.finished = 0
        return self._observation(), {"task_id": self.current_task_id, "inner_seed": self.current_seed}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:  # noqa: ARG002
        self.finished += 1
        return self._observation(), 0.0, self.finished >= 2, False, {}

    def get_observation(self) -> dict[str, np.ndarray]:
        return self._observation()

    def get_n_finished_trials(self) -> int:
        return self.finished

    @staticmethod
    def get_incumbent_cost() -> float:
        return 10.0


def _late_factory(task_id: str, inner_seed: int, action_space_name: str) -> _LateSpacePolicyEnv:
    if action_space_name != "wei":
        raise ValueError(action_space_name)
    return _LateSpacePolicyEnv(task_id, inner_seed)


def _snapshots(*, action_space: str = "wei", source_hash: str = _VALIDATION_HASH) -> tuple[BOSnapshot, ...]:
    return tuple(
        BOSnapshot(
            task_id,
            inner_seed,
            action_space=action_space,
            interaction_frequency=1,
            budget_fraction=0.0,
            history_policy="initial_design_only",
            source_manifest=_VALIDATION_ID,
            source_manifest_hash=source_hash,
            code_commit="b" * 40,
            observation_hash=observation_digest(_PolicyEnv(task_id, inner_seed)._observation()),
            incumbent=10.0,
            initial_design_incumbent=10.0,
            reference_kind="exact",
            reference_value=0.0,
            reference_source="toy",
            reference_source_hash="c" * 64,
            reference_runtime_objective_transform="identity",
            reference_reporting_objective_transform="identity",
            reference_fidelity_json='"not_applicable"',
            reference_tolerance=1e-12,
            reference_benchmark_code_version="toy-code",
            reference_benchmark_data_version="toy-data",
        )
        for task_id, inner_seed in _VALIDATION_CONTEXTS
    )


def _write_branch_matrix(path: Path, snapshots: tuple[BOSnapshot, ...], *, tamper_hash: bool = False) -> None:
    values = ((3.0, 1.0, 0.0), (0.0, 1.0, 3.0))
    fields: tuple[str, ...] | None = None
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer: csv.DictWriter | None = None
        for snapshot_index, snapshot in enumerate(snapshots):
            for action, value in enumerate(values[snapshot_index]):
                row = {
                    "schema_version": BRANCH_SCHEMA_VERSION,
                    "snapshot_index": snapshot_index,
                    "snapshot_record_hash": ("0" * 64 if tamper_hash else snapshot_record_digest(snapshot)),
                    "task_id": snapshot.task_id,
                    "inner_seed": snapshot.inner_seed,
                    "action_space": snapshot.action_space,
                    "interaction_frequency": snapshot.interaction_frequency,
                    "action_history": json.dumps(snapshot.action_history),
                    "completed_evaluation_count": len(snapshot.completed_evaluations),
                    "budget_fraction": snapshot.budget_fraction,
                    "history_policy": snapshot.history_policy,
                    "outer_policy_seed": snapshot.outer_policy_seed,
                    "source_manifest": snapshot.source_manifest,
                    "source_manifest_hash": snapshot.source_manifest_hash,
                    "code_commit": snapshot.code_commit,
                    "observation_hash": snapshot.observation_hash,
                    "reference_kind": snapshot.reference_kind,
                    "reference_source": snapshot.reference_source,
                    "reference_source_hash": snapshot.reference_source_hash,
                    "reference_runtime_objective_transform": snapshot.reference_runtime_objective_transform,
                    "reference_reporting_objective_transform": snapshot.reference_reporting_objective_transform,
                    "reference_fidelity_json": snapshot.reference_fidelity_json,
                    "reference_tolerance": snapshot.reference_tolerance,
                    "reference_benchmark_code_version": snapshot.reference_benchmark_code_version,
                    "reference_benchmark_data_version": snapshot.reference_benchmark_data_version,
                    "snapshot_reference_value": snapshot.reference_value,
                    "snapshot_initial_design_incumbent": snapshot.initial_design_incumbent,
                    "action": action,
                    "horizon": 1,
                    "normalized_potential_improvement": value,
                    "configuration_trace": json.dumps([]),
                }
                if writer is None:
                    fields = tuple(row)
                    writer = csv.DictWriter(stream, fieldnames=fields)
                    writer.writeheader()
                writer.writerow(row)


def _write_run(run_root: Path, *, vecnormalize: bool = False, action_space_id: str = "WEI-discrete-f1") -> None:
    (run_root / ".hydra").mkdir(parents=True)
    (run_root / ".hydra/config.yaml").write_text(
        f"seed: 7\naction_space_id: {action_space_id}\nexperiment:\n  vecnormalize: {str(vecnormalize).lower()}\n",
        encoding="utf-8",
    )
    raw_vector_env = DummyVecEnv([_PolicyEnv])
    vector_env: DummyVecEnv | VecNormalize
    if vecnormalize:
        vector_env = VecNormalize(raw_vector_env, norm_obs=True, norm_reward=False)
        vector_env.save(run_root / "vecnormalize.pkl")
    else:
        vector_env = raw_vector_env
    model = PPO(
        "MultiInputPolicy",
        vector_env,
        n_steps=2,
        batch_size=2,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        seed=7,
        verbose=0,
    )
    model.save(run_root / "model")
    vector_env.close()


def test_dummy_sb3_policy_is_compared_with_validation_only_controls(tmp_path: Path) -> None:
    run_root = tmp_path / "stage-a"
    _write_run(run_root)
    snapshots = _snapshots()
    snapshot_path = tmp_path / "snapshots.jsonl"
    branch_path = tmp_path / "branches.csv"
    write_snapshots(snapshot_path, snapshots)
    _write_branch_matrix(branch_path, snapshots)

    result = analyze_snapshot_policy(
        run_root=run_root,
        checkpoint="final",
        snapshot_path=snapshot_path,
        branch_csv=branch_path,
        env_factory=_factory,
        validation_manifest_path=_VALIDATION_MANIFEST,
        forbidden_task_ids={"test/sealed"},
    )

    horizon = result["horizons"][0]
    assert result["outer_ppo_seed"] == 7
    assert result["n_snapshots"] == 2
    assert sum(result["deterministic_action_frequencies"].values()) == pytest.approx(1.0)
    assert sum(result["mean_stochastic_action_probabilities"].values()) == pytest.approx(1.0)
    assert horizon["values"]["best_validation_static"] == pytest.approx(1.5)
    assert horizon["values"]["dynamic_oracle"] == pytest.approx(3.0)
    assert horizon["dynamic_headroom"] == pytest.approx(1.5)
    assert set(horizon["captured_headroom"]) == {
        "learned_deterministic",
        "learned_stochastic_expected",
        "modal_static_clone",
        "marginal_frequency_matched_random",
        "best_validation_static",
        "dynamic_oracle",
    }
    json.dumps(result)


def test_vecnormalize_analysis_prepares_late_environment_spaces(tmp_path: Path) -> None:
    run_root = tmp_path / "normalized-stage-a"
    _write_run(run_root, vecnormalize=True)
    snapshots = _snapshots()
    snapshot_path = tmp_path / "snapshots.jsonl"
    branch_path = tmp_path / "branches.csv"
    write_snapshots(snapshot_path, snapshots)
    _write_branch_matrix(branch_path, snapshots)

    result = analyze_snapshot_policy(
        run_root=run_root,
        checkpoint="final",
        snapshot_path=snapshot_path,
        branch_csv=branch_path,
        env_factory=_late_factory,
        validation_manifest_path=_VALIDATION_MANIFEST,
        forbidden_task_ids=set(),
    )

    assert result["normalization_path"] == str(run_root / "vecnormalize.pkl")
    assert result["n_snapshots"] == 2


def test_validation_manifest_identity_and_hash_are_exact(tmp_path: Path) -> None:
    run_root = tmp_path / "stage-a"
    _write_run(run_root)
    snapshots = _snapshots(source_hash="a" * 64)
    snapshot_path = tmp_path / "snapshots.jsonl"
    write_snapshots(snapshot_path, snapshots)

    with pytest.raises(PermissionError, match="identity/hash"):
        analyze_snapshot_policy(
            run_root=run_root,
            checkpoint="final",
            snapshot_path=snapshot_path,
            branch_csv=tmp_path / "unused.csv",
            env_factory=_factory,
            validation_manifest_path=_VALIDATION_MANIFEST,
            forbidden_task_ids=set(),
        )


def test_checkpoint_and_snapshot_action_families_must_match(tmp_path: Path) -> None:
    run_root = tmp_path / "stage-a"
    _write_run(run_root)
    snapshots = _snapshots(action_space="lcb_quantile")
    snapshot_path = tmp_path / "snapshots.jsonl"
    write_snapshots(snapshot_path, snapshots)

    with pytest.raises(ValueError, match="checkpoint action family"):
        analyze_snapshot_policy(
            run_root=run_root,
            checkpoint="final",
            snapshot_path=snapshot_path,
            branch_csv=tmp_path / "unused.csv",
            env_factory=_factory,
            validation_manifest_path=_VALIDATION_MANIFEST,
            forbidden_task_ids=set(),
        )


def test_branch_matrix_must_match_complete_snapshot_provenance(tmp_path: Path) -> None:
    run_root = tmp_path / "stage-a"
    _write_run(run_root)
    snapshots = _snapshots()
    snapshot_path = tmp_path / "snapshots.jsonl"
    branch_path = tmp_path / "branches.csv"
    write_snapshots(snapshot_path, snapshots)
    _write_branch_matrix(branch_path, snapshots, tamper_hash=True)

    with pytest.raises(ValueError, match="snapshot hash"):
        analyze_snapshot_policy(
            run_root=run_root,
            checkpoint="final",
            snapshot_path=snapshot_path,
            branch_csv=branch_path,
            env_factory=_factory,
            validation_manifest_path=_VALIDATION_MANIFEST,
            forbidden_task_ids=set(),
        )


def test_analyzer_intrinsically_rejects_strict_test_before_factory_use(tmp_path: Path) -> None:
    run_root = tmp_path / "stage-a"
    _write_run(run_root)
    snapshot = BOSnapshot(
        "bbob/2/1/2",
        17,
        action_space="wei",
        source_manifest=_VALIDATION_ID,
        source_manifest_hash=_VALIDATION_HASH,
    )
    snapshot_path = tmp_path / "strict.jsonl"
    write_snapshots(snapshot_path, [snapshot])
    factory_calls: list[tuple[str, int, str]] = []

    def recording_factory(task_id: str, inner_seed: int, action_space_name: str) -> _PolicyEnv:
        factory_calls.append((task_id, inner_seed, action_space_name))
        return _PolicyEnv(task_id, inner_seed)

    with pytest.raises(PermissionError, match="sealed/test task IDs"):
        analyze_snapshot_policy(
            run_root=run_root,
            checkpoint="final",
            snapshot_path=snapshot_path,
            branch_csv=tmp_path / "unused.csv",
            env_factory=recording_factory,
            validation_manifest_path=_VALIDATION_MANIFEST,
            forbidden_task_ids=set(),
        )
    assert factory_calls == []
