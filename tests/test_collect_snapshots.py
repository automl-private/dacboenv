"""Portable snapshot collection, replay verification, and saved branching tests."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from dacboenv.experiment import collect_snapshots as collect_snapshots_module
from dacboenv.experiment.collect_snapshots import (
    DefaultSMACEquivalentSnapshotPolicy,
    InitialDesignOnlySnapshotPolicy,
    StaticSnapshotPolicy,
    collect_context_snapshots,
    main as collect_snapshots_main,
    read_snapshots,
    verify_portable_snapshot_replay,
    write_snapshots,
)
from dacboenv.experiment.run_snapshot_branches import run_saved_snapshot_branches
from dacboenv.experiment.snapshot_branch import (
    DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT,
    BOSnapshot,
    SnapshotReplayError,
)


@dataclass(frozen=True)
class _TrialKey:
    config_id: int
    seed: int
    budget: float | None = None


class _RunHistory:
    def __init__(self) -> None:
        self._data: dict[_TrialKey, Any] = {}
        self._configs: dict[int, dict[str, float]] = {}

    def add(self, *, cost: float, seed: int, action: int) -> None:
        config_id = len(self._data)
        key = _TrialKey(config_id=config_id, seed=seed)
        self._configs[config_id] = {"action": float(action), "trial": float(config_id)}
        self._data[key] = SimpleNamespace(cost=cost, status=SimpleNamespace(name="SUCCESS"))

    def get_config(self, config_id: int) -> dict[str, float]:
        return self._configs[config_id]


class _Discrete:
    n = 3
    start = 0

    @staticmethod
    def contains(action: object) -> bool:
        return isinstance(action, int) and not isinstance(action, bool) and 0 <= action < 3


class _PortableToyEnv:
    improvements: ClassVar[dict[str, tuple[float, ...]]] = {
        "train/a": (3.0, 1.0, 0.0),
        "train/b": (0.0, 1.0, 3.0),
    }

    def __init__(self, task_id: str, inner_seed: int, action_space_name: str = "wei") -> None:
        self.current_task_id = task_id
        self.current_seed = inner_seed
        self.action_space_name = action_space_name
        self.action_space = _Discrete()
        self.interaction_frequency = 1
        self.closed = False
        self._n_trials = 20
        self._smac_instance: Any = None
        self._incumbent = 20.0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "global_state": np.asarray(
                [self.get_n_finished_trials() / self._n_trials, self._incumbent],
                dtype=np.float32,
            ),
            "action_features": np.zeros((3, 2), dtype=np.float32),
        }

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        runhistory = _RunHistory()
        runhistory.add(cost=20.0, seed=self.current_seed, action=-1)
        self._incumbent = 20.0
        self._smac_instance = SimpleNamespace(
            runhistory=runhistory,
            intensifier=SimpleNamespace(
                config_selector=SimpleNamespace(_initial_design_configs=[object()]),
            ),
            _scenario=SimpleNamespace(n_trials=self._n_trials),
        )
        self._carps_solver = SimpleNamespace(
            task=SimpleNamespace(objective_function=SimpleNamespace(f_min=0.0)),
        )
        return self._observation(), {"task_id": self.current_task_id, "inner_seed": self.current_seed}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        improvements = self.improvements.get(self.current_task_id, (3.0, 1.0, 0.0))
        self._incumbent -= improvements[action]
        self._smac_instance.runhistory.add(cost=self._incumbent, seed=self.current_seed, action=action)
        return self._observation(), 0.0, False, False, {}

    def get_observation(self) -> dict[str, np.ndarray]:
        return self._observation()

    def get_n_finished_trials(self) -> int:
        return len(self._smac_instance.runhistory._data)

    def get_incumbent_cost(self) -> float:
        return self._incumbent

    def close(self) -> None:
        self.closed = True


class _PortableFactory:
    def __init__(self) -> None:
        self.created: list[_PortableToyEnv] = []
        self.requested_action_spaces: list[str] = []

    def __call__(self, task_id: str, inner_seed: int, action_space_name: str = "wei") -> _PortableToyEnv:
        self.requested_action_spaces.append(action_space_name)
        env = _PortableToyEnv(task_id, inner_seed, action_space_name)
        self.created.append(env)
        return env


def _reference() -> Any:
    return SimpleNamespace(
        kind="exact",
        value=0.0,
        source="toy-live-optimum",
        source_hash="sha256:toy",
        runtime_objective_transform="identity",
        reporting_objective_transform="identity",
        fidelity="not_applicable",
        tolerance=1e-12,
        benchmark_code_version="toy-code-1",
        benchmark_data_version="toy-data-1",
    )


def _collect(factory: _PortableFactory) -> tuple[BOSnapshot, ...]:
    return collect_context_snapshots(
        task_id="train/a",
        inner_seed=17,
        env_factory=factory,
        policy=StaticSnapshotPolicy(1),
        budget_fractions=(0.05, 0.15),
        action_space_name="wei",
        source_manifest="toy-train",
        source_manifest_hash="sha256:manifest",
        reference=_reference(),
        code_commit="0123456789abcdef",
    )


def test_portable_snapshot_jsonl_round_trip_and_replay_verification(tmp_path: Path) -> None:
    snapshots = _collect(_PortableFactory())
    path = tmp_path / "snapshots.jsonl"
    write_snapshots(path, snapshots)

    loaded = read_snapshots(path)
    assert loaded == snapshots
    assert loaded[0].action_history == ()
    assert loaded[1].action_history == (1, 1)
    assert loaded[0].reference_value == 0.0
    assert loaded[0].reference_fidelity_json == '"not_applicable"'
    for snapshot in loaded:
        verify_portable_snapshot_replay(snapshot, _PortableFactory())


def test_portable_replay_detects_changed_completed_evaluation(tmp_path: Path) -> None:
    snapshot = _collect(_PortableFactory())[1]
    row = json.loads(json.dumps(snapshot, default=lambda value: value.__dict__))
    row["completed_evaluations"][1]["cost"] += 1.0
    path = tmp_path / "changed.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    changed = read_snapshots(path)[0]
    with pytest.raises(SnapshotReplayError, match="Completed evaluation 1 changed"):
        verify_portable_snapshot_replay(changed, _PortableFactory())


def test_saved_snapshot_branch_runner_writes_tidy_and_grouped_outputs(tmp_path: Path) -> None:
    snapshots = _collect(_PortableFactory())
    snapshot_path = tmp_path / "snapshots.jsonl"
    csv_path = tmp_path / "branches.csv"
    summary_path = tmp_path / "summary.json"
    write_snapshots(snapshot_path, snapshots)

    report = run_saved_snapshot_branches(
        snapshot_path,
        env_factory=_PortableFactory(),
        reference_values={"train/a": 0.0},
        forbidden_task_ids={"test/sealed"},
        output_csv=csv_path,
        output_summary=summary_path,
        horizons=(1,),
    )

    assert len(report.branches) == 6
    assert csv_path.read_text(encoding="utf-8").count("\n") == 7
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["n_snapshots"] == 2
    assert summary["horizons"] == [1]
    assert summary["groupings"]["by_action_space"][0]["group"] == {"action_space": "wei"}


def test_saved_snapshot_branch_runner_can_use_portable_references(tmp_path: Path) -> None:
    snapshots = tuple(
        replace(snapshot, reference_value=0.0, reference_kind="exact") for snapshot in _collect(_PortableFactory())
    )
    snapshot_path = tmp_path / "snapshots.jsonl"
    write_snapshots(snapshot_path, snapshots)

    report = run_saved_snapshot_branches(
        snapshot_path,
        env_factory=_PortableFactory(),
        forbidden_task_ids=set(),
        output_csv=tmp_path / "branches.csv",
        output_summary=tmp_path / "summary.json",
        horizons=(1,),
    )

    assert {branch.reference_value for branch in report.branches} == {0.0}


def test_saved_runner_rejects_sealed_task_before_factory_use(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "sealed.jsonl"
    write_snapshots(snapshot_path, [BOSnapshot("test/sealed", 3)])
    factory = _PortableFactory()

    with pytest.raises(ValueError, match="forbidden/test task IDs"):
        run_saved_snapshot_branches(
            snapshot_path,
            env_factory=factory,
            reference_values={"test/sealed": 0.0},
            forbidden_task_ids={"test/sealed"},
            output_csv=tmp_path / "never.csv",
            output_summary=tmp_path / "never.json",
            horizons=(1,),
        )

    assert factory.created == []


def test_saved_runner_intrinsically_rejects_strict_task_with_empty_caller_guard(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "strict.jsonl"
    write_snapshots(
        snapshot_path,
        [BOSnapshot("bbob/2/1/2", 3, action_space="wei", reference_kind="exact", reference_value=0.0)],
    )
    factory = _PortableFactory()

    with pytest.raises(ValueError, match="forbidden/test task IDs"):
        run_saved_snapshot_branches(
            snapshot_path,
            env_factory=factory,
            forbidden_task_ids=set(),
            output_csv=tmp_path / "never.csv",
            output_summary=tmp_path / "never.json",
            horizons=(1,),
        )
    assert factory.created == []


def test_default_smac_history_is_available_only_for_ei_compatible_action_spaces() -> None:
    assert DefaultSMACEquivalentSnapshotPolicy("wei").action == 2
    assert DefaultSMACEquivalentSnapshotPolicy("af_selection").name == "default_smac_equivalent"
    with pytest.raises(ValueError, match="meaningful only"):
        DefaultSMACEquivalentSnapshotPolicy("lcb_quantile")


def test_initial_design_only_policy_never_fabricates_a_history_action() -> None:
    with pytest.raises(RuntimeError, match="already reached"):
        InitialDesignOnlySnapshotPolicy()(None, None)


def test_cli_collects_from_explicit_non_test_bbob_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    factory = _PortableFactory()
    monkeypatch.setattr(collect_snapshots_module, "_load_callable", lambda _specification: factory)
    repository = Path(__file__).parents[1]
    output = tmp_path / "cli-snapshots.jsonl"

    exit_code = collect_snapshots_main(
        [
            "--output",
            str(output),
            "--manifest",
            str(repository / "dacboenv/configs/instance_sets/bbob_train.yaml"),
            "--forbidden-task-ids",
            str(repository / "dacboenv/configs/instance_sets/bbob_test_strict.yaml"),
            "--factory",
            "ignored:factory",
            "--task-id",
            "bbob/2/3/0",
            "--inner-seed",
            "17",
            "--budget-fraction",
            "0.05",
            "--action-space",
            "wei",
            "--policy",
            "static",
            "--static-action",
            "1",
        ]
    )

    snapshots = read_snapshots(output)
    assert exit_code == 0
    assert len(snapshots) == 1
    assert snapshots[0].task_id == "bbob/2/3/0"
    assert snapshots[0].source_manifest == "bbob-train-v1"
    assert snapshots[0].reference_kind == "exact"
    assert snapshots[0].reference_value == 0.0
    assert factory.requested_action_spaces == ["wei"]


def test_cli_binds_and_records_the_requested_action_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    factory = _PortableFactory()
    monkeypatch.setattr(collect_snapshots_module, "_load_callable", lambda _specification: factory)
    repository = Path(__file__).parents[1]
    output = tmp_path / "lcb-snapshot.jsonl"

    assert (
        collect_snapshots_main(
            [
                "--output",
                str(output),
                "--manifest",
                str(repository / "dacboenv/configs/instance_sets/bbob_train.yaml"),
                "--forbidden-task-ids",
                str(repository / "dacboenv/configs/instance_sets/bbob_test_strict.yaml"),
                "--factory",
                "ignored:factory",
                "--task-id",
                "bbob/2/3/0",
                "--inner-seed",
                "17",
                "--budget-fraction",
                "0.05",
                "--action-space",
                "lcb_quantile",
                "--policy",
                "initial_design_only",
            ]
        )
        == 0
    )
    assert factory.requested_action_spaces == ["lcb_quantile"]
    assert read_snapshots(output)[0].action_space == "lcb_quantile"


@pytest.mark.parametrize("task_id", ["bbob/2/1/2", "yahpo/so/lcbench/167168/None"])
def test_collection_intrinsically_rejects_final_test_before_factory_use(task_id: str) -> None:
    factory = _PortableFactory()
    with pytest.raises(ValueError, match="intrinsically refuses"):
        collect_context_snapshots(
            task_id=task_id,
            inner_seed=17,
            env_factory=factory,
            policy=InitialDesignOnlySnapshotPolicy(),
            budget_fractions=(0.05,),
            action_space_name="wei",
            source_manifest="sealed",
            source_manifest_hash="a" * 64,
            reference=_reference(),
        )
    assert factory.created == []


def test_collection_rejects_factory_that_ignores_requested_action_family() -> None:
    created: list[_PortableToyEnv] = []

    def wrong_factory(task_id: str, inner_seed: int, action_space_name: str) -> _PortableToyEnv:  # noqa: ARG001
        env = _PortableToyEnv(task_id, inner_seed, "wei")
        created.append(env)
        return env

    with pytest.raises(SnapshotReplayError, match="action family"):
        collect_context_snapshots(
            task_id="train/a",
            inner_seed=17,
            env_factory=wrong_factory,
            policy=InitialDesignOnlySnapshotPolicy(),
            budget_fractions=(0.05,),
            action_space_name="lcb_quantile",
            source_manifest="toy-validation",
            source_manifest_hash="a" * 64,
            reference=_reference(),
        )
    assert len(created) == 1
    assert created[0].closed
