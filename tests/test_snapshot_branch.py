"""Toy contracts for offline BO snapshot replay and branching."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
import pytest
from dacboenv.experiment.snapshot_branch import (
    DEFAULT_BRANCH_HORIZONS,
    DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT,
    BOSnapshot,
    ExactHorizonError,
    SnapshotReplayError,
    replay_snapshot,
    require_deterministic_replay_process_environment,
    run_snapshot_branch_diagnostic,
)


@dataclass
class ToyDiscrete:
    """Small stand-in exposing the Gymnasium Discrete contract used here."""

    n: int = 3
    start: int = 0


class ToyReplayEnv:
    """Deterministic minimization environment with an auditable RNG trace."""

    IMPROVEMENTS: ClassVar[dict[str, tuple[float, ...]]] = {
        "train/a": (3.0, 1.0, 0.0),
        "train/b": (0.0, 1.0, 3.0),
    }

    def __init__(
        self,
        task_id: str,
        inner_seed: int,
        *,
        action_start: int = 0,
        bo_evaluations_per_step: int = 1,
        context_task_id: str | None = None,
    ) -> None:
        self.task_id = task_id
        self.inner_seed = inner_seed
        self.context_task_id = context_task_id or task_id
        self.action_space = ToyDiscrete(start=action_start)
        self.bo_evaluations_per_step = bo_evaluations_per_step
        self.noise_trace: list[int] = []
        self.closed = False
        self.reset_calls = 0
        self.finished_trials = 0
        self.incumbent = 20.0
        self.rng = np.random.default_rng(inner_seed)

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Recreate the fixed task context and RNG stream."""
        self.reset_calls += 1
        self.finished_trials = 0
        self.incumbent = 20.0
        self.noise_trace = []
        self.rng = np.random.default_rng(self.inner_seed)
        return {}, {"task_id": self.context_task_id, "inner_seed": self.inner_seed}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply one action and consume a seeded common random number."""
        action_index = action - self.action_space.start
        if not 0 <= action_index < self.action_space.n:
            raise ValueError(action)
        self.noise_trace.append(int(self.rng.integers(0, 2**31)))
        self.incumbent -= self.IMPROVEMENTS[self.task_id][action_index]
        self.finished_trials += self.bo_evaluations_per_step
        return {}, 0.0, False, False, {"bo_evaluations": self.finished_trials}

    def get_incumbent_cost(self) -> float:
        """Return the current minimization incumbent."""
        return self.incumbent

    def get_n_finished_trials(self) -> int:
        """Return the toy BO evaluation counter."""
        return self.finished_trials

    def close(self) -> None:
        """Record resource cleanup."""
        self.closed = True


class ToyFactory:
    """Record every independently reconstructed toy environment."""

    def __init__(
        self,
        *,
        action_start: int = 0,
        bo_evaluations_per_step: int = 1,
        wrong_context: bool = False,
    ) -> None:
        self.action_start = action_start
        self.bo_evaluations_per_step = bo_evaluations_per_step
        self.wrong_context = wrong_context
        self.created: list[ToyReplayEnv] = []

    def __call__(self, task_id: str, inner_seed: int) -> ToyReplayEnv:
        """Construct one independent fixed-context environment."""
        context_task_id = f"wrong/{task_id}" if self.wrong_context else task_id
        env = ToyReplayEnv(
            task_id,
            inner_seed,
            action_start=self.action_start,
            bo_evaluations_per_step=self.bo_evaluations_per_step,
            context_task_id=context_task_id,
        )
        self.created.append(env)
        return env


def _zero_reference(task_id: str) -> float:  # noqa: ARG001
    return 0.0


def test_live_replay_process_environment_is_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="process settings fixed before Python starts"):
        require_deterministic_replay_process_environment()

    for name, value in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    require_deterministic_replay_process_environment()


def test_snapshot_replay_reconstructs_action_prefix() -> None:
    factory = ToyFactory()
    snapshot = BOSnapshot("train/a", np.int64(23), [1, 0])  # type: ignore[arg-type]

    env = cast(ToyReplayEnv, replay_snapshot(snapshot, factory))
    try:
        assert snapshot.inner_seed == 23
        assert snapshot.action_history == (1, 0)
        assert env.reset_calls == 1
        assert env.get_n_finished_trials() == 2
        assert env.get_incumbent_cost() == pytest.approx(16.0)
    finally:
        env.close()


def test_branches_use_common_random_numbers_and_exact_bo_horizons() -> None:
    factory = ToyFactory()
    snapshot = BOSnapshot("train/a", 37, (1,))

    report = run_snapshot_branch_diagnostic(
        [snapshot],
        factory,
        _zero_reference,
        forbidden_task_ids={"test/held-out"},
        horizons=(1, 3),
    )

    assert len(factory.created) == 4  # one replay check plus one clone per action
    assert all(env.closed for env in factory.created)
    branch_rng_traces = [env.noise_trace for env in factory.created[1:]]
    assert branch_rng_traces[0] == branch_rng_traces[1] == branch_rng_traces[2]
    assert len(branch_rng_traces[0]) == len(snapshot.action_history) + 3

    assert report.horizons == (1, 3)
    assert report.action_count == 3
    assert {(result.action, result.horizon) for result in report.branches} == {
        (action, horizon) for action in range(3) for horizon in (1, 3)
    }
    assert {result.policy_steps for result in report.branches if result.horizon == 1} == {1}
    assert {result.policy_steps for result in report.branches if result.horizon == 3} == {3}

    replayed_report = run_snapshot_branch_diagnostic(
        [snapshot],
        ToyFactory(),
        _zero_reference,
        forbidden_task_ids={"test/held-out"},
        horizons=(1, 3),
    )
    assert replayed_report == report


def test_dynamic_oracle_static_value_frequencies_and_gaps() -> None:
    report = run_snapshot_branch_diagnostic(
        [BOSnapshot("train/a", 3), BOSnapshot("train/b", 5)],
        ToyFactory(),
        _zero_reference,
        forbidden_task_ids=set(),
        horizons=(2, 1),
    )

    one = report.summary_for(1)
    assert report.horizons == (1, 2)
    assert one.dynamic_oracle_value == pytest.approx(3.0)
    assert one.best_static_action == 0  # deterministic tie break between actions 0 and 2
    assert one.best_static_value == pytest.approx(1.5)
    assert one.dynamic_headroom == pytest.approx(1.5)
    assert one.relative_dynamic_headroom == pytest.approx(0.5)
    assert one.normalized_dynamic_headroom >= 0.0
    assert one.normalized_relative_dynamic_headroom >= 0.0
    assert one.mean_value_by_action == pytest.approx({0: 1.5, 1: 1.0, 2: 1.5})
    assert one.best_action_frequencies == pytest.approx({0: 0.5, 1: 0.0, 2: 0.5})
    assert one.mean_gap_to_dynamic_best_by_action == pytest.approx({0: 1.5, 1: 2.0, 2: 1.5})
    assert one.mean_top1_top2_gap == pytest.approx(2.0)

    two = report.summary_for(2)
    assert two.dynamic_oracle_value == pytest.approx(6.0)
    assert two.best_static_value == pytest.approx(3.0)
    assert two.dynamic_headroom == pytest.approx(3.0)
    json.dumps(report.to_dict())


def test_true_regret_improvement_clips_and_records_reference_breach() -> None:
    report = run_snapshot_branch_diagnostic(
        [BOSnapshot("train/a", 11)],
        ToyFactory(),
        lambda _task_id: 18.0,
        forbidden_task_ids=set(),
        horizons=(1,),
    )
    action_zero = next(result for result in report.branches if result.action == 0)

    assert action_zero.initial_regret == pytest.approx(2.0)
    assert action_zero.final_incumbent == pytest.approx(17.0)
    assert action_zero.final_regret == 0.0
    assert action_zero.regret_improvement == pytest.approx(2.0)
    assert action_zero.reference_breach == pytest.approx(1.0)


def test_nonzero_discrete_start_is_respected() -> None:
    report = run_snapshot_branch_diagnostic(
        [BOSnapshot("train/a", 13, (3,))],
        ToyFactory(action_start=2),
        _zero_reference,
        forbidden_task_ids=set(),
        horizons=(1,),
    )

    assert report.actions == (2, 3, 4)
    assert {result.action for result in report.branches} == {2, 3, 4}


def test_forbidden_task_is_rejected_before_factory_or_reference_use() -> None:
    factory = ToyFactory()
    reference_calls: list[str] = []

    def reference(task_id: str) -> float:
        reference_calls.append(task_id)
        return 0.0

    with pytest.raises(ValueError, match="forbidden/test task IDs"):
        run_snapshot_branch_diagnostic(
            [BOSnapshot("test/held-out", 7)],
            factory,
            reference,
            forbidden_task_ids={"test/held-out"},
        )

    assert factory.created == []
    assert reference_calls == []


def test_intrinsic_final_test_guard_does_not_depend_on_caller_list() -> None:
    factory = ToyFactory()
    reference_calls: list[str] = []

    def reference(task_id: str) -> float:
        reference_calls.append(task_id)
        return 0.0

    with pytest.raises(ValueError, match="forbidden/test task IDs"):
        run_snapshot_branch_diagnostic(
            [BOSnapshot("bbob/2/1/2", 7)],
            factory,
            reference,
            forbidden_task_ids=set(),
        )
    assert factory.created == []
    assert reference_calls == []


def test_wrong_factory_context_is_rejected_and_closed() -> None:
    factory = ToyFactory(wrong_context=True)

    with pytest.raises(SnapshotReplayError, match="wrong context"):
        run_snapshot_branch_diagnostic(
            [BOSnapshot("train/a", 7)],
            factory,
            _zero_reference,
            forbidden_task_ids=set(),
            horizons=(1,),
        )

    assert len(factory.created) == 1
    assert factory.created[0].closed


def test_multi_evaluation_step_cannot_be_mislabeled_as_shorter_bo_horizon() -> None:
    factory = ToyFactory(bo_evaluations_per_step=2)

    with pytest.raises(ExactHorizonError, match="overshot BO horizon 1"):
        run_snapshot_branch_diagnostic(
            [BOSnapshot("train/a", 7)],
            factory,
            _zero_reference,
            forbidden_task_ids=set(),
            horizons=(1,),
        )

    assert all(env.closed for env in factory.created)


def test_default_horizons_are_one_five_and_ten_bo_evaluations() -> None:
    assert DEFAULT_BRANCH_HORIZONS == (1, 5, 10)


def test_portable_snapshot_metadata_is_validated_and_serializable() -> None:
    snapshot = BOSnapshot(
        "train/a",
        19,
        (0, 1),
        action_space="wei",
        interaction_frequency=1,
        budget_fraction=0.5,
        history_policy="uniform_random",
        source_manifest="bbob-validation-v1",
        source_manifest_hash="a" * 64,
        code_commit="b" * 40,
        observation_hash="c" * 64,
        incumbent=4.0,
        initial_design_incumbent=10.0,
        reference_kind="exact",
    )

    assert json.loads(json.dumps(snapshot.__dict__))["history_policy"] == "uniform_random"
    with pytest.raises(ValueError, match="budget_fraction"):
        BOSnapshot("train/a", 19, budget_fraction=1.1)
    with pytest.raises(ValueError, match="reference_kind"):
        BOSnapshot("train/a", 19, reference_kind="optimum")
