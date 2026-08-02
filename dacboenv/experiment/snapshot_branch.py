"""Offline, common-random-number snapshot branching for BO action diagnostics.

The diagnostic deliberately depends on a caller-provided fixed-context environment
factory.  Reconstructing every branch from the same task, inner seed, and action
history makes comparisons between discrete actions paired without requiring a
general-purpose serializer for SMAC's internal state.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Protocol

import numpy as np

DEFAULT_BRANCH_HORIZONS = (1, 5, 10)
DEFAULT_REPLAY_TOLERANCE = 1e-12
MAX_INNER_SEED = int(np.iinfo(np.uint32).max)
_INSTANCE_TUPLE_SIZE = 2
_RESET_RESULT_SIZE = 2
_STEP_RESULT_SIZE = 5


class ReplayableBOEnv(Protocol):
    """Minimal environment interface required by snapshot branching."""

    action_space: Any

    def reset(self) -> tuple[Any, Mapping[str, Any]]:
        """Reset to the fixed context supplied to the environment factory."""

    def step(self, action: int) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
        """Apply one external action."""

    def get_incumbent_cost(self) -> float:
        """Return the current scalar incumbent cost (lower is better)."""

    def get_n_finished_trials(self) -> int:
        """Return the number of completed BO evaluations."""


@dataclass(frozen=True)
class BOSnapshot:
    """Replayable BO state identified by context and its external action prefix."""

    task_id: str
    inner_seed: int
    action_history: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Normalize and validate the portable snapshot representation."""
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("A snapshot task_id must be a non-empty string.")
        if not isinstance(self.inner_seed, Integral) or isinstance(self.inner_seed, bool):
            raise TypeError(f"A snapshot inner_seed must be an integer, got {self.inner_seed!r}.")
        if not 0 <= int(self.inner_seed) <= MAX_INNER_SEED:
            raise ValueError(f"A snapshot inner_seed must be in [0, {MAX_INNER_SEED}], got {self.inner_seed!r}.")

        history = tuple(self.action_history)
        if any(not isinstance(action, Integral) or isinstance(action, bool) for action in history):
            raise TypeError(f"Snapshot actions must be integers, got {history!r}.")

        object.__setattr__(self, "inner_seed", int(self.inner_seed))
        object.__setattr__(self, "action_history", tuple(int(action) for action in history))


@dataclass(frozen=True)
class SnapshotBranchResult:
    """Regret-improvement outcome for one snapshot, action, and BO horizon."""

    snapshot_index: int
    snapshot: BOSnapshot
    action: int
    horizon: int
    reference_value: float
    initial_incumbent: float
    final_incumbent: float
    initial_regret: float
    final_regret: float
    regret_improvement: float
    policy_steps: int
    terminated: bool
    truncated: bool
    reference_breach: float


@dataclass(frozen=True)
class HorizonBranchSummary:
    """Dynamic-oracle and best-static comparison at one BO horizon.

    ``best_action_frequencies`` awards fractional credit to tied actions, so
    the frequencies always sum to one.  The action gaps are measured from the
    per-snapshot dynamic best value.
    """

    horizon: int
    dynamic_oracle_value: float
    best_static_action: int
    best_static_value: float
    dynamic_headroom: float
    mean_value_by_action: dict[int, float]
    best_action_frequencies: dict[int, float]
    mean_gap_to_dynamic_best_by_action: dict[int, float]
    mean_top1_top2_gap: float


@dataclass(frozen=True)
class SnapshotBranchReport:
    """Complete outputs of a snapshot-and-branch diagnostic."""

    snapshots: tuple[BOSnapshot, ...]
    horizons: tuple[int, ...]
    actions: tuple[int, ...]
    action_count: int
    branches: tuple[SnapshotBranchResult, ...]
    summaries: tuple[HorizonBranchSummary, ...]

    def summary_for(self, horizon: int) -> HorizonBranchSummary:
        """Return the summary for one requested horizon."""
        for summary in self.summaries:
            if summary.horizon == horizon:
                return summary
        raise KeyError(f"Horizon {horizon} was not evaluated; available horizons are {self.horizons}.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable nested representation."""
        return asdict(self)


class SnapshotReplayError(RuntimeError):
    """Raised when a factory cannot deterministically reconstruct a snapshot."""


class ExactHorizonError(RuntimeError):
    """Raised when an environment cannot reach a requested BO horizon exactly."""


def _normalize_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for horizon in horizons:
        if not isinstance(horizon, Integral) or isinstance(horizon, bool):
            raise TypeError(f"BO horizons must be integers, got {horizon!r}.")
        value = int(horizon)
        if value <= 0:
            raise ValueError(f"BO horizons must be positive, got {value}.")
        normalized.append(value)
    if not normalized:
        raise ValueError("At least one BO horizon is required.")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"BO horizons must be unique, got {normalized!r}.")
    return tuple(sorted(normalized))


def _close_env(env: ReplayableBOEnv) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _validate_reset_context(env: ReplayableBOEnv, info: Mapping[str, Any], snapshot: BOSnapshot) -> None:
    task_id = info.get("task_id", getattr(env, "current_task_id", None))
    inner_seed = info.get("inner_seed", getattr(env, "current_seed", None))
    instance = getattr(env, "instance", None)
    if isinstance(instance, tuple) and len(instance) == _INSTANCE_TUPLE_SIZE:
        if task_id is None:
            task_id = instance[1]
        if inner_seed is None:
            inner_seed = instance[0]

    if task_id is None or inner_seed is None:
        raise SnapshotReplayError(
            "The fixed-context environment must expose task_id and inner_seed in reset info "
            "or through current_task_id/current_seed (or instance)."
        )
    if str(task_id) != snapshot.task_id or int(inner_seed) != snapshot.inner_seed:
        raise SnapshotReplayError(
            "Fixed-context factory returned the wrong context: "
            f"expected ({snapshot.task_id!r}, {snapshot.inner_seed}), got ({task_id!r}, {inner_seed!r})."
        )


def _discrete_actions(env: ReplayableBOEnv) -> tuple[int, ...]:
    n_actions = getattr(getattr(env, "action_space", None), "n", None)
    if not isinstance(n_actions, Integral) or isinstance(n_actions, bool) or int(n_actions) <= 0:
        raise TypeError("Snapshot branching requires an environment with a finite Discrete action_space.n.")
    start = getattr(env.action_space, "start", 0)
    if not isinstance(start, Integral) or isinstance(start, bool):
        raise TypeError(f"Discrete action_space.start must be an integer, got {start!r}.")
    return tuple(range(int(start), int(start) + int(n_actions)))


def _finished_trials(env: ReplayableBOEnv) -> int:
    count = env.get_n_finished_trials()
    if not isinstance(count, Integral) or isinstance(count, bool) or int(count) < 0:
        raise SnapshotReplayError(f"get_n_finished_trials() returned an invalid count: {count!r}.")
    return int(count)


def _incumbent_cost(env: ReplayableBOEnv) -> float:
    incumbent = float(env.get_incumbent_cost())
    if not np.isfinite(incumbent):
        raise SnapshotReplayError(f"get_incumbent_cost() returned a non-finite value: {incumbent!r}.")
    return incumbent


def _step(env: ReplayableBOEnv, action: int) -> tuple[bool, bool]:
    result = env.step(action)
    if not isinstance(result, tuple) or len(result) != _STEP_RESULT_SIZE:
        raise SnapshotReplayError("Environment step() must return the Gymnasium five-tuple.")
    _observation, _reward, terminated, truncated, _info = result
    return bool(terminated), bool(truncated)


def replay_snapshot(
    snapshot: BOSnapshot,
    env_factory: Callable[[str, int], ReplayableBOEnv],
) -> ReplayableBOEnv:
    """Rebuild and replay one snapshot, returning the live cloned environment.

    The caller owns the returned environment and should close it.  On replay
    failure this function closes the partially reconstructed environment.
    """
    env = env_factory(snapshot.task_id, snapshot.inner_seed)
    try:
        reset_result = env.reset()
        if not isinstance(reset_result, tuple) or len(reset_result) != _RESET_RESULT_SIZE:
            raise SnapshotReplayError("Environment reset() must return the Gymnasium (observation, info) tuple.")
        _observation, info = reset_result
        if not isinstance(info, Mapping):
            raise SnapshotReplayError("Environment reset() info must be a mapping.")
        _validate_reset_context(env, info, snapshot)

        actions = _discrete_actions(env)
        for history_index, action in enumerate(snapshot.action_history):
            if action not in actions:
                raise SnapshotReplayError(
                    f"Snapshot action {action} at history index {history_index} is not in {actions!r}."
                )
            before = _finished_trials(env)
            terminated, truncated = _step(env, action)
            after = _finished_trials(env)
            if after <= before:
                raise SnapshotReplayError(
                    f"Replay action at history index {history_index} completed no BO evaluation ({before} -> {after})."
                )
            if terminated or truncated:
                raise SnapshotReplayError(
                    f"Snapshot action history reaches a terminal state at index {history_index}; it cannot be branched."
                )
    except Exception:
        _close_env(env)
        raise
    return env


def _assert_replayed_state(
    *,
    snapshot: BOSnapshot,
    expected_actions: tuple[int, ...],
    expected_finished_trials: int,
    expected_incumbent: float,
    env: ReplayableBOEnv,
    replay_tolerance: float,
) -> None:
    actions = _discrete_actions(env)
    finished_trials = _finished_trials(env)
    incumbent = _incumbent_cost(env)
    if actions != expected_actions:
        raise SnapshotReplayError(
            f"Discrete actions changed while cloning {snapshot.task_id!r}: {expected_actions!r} != {actions!r}."
        )
    if finished_trials != expected_finished_trials:
        raise SnapshotReplayError(
            f"BO evaluation count changed while cloning {snapshot.task_id!r}: "
            f"{expected_finished_trials} != {finished_trials}."
        )
    if not np.isclose(incumbent, expected_incumbent, rtol=0.0, atol=replay_tolerance):
        raise SnapshotReplayError(
            f"Incumbent changed while replaying {snapshot.task_id!r}: {expected_incumbent} != {incumbent}."
        )


def _branch_one_action(
    *,
    snapshot_index: int,
    snapshot: BOSnapshot,
    action: int,
    horizons: tuple[int, ...],
    reference_value: float,
    initial_incumbent: float,
    initial_finished_trials: int,
    env: ReplayableBOEnv,
) -> list[SnapshotBranchResult]:
    initial_regret = max(initial_incumbent - reference_value, 0.0)
    results: list[SnapshotBranchResult] = []
    policy_steps = 0
    terminated = False
    truncated = False

    for horizon in horizons:
        target_finished_trials = initial_finished_trials + horizon
        while _finished_trials(env) < target_finished_trials:
            if terminated or truncated:
                raise ExactHorizonError(
                    f"Task {snapshot.task_id!r}, action {action} terminated before the requested horizon {horizon}."
                )
            before = _finished_trials(env)
            terminated, truncated = _step(env, action)
            policy_steps += 1
            after = _finished_trials(env)
            if after <= before:
                raise ExactHorizonError(
                    f"Task {snapshot.task_id!r}, action {action} completed no BO evaluation ({before} -> {after})."
                )
            if after > target_finished_trials:
                raise ExactHorizonError(
                    f"One external step overshot BO horizon {horizon} for task {snapshot.task_id!r}, action {action}: "
                    f"finished-trial count {before} -> {after}. Use an interaction frequency that reaches every "
                    "requested BO horizon exactly."
                )

        final_incumbent = _incumbent_cost(env)
        final_regret = max(final_incumbent - reference_value, 0.0)
        results.append(
            SnapshotBranchResult(
                snapshot_index=snapshot_index,
                snapshot=snapshot,
                action=action,
                horizon=horizon,
                reference_value=reference_value,
                initial_incumbent=initial_incumbent,
                final_incumbent=final_incumbent,
                initial_regret=initial_regret,
                final_regret=final_regret,
                regret_improvement=initial_regret - final_regret,
                policy_steps=policy_steps,
                terminated=terminated,
                truncated=truncated,
                reference_breach=max(reference_value - final_incumbent, 0.0),
            )
        )

    return results


def _summarize(
    branches: Sequence[SnapshotBranchResult],
    *,
    n_snapshots: int,
    actions: tuple[int, ...],
    horizons: tuple[int, ...],
    tie_tolerance: float,
) -> tuple[HorizonBranchSummary, ...]:
    outcomes = {
        (branch.horizon, branch.snapshot_index, branch.action): branch.regret_improvement for branch in branches
    }
    summaries: list[HorizonBranchSummary] = []
    for horizon in horizons:
        matrix = np.empty((n_snapshots, len(actions)), dtype=float)
        for snapshot_index in range(n_snapshots):
            for action_index, action in enumerate(actions):
                try:
                    matrix[snapshot_index, action_index] = outcomes[(horizon, snapshot_index, action)]
                except KeyError as error:
                    raise RuntimeError(
                        f"Missing branch result for horizon={horizon}, snapshot={snapshot_index}, action={action}."
                    ) from error

        dynamic_best = np.max(matrix, axis=1)
        dynamic_oracle_value = float(np.mean(dynamic_best))
        static_values = np.mean(matrix, axis=0)
        best_static_value = float(np.max(static_values))
        tied_static_actions = np.flatnonzero(np.isclose(static_values, best_static_value, rtol=0.0, atol=tie_tolerance))
        best_static_action = actions[int(tied_static_actions[0])]

        tied_dynamic_actions = np.isclose(matrix, dynamic_best[:, None], rtol=0.0, atol=tie_tolerance)
        fractional_winners = tied_dynamic_actions / np.sum(tied_dynamic_actions, axis=1, keepdims=True)
        best_action_frequencies = np.mean(fractional_winners, axis=0)
        gaps = dynamic_best[:, None] - matrix
        if len(actions) == 1:
            top1_top2_gaps = np.zeros(n_snapshots, dtype=float)
        else:
            ordered = np.sort(matrix, axis=1)
            top1_top2_gaps = ordered[:, -1] - ordered[:, -2]

        dynamic_headroom = dynamic_oracle_value - best_static_value
        if dynamic_headroom < 0 and abs(dynamic_headroom) <= tie_tolerance:
            dynamic_headroom = 0.0
        if dynamic_headroom < 0:
            raise RuntimeError(f"Computed negative dynamic headroom at horizon {horizon}: {dynamic_headroom}.")

        summaries.append(
            HorizonBranchSummary(
                horizon=horizon,
                dynamic_oracle_value=dynamic_oracle_value,
                best_static_action=best_static_action,
                best_static_value=best_static_value,
                dynamic_headroom=dynamic_headroom,
                mean_value_by_action={
                    action: float(static_values[action_index]) for action_index, action in enumerate(actions)
                },
                best_action_frequencies={
                    action: float(best_action_frequencies[action_index]) for action_index, action in enumerate(actions)
                },
                mean_gap_to_dynamic_best_by_action={
                    action: float(np.mean(gaps[:, action_index])) for action_index, action in enumerate(actions)
                },
                mean_top1_top2_gap=float(np.mean(top1_top2_gaps)),
            )
        )
    return tuple(summaries)


def run_snapshot_branch_diagnostic(
    snapshots: Sequence[BOSnapshot],
    env_factory: Callable[[str, int], ReplayableBOEnv],
    reference_value: Callable[[str], float],
    *,
    forbidden_task_ids: Collection[str],
    horizons: Sequence[int] = DEFAULT_BRANCH_HORIZONS,
    replay_tolerance: float = DEFAULT_REPLAY_TOLERANCE,
    tie_tolerance: float = DEFAULT_REPLAY_TOLERANCE,
) -> SnapshotBranchReport:
    """Replay snapshots and hold every discrete action for each BO horizon.

    Every action branch is reconstructed from the same ``task_id``,
    ``inner_seed``, and action prefix, providing common random numbers when the
    factory seeds all stochastic BO components from ``inner_seed``.  External
    environment steps may contain multiple BO evaluations only when they land
    exactly on every requested horizon; otherwise the diagnostic fails rather
    than mislabelling policy steps as BO steps.

    Parameters
    ----------
    snapshots : Sequence[BOSnapshot]
        Non-test BO states collected from training or validation contexts.
    env_factory : Callable[[str, int], ReplayableBOEnv]
        Factory pinned to the requested task and inner seed.
    reference_value : Callable[[str], float]
        Exact or best-known minimization reference for a task ID.
    forbidden_task_ids : Collection[str]
        Test or otherwise prohibited task IDs.  Membership is checked before
        creating any environment.
    horizons : Sequence[int]
        Positive, unique numbers of future BO evaluations.
    replay_tolerance : float
        Absolute tolerance when checking replayed incumbent equality.
    tie_tolerance : float
        Absolute tolerance used for tie-aware action summaries.

    Returns
    -------
    SnapshotBranchReport
        Per-branch regret improvements and dynamic-versus-static summaries.
    """
    frozen_snapshots = tuple(snapshots)
    if not frozen_snapshots:
        raise ValueError("At least one BO snapshot is required.")
    normalized_horizons = _normalize_horizons(horizons)
    if not np.isfinite(replay_tolerance) or replay_tolerance < 0:
        raise ValueError(f"replay_tolerance must be finite and non-negative, got {replay_tolerance!r}.")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0:
        raise ValueError(f"tie_tolerance must be finite and non-negative, got {tie_tolerance!r}.")

    forbidden = set(forbidden_task_ids)
    prohibited = sorted({snapshot.task_id for snapshot in frozen_snapshots} & forbidden)
    if prohibited:
        raise ValueError(f"Snapshot diagnostic refuses forbidden/test task IDs: {prohibited!r}.")

    all_branches: list[SnapshotBranchResult] = []
    expected_actions: tuple[int, ...] | None = None
    for snapshot_index, snapshot in enumerate(frozen_snapshots):
        task_reference = float(reference_value(snapshot.task_id))
        if not np.isfinite(task_reference):
            raise ValueError(
                f"Reference function returned a non-finite value for {snapshot.task_id!r}: {task_reference!r}."
            )

        baseline_env = replay_snapshot(snapshot, env_factory)
        try:
            actions = _discrete_actions(baseline_env)
            initial_finished_trials = _finished_trials(baseline_env)
            initial_incumbent = _incumbent_cost(baseline_env)
        finally:
            _close_env(baseline_env)

        if expected_actions is None:
            expected_actions = actions
        elif actions != expected_actions:
            raise ValueError(
                "Every snapshot must expose the same discrete actions: "
                f"expected {expected_actions!r}, got {actions!r} for {snapshot.task_id!r}."
            )

        for action in actions:
            branch_env = replay_snapshot(snapshot, env_factory)
            try:
                _assert_replayed_state(
                    snapshot=snapshot,
                    expected_actions=actions,
                    expected_finished_trials=initial_finished_trials,
                    expected_incumbent=initial_incumbent,
                    env=branch_env,
                    replay_tolerance=replay_tolerance,
                )
                all_branches.extend(
                    _branch_one_action(
                        snapshot_index=snapshot_index,
                        snapshot=snapshot,
                        action=action,
                        horizons=normalized_horizons,
                        reference_value=task_reference,
                        initial_incumbent=initial_incumbent,
                        initial_finished_trials=initial_finished_trials,
                        env=branch_env,
                    )
                )
            finally:
                _close_env(branch_env)

    assert expected_actions is not None
    summaries = _summarize(
        all_branches,
        n_snapshots=len(frozen_snapshots),
        actions=expected_actions,
        horizons=normalized_horizons,
        tie_tolerance=tie_tolerance,
    )
    return SnapshotBranchReport(
        snapshots=frozen_snapshots,
        horizons=normalized_horizons,
        actions=expected_actions,
        action_count=len(expected_actions),
        branches=tuple(all_branches),
        summaries=summaries,
    )
