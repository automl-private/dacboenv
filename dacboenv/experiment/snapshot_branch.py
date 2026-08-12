"""Offline, common-random-number snapshot branching for BO action diagnostics.

The diagnostic deliberately depends on a caller-provided fixed-context environment
factory.  Reconstructing every branch from the same task, inner seed, and action
history makes comparisons between discrete actions paired without requiring a
general-purpose serializer for SMAC's internal state.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Protocol

import numpy as np

from dacboenv.env.action import PosteriorModeActionSpace, PosteriorQuantileActionSpace, WEIDiscreteActionSpace
from dacboenv.env.reward import normalized_reference_regret_potential
from dacboenv.experiment.protocol import sealed_final_test_task_ids

DEFAULT_BRANCH_HORIZONS = (1, 5, 10)
DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
DEFAULT_REPLAY_TOLERANCE = 1e-12
MAX_INNER_SEED = int(np.iinfo(np.uint32).max)
_INSTANCE_TUPLE_SIZE = 2
_RESET_RESULT_SIZE = 2
_STEP_RESULT_SIZE = 5
CANONICAL_ACTION_SPACE_NAMES = ("wei", "lcb_quantile", "ucb_quantile", "af_selection")


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
class CompletedBOEvaluation:
    """Portable summary of one completed BO trial.

    Configurations are canonical JSON rather than Python/SMAC objects, which
    keeps snapshot files language-neutral and reviewable.
    """

    trial: int
    configuration_json: str
    cost: float
    status: str = "SUCCESS"
    seed: int | None = None
    budget: float | None = None


@dataclass(frozen=True)
class BOSnapshot:
    """Replayable BO state identified by context and its external action prefix."""

    task_id: str
    inner_seed: int
    action_history: tuple[int, ...] = ()
    action_space: str = ""
    interaction_frequency: int = 1
    completed_evaluations: tuple[CompletedBOEvaluation, ...] = ()
    budget_fraction: float | None = None
    history_policy: str = "unspecified"
    outer_policy_seed: int | None = None
    source_manifest: str = ""
    source_manifest_hash: str = ""
    code_commit: str = ""
    observation_hash: str = ""
    incumbent: float | None = None
    initial_design_incumbent: float | None = None
    reference_kind: str = ""
    reference_value: float | None = None
    reference_source: str = ""
    reference_source_hash: str = ""
    reference_runtime_objective_transform: str = ""
    reference_reporting_objective_transform: str = ""
    reference_fidelity_json: str = ""
    reference_tolerance: float | None = None
    reference_benchmark_code_version: str = ""
    reference_benchmark_data_version: str = ""
    snapshot_id: str = ""
    domain: str = ""
    native_instance: str = ""
    scenario: str = ""
    dimension: int | None = None
    history_seed: int | None = None
    total_budget: int | None = None
    observation_json: str = ""
    initial_design_hash: str = ""
    deterministic_environment_json: str = ""

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
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
        if (
            not isinstance(self.interaction_frequency, Integral)
            or isinstance(self.interaction_frequency, bool)
            or int(self.interaction_frequency) <= 0
        ):
            raise ValueError("Snapshot interaction_frequency must be a positive integer.")
        object.__setattr__(self, "interaction_frequency", int(self.interaction_frequency))
        object.__setattr__(self, "completed_evaluations", tuple(self.completed_evaluations))
        if self.action_space and self.action_space not in CANONICAL_ACTION_SPACE_NAMES:
            raise ValueError(
                "Snapshot action_space must be empty or one of "
                f"{CANONICAL_ACTION_SPACE_NAMES!r}, got {self.action_space!r}."
            )
        if self.budget_fraction is not None and (
            not np.isfinite(self.budget_fraction) or not 0.0 <= float(self.budget_fraction) <= 1.0
        ):
            raise ValueError("Snapshot budget_fraction must be finite and in [0, 1].")
        for name in ("incumbent", "initial_design_incumbent", "reference_value", "reference_tolerance"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"Snapshot {name} must be finite when supplied.")
        if self.reference_tolerance is not None and self.reference_tolerance < 0:
            raise ValueError("Snapshot reference_tolerance must be non-negative when supplied.")
        if self.reference_kind not in {"", "exact", "best_known"}:
            raise ValueError("Snapshot reference_kind must be '', 'exact', or 'best_known'.")
        if self.dimension is not None and int(self.dimension) <= 0:
            raise ValueError("Snapshot dimension must be positive when supplied.")
        if self.total_budget is not None and int(self.total_budget) <= 0:
            raise ValueError("Snapshot total_budget must be positive when supplied.")


def snapshot_record_digest(snapshot: BOSnapshot) -> str:
    """Hash every portable snapshot field, including completed evaluations."""
    payload = json.dumps(
        asdict(snapshot),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def environment_action_space_name(env: ReplayableBOEnv) -> str:
    """Return the canonical structured action family installed in an environment."""
    declared = getattr(env, "action_space_name", None)
    if declared is not None:
        if declared not in CANONICAL_ACTION_SPACE_NAMES:
            raise SnapshotReplayError(f"Environment declares unsupported action family {declared!r}.")
        return str(declared)

    controller = getattr(env, "_action_space", None)
    if isinstance(controller, WEIDiscreteActionSpace):
        return "wei"
    if isinstance(controller, PosteriorQuantileActionSpace):
        bound_type = controller.bound_type.upper()
        if bound_type == "LCB":
            return "lcb_quantile"
        if bound_type == "UCB":
            return "ucb_quantile"
    if isinstance(controller, PosteriorModeActionSpace):
        return "af_selection"
    raise SnapshotReplayError(
        "Could not identify the structured action family installed by the snapshot environment factory."
    )


def assert_snapshot_action_space(snapshot: BOSnapshot, env: ReplayableBOEnv) -> None:
    """Fail if a replay factory silently installs a different action family."""
    if not snapshot.action_space:
        return
    actual = environment_action_space_name(env)
    if actual != snapshot.action_space:
        raise SnapshotReplayError(
            "Snapshot action family does not match its replay environment: "
            f"saved={snapshot.action_space!r}, actual={actual!r}."
        )


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
    initial_potential: float
    final_potential: float
    normalized_potential_improvement: float
    policy_steps: int
    terminated: bool
    truncated: bool
    reference_breach: float
    configuration_trace: tuple[str, ...]


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
    relative_dynamic_headroom: float
    normalized_dynamic_oracle_value: float
    normalized_best_static_action: int
    normalized_best_static_value: float
    normalized_dynamic_headroom: float
    normalized_relative_dynamic_headroom: float
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


def replay_process_environment() -> dict[str, str | None]:
    """Return process settings that materially affect deterministic SMAC replay."""
    return {name: os.environ.get(name) for name in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT}


def require_deterministic_replay_process_environment() -> None:
    """Fail before live replay when process-level determinism was not pinned."""
    actual = replay_process_environment()
    mismatched = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT.items()
        if actual[name] != expected
    }
    if mismatched:
        assignments = " ".join(f"{name}={value}" for name, value in DETERMINISTIC_REPLAY_PROCESS_ENVIRONMENT.items())
        raise RuntimeError(
            "Deterministic SMAC replay requires process settings fixed before Python starts. "
            f"Re-run as `{assignments} python ...`; mismatches={mismatched!r}."
        )


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
        assert_snapshot_action_space(snapshot, env)

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
    episode_initial_incumbent = (
        initial_incumbent if snapshot.initial_design_incumbent is None else snapshot.initial_design_incumbent
    )
    initial_potential = normalized_reference_regret_potential(
        initial_incumbent,
        reference_value,
        episode_initial_incumbent,
    )
    results: list[SnapshotBranchResult] = []
    policy_steps = 0
    terminated = False
    truncated = False

    def configuration_trace() -> tuple[str, ...]:
        smac = getattr(env, "_smac_instance", None)
        runhistory = getattr(smac, "runhistory", None)
        data = getattr(runhistory, "_data", None)
        get_config = getattr(runhistory, "get_config", None)
        if not isinstance(data, Mapping) or not callable(get_config):
            return ()
        configurations = []
        for trial_key in tuple(data)[initial_finished_trials:]:
            config = get_config(trial_key.config_id)
            configurations.append(json.dumps(dict(config), allow_nan=False, separators=(",", ":"), sort_keys=True))
        return tuple(configurations)

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
        final_potential = normalized_reference_regret_potential(
            final_incumbent,
            reference_value,
            episode_initial_incumbent,
        )
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
                initial_potential=initial_potential,
                final_potential=final_potential,
                normalized_potential_improvement=final_potential - initial_potential,
                policy_steps=policy_steps,
                terminated=terminated,
                truncated=truncated,
                reference_breach=max(reference_value - final_incumbent, 0.0),
                configuration_trace=configuration_trace(),
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
    normalized_outcomes = {
        (branch.horizon, branch.snapshot_index, branch.action): branch.normalized_potential_improvement
        for branch in branches
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

        normalized_matrix = np.empty_like(matrix)
        for snapshot_index in range(n_snapshots):
            for action_index, action in enumerate(actions):
                normalized_matrix[snapshot_index, action_index] = normalized_outcomes[(horizon, snapshot_index, action)]
        normalized_dynamic_best = np.max(normalized_matrix, axis=1)
        normalized_dynamic_oracle = float(np.mean(normalized_dynamic_best))
        normalized_static_values = np.mean(normalized_matrix, axis=0)
        normalized_best_static_value = float(np.max(normalized_static_values))
        normalized_static_ties = np.flatnonzero(
            np.isclose(normalized_static_values, normalized_best_static_value, rtol=0.0, atol=tie_tolerance)
        )
        normalized_headroom = max(normalized_dynamic_oracle - normalized_best_static_value, 0.0)
        relative_headroom = dynamic_headroom / max(abs(dynamic_oracle_value), np.finfo(float).eps)
        normalized_relative_headroom = normalized_headroom / max(abs(normalized_dynamic_oracle), np.finfo(float).eps)

        summaries.append(
            HorizonBranchSummary(
                horizon=horizon,
                dynamic_oracle_value=dynamic_oracle_value,
                best_static_action=best_static_action,
                best_static_value=best_static_value,
                dynamic_headroom=dynamic_headroom,
                relative_dynamic_headroom=float(relative_headroom),
                normalized_dynamic_oracle_value=normalized_dynamic_oracle,
                normalized_best_static_action=actions[int(normalized_static_ties[0])],
                normalized_best_static_value=normalized_best_static_value,
                normalized_dynamic_headroom=float(normalized_headroom),
                normalized_relative_dynamic_headroom=float(normalized_relative_headroom),
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

    forbidden = set(forbidden_task_ids) | set(sealed_final_test_task_ids())
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
