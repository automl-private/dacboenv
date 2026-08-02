"""Non-destructive fidelity audit for structured action-feature candidates.

The Stage-A observation describes deterministic proxy candidates.  This module
compares those rows with the configuration returned by the next real SMAC
``ask()`` without evaluating that configuration.  Every probe is performed in
an independently reconstructed environment; the environment from which a
portable snapshot was collected is never mutated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from scipy.stats import rankdata

from dacboenv.env.action import (
    PosteriorModeActionSpace,
    PosteriorQuantileActionSpace,
    WEIDiscreteActionSpace,
    WEITempoRLActionSpace,
)
from dacboenv.env.observation import (
    ACTION_FEATURE_INDEX,
    ACTION_FEATURE_NAMES,
    AF_ACTION_FEATURE_INDEX,
    AF_ACTION_FEATURE_NAMES,
    calculate_candidate_semantic_descriptors,
    configuration_distance,
    selected_action_feature_candidates,
)
from dacboenv.experiment.protocol import sealed_final_test_task_ids
from dacboenv.experiment.snapshot_branch import (
    BOSnapshot,
    ReplayableBOEnv,
    replay_process_environment,
    replay_snapshot,
    require_deterministic_replay_process_environment,
)
from dacboenv.utils.posterior_decision import LOWER_CONFIDENCE_BOUND, POSTERIOR_MEAN

FIDELITY_SCHEMA_VERSION = 1
DESCRIPTOR_NAMES = (
    "standardized_improvement",
    "normalized_uncertainty",
    "novelty",
)
_BUDGET_PHASES = (0.25, 0.5, 0.75)
_REPLAY_TOLERANCE = 1e-12
_BBOB_TASK_PARTS = 4
_MATRIX_NDIM = 2
_MIN_CORRELATION_POINTS = 2


class ActionFeatureFidelityError(RuntimeError):
    """Raised when a replay clone cannot support a valid fidelity probe."""


@dataclass(frozen=True)
class FidelityPanelEntry:
    """One replayable state requested by a fidelity panel."""

    task_id: str
    inner_seed: int
    action_space: str
    history_policy: str
    action_history: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """Validate the entry using the portable snapshot contract."""
        snapshot = BOSnapshot(self.task_id, self.inner_seed, self.action_history)
        if not isinstance(self.action_space, str) or not self.action_space.strip():
            raise ValueError("A fidelity-panel action_space label must be non-empty.")
        if not isinstance(self.history_policy, str) or not self.history_policy.strip():
            raise ValueError("A fidelity-panel history_policy label must be non-empty.")
        object.__setattr__(self, "task_id", snapshot.task_id)
        object.__setattr__(self, "inner_seed", snapshot.inner_seed)
        object.__setattr__(self, "action_history", snapshot.action_history)

    @property
    def snapshot(self) -> BOSnapshot:
        """Return the minimal snapshot understood by the shared replay utility."""
        return BOSnapshot(self.task_id, self.inner_seed, self.action_history)


@dataclass(frozen=True)
class ActionFeatureFidelityRecord:
    """Tidy comparison for one replayed state and one discrete action."""

    schema_version: int
    snapshot_id: str
    observation_hash: str
    task_id: str
    inner_seed: int
    action_space: str
    action: int
    dimension: int
    bo_evaluations: int
    budget_fraction: float
    budget_phase: str
    history_policy: str
    action_history: tuple[int, ...]
    proxy_candidate: str
    actual_candidate: str
    exact_candidate_equality: bool
    mixed_space_distance: float
    proxy_control_identity: tuple[float, ...]
    actual_control_identity: tuple[float, ...]
    control_identity_equality: bool
    proxy_standardized_improvement: float
    actual_standardized_improvement: float
    proxy_normalized_uncertainty: float
    actual_normalized_uncertainty: float
    proxy_novelty: float
    actual_novelty: float
    standardized_improvement_absolute_error: float
    normalized_uncertainty_absolute_error: float
    novelty_absolute_error: float
    proxy_candidate_duplicate_count: int
    actual_candidate_duplicate_count: int


@dataclass(frozen=True)
class _ProxyReplayState:
    feature_rows: np.ndarray
    candidates: tuple[Any, ...]
    n_actions: int
    finished: int
    incumbent: float
    evaluations: int
    budget_fraction: float
    budget_phase: str
    observation_hash: str
    snapshot_id: str


@dataclass(frozen=True)
class _ActualCandidateProbe:
    candidate: Any
    identity: tuple[float, ...]
    descriptors: dict[str, float]
    distance: float


FidelityEnvironmentFactory = Callable[[str, int, str], ReplayableBOEnv]


def _close_env(env: ReplayableBOEnv) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def _finished_trials(env: ReplayableBOEnv) -> int:
    value = env.get_n_finished_trials()
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise ActionFeatureFidelityError(f"Invalid completed-evaluation count {value!r}.")
    return int(value)


def _incumbent(env: ReplayableBOEnv) -> float:
    value = float(env.get_incumbent_cost())
    if not np.isfinite(value):
        raise ActionFeatureFidelityError(f"Replay clone has non-finite incumbent {value!r}.")
    return value


def _current_observation(env: ReplayableBOEnv) -> dict[str, np.ndarray]:
    get_observation = getattr(env, "get_observation", None)
    if not callable(get_observation):
        raise ActionFeatureFidelityError("Fidelity inspection requires env.get_observation().")
    observation = get_observation()
    if not isinstance(observation, Mapping) or "action_features" not in observation:
        raise ActionFeatureFidelityError("Replay clone did not expose an action_features observation.")
    return {str(key): np.asarray(value).copy() for key, value in observation.items()}


def _observation_hash(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        value = np.ascontiguousarray(np.asarray(observation[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def canonical_candidate(candidate: Any) -> str:
    """Return a stable JSON representation of an active configuration."""
    try:
        values = dict(candidate)
    except (TypeError, ValueError) as error:
        raise ActionFeatureFidelityError(f"Could not serialize candidate {candidate!r}.") from error
    return json.dumps(_json_value(values), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _dimension(task_id: str) -> int:
    parts = task_id.split("/")
    if len(parts) != _BBOB_TASK_PARTS or parts[0].lower() != "bbob":
        raise ValueError(f"Action-feature fidelity panels are restricted to canonical BBOB task IDs, got {task_id!r}.")
    try:
        dimension = int(parts[1])
    except ValueError as error:
        raise ValueError(f"Invalid BBOB dimension in task ID {task_id!r}.") from error
    if dimension <= 0:
        raise ValueError(f"Invalid BBOB dimension in task ID {task_id!r}.")
    return dimension


def _budget_metadata(env: ReplayableBOEnv) -> tuple[int, float, str]:
    evaluations = _finished_trials(env)
    scenario = getattr(getattr(env, "_smac_instance", None), "_scenario", None)
    budget = getattr(scenario, "n_trials", None)
    if not isinstance(budget, Integral) or isinstance(budget, bool) or int(budget) <= 0:
        raise ActionFeatureFidelityError("Replay clone does not expose a positive SMAC scenario n_trials.")
    fraction = evaluations / int(budget)
    nearest = min(_BUDGET_PHASES, key=lambda phase: abs(fraction - phase))
    return evaluations, float(fraction), f"{int(nearest * 100)}%"


def _snapshot_id(entry: FidelityPanelEntry, observation_hash: str, evaluations: int) -> str:
    payload = {
        "task_id": entry.task_id,
        "inner_seed": entry.inner_seed,
        "action_space": entry.action_space,
        "history_policy": entry.history_policy,
        "action_history": entry.action_history,
        "bo_evaluations": evaluations,
        "observation_hash": observation_hash,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _feature_parts(row: np.ndarray) -> tuple[tuple[float, ...], dict[str, float]]:
    values = np.asarray(row, dtype=float).reshape(-1)
    if values.size == len(ACTION_FEATURE_NAMES):
        identity = (float(values[ACTION_FEATURE_INDEX["control_value"]]),)
        indices = ACTION_FEATURE_INDEX
    elif values.size == len(AF_ACTION_FEATURE_NAMES):
        identity = tuple(float(value) for value in values[: len(AF_ACTION_FEATURE_NAMES) - 4])
        indices = AF_ACTION_FEATURE_INDEX
    else:
        raise ActionFeatureFidelityError(
            f"Expected a four-column or ten-column action feature row, got shape {values.shape}."
        )
    descriptors = {name: float(values[indices[name]]) for name in DESCRIPTOR_NAMES}
    if not np.isfinite([*identity, *descriptors.values()]).all():
        raise ActionFeatureFidelityError("Proxy action-feature rows must be finite.")
    return identity, descriptors


def _actual_control_identity(
    env: ReplayableBOEnv,
    action: int,
    proxy_identity: tuple[float, ...],
) -> tuple[float, ...]:
    action_controller = getattr(env, "_action_space", None)
    if isinstance(action_controller, WEIDiscreteActionSpace):
        acquisition = env._smac_instance.intensifier.config_selector._acquisition_function  # type: ignore[attr-defined]
        return (float(acquisition._alpha),)
    if isinstance(action_controller, PosteriorQuantileActionSpace):
        return (float(action_controller.current_control_value),)
    if isinstance(action_controller, PosteriorModeActionSpace):
        identity = np.zeros(len(AF_ACTION_FEATURE_NAMES) - 4, dtype=float)
        identity[action] = 1.0
        acquisition = env._smac_instance.intensifier.config_selector._acquisition_function  # type: ignore[attr-defined]
        if action_controller.selected_mode == POSTERIOR_MEAN:
            identity[-1] = 0.5
        elif action_controller.selected_mode == LOWER_CONFIDENCE_BOUND:
            identity[-1] = float(acquisition.lower_quantile)
        return tuple(identity.tolist())
    return proxy_identity


def _uses_wei_xi(env: ReplayableBOEnv, label: str) -> bool:
    controller = getattr(env, "_action_space", None)
    return isinstance(controller, WEIDiscreteActionSpace | WEITempoRLActionSpace) or "wei" in label.lower()


def _validate_context(entry: FidelityPanelEntry, forbidden_task_ids: Collection[str]) -> int:
    dimension = _dimension(entry.task_id)
    if entry.task_id in set(forbidden_task_ids) | set(sealed_final_test_task_ids()):
        raise ValueError(f"Action-feature fidelity refuses forbidden/test task ID {entry.task_id!r}.")
    return dimension


def _capture_proxy_state(
    entry: FidelityPanelEntry,
    fixed_factory: Callable[[str, int], ReplayableBOEnv],
) -> _ProxyReplayState:
    """Replay one disposable clone and copy its Stage-A proxy state."""
    proxy_env = replay_snapshot(entry.snapshot, fixed_factory)
    try:
        proxy_observation = _current_observation(proxy_env)
        feature_rows = np.asarray(proxy_observation["action_features"], dtype=float)
        n_actions = getattr(getattr(proxy_env, "action_space", None), "n", None)
        if not isinstance(n_actions, Integral) or isinstance(n_actions, bool) or int(n_actions) <= 0:
            raise ActionFeatureFidelityError("Fidelity inspection requires a finite Discrete action space.")
        if feature_rows.ndim != _MATRIX_NDIM or feature_rows.shape[0] != int(n_actions):
            raise ActionFeatureFidelityError(
                f"The action_features row count must equal action_space.n, got {feature_rows.shape} and {n_actions}."
            )
        candidates = selected_action_feature_candidates(proxy_env._smac_instance)  # type: ignore[attr-defined]
        if len(candidates) != int(n_actions) or any(candidate is None for candidate in candidates):
            raise ActionFeatureFidelityError("Every inspected action row must have a fitted, selected proxy candidate.")
        finished = _finished_trials(proxy_env)
        incumbent = _incumbent(proxy_env)
        evaluations, budget_fraction, budget_phase = _budget_metadata(proxy_env)
        observation_hash = _observation_hash(proxy_observation)
        snapshot_id = _snapshot_id(entry, observation_hash, evaluations)
        return _ProxyReplayState(
            feature_rows=feature_rows.copy(),
            candidates=tuple(candidate for candidate in candidates if candidate is not None),
            n_actions=int(n_actions),
            finished=finished,
            incumbent=incumbent,
            evaluations=evaluations,
            budget_fraction=budget_fraction,
            budget_phase=budget_phase,
            observation_hash=observation_hash,
            snapshot_id=snapshot_id,
        )
    finally:
        _close_env(proxy_env)


def _inspect_actual_candidate(
    *,
    entry: FidelityPanelEntry,
    fixed_factory: Callable[[str, int], ReplayableBOEnv],
    proxy_state: _ProxyReplayState,
    proxy_candidate: Any,
    proxy_identity: tuple[float, ...],
    action: int,
    replay_tolerance: float,
) -> _ActualCandidateProbe:
    """Call real ``update_optimizer``/``ask`` on one disposable replay clone."""
    actual_env = replay_snapshot(entry.snapshot, fixed_factory)
    try:
        actual_observation = _current_observation(actual_env)
        if _observation_hash(actual_observation) != proxy_state.observation_hash:
            raise ActionFeatureFidelityError(
                "Independent replay clones produced different observations before candidate inspection."
            )
        if _finished_trials(actual_env) != proxy_state.finished or not np.isclose(
            _incumbent(actual_env),
            proxy_state.incumbent,
            rtol=0.0,
            atol=replay_tolerance,
        ):
            raise ActionFeatureFidelityError("Independent replay clone does not match the proxy clone's BO state.")

        update_optimizer = getattr(actual_env, "update_optimizer", None)
        if not callable(update_optimizer):
            raise ActionFeatureFidelityError("Fidelity inspection requires env.update_optimizer(action).")
        update_optimizer(action)
        actual_identity = _actual_control_identity(actual_env, action, proxy_identity)

        smbo = getattr(actual_env, "_smac_instance", None)
        ask = getattr(smbo, "ask", None)
        if not callable(ask):
            raise ActionFeatureFidelityError("Fidelity inspection requires env._smac_instance.ask().")
        completed_before_ask = tuple(smbo.runhistory.get_configs())
        finished_before_ask = _finished_trials(actual_env)
        incumbent_before_ask = _incumbent(actual_env)
        trial_info = ask()
        if _finished_trials(actual_env) != finished_before_ask:
            raise ActionFeatureFidelityError(
                "SMAC ask() unexpectedly completed an objective evaluation during candidate inspection."
            )
        if not np.isclose(
            _incumbent(actual_env),
            incumbent_before_ask,
            rtol=0.0,
            atol=replay_tolerance,
        ):
            raise ActionFeatureFidelityError("SMAC ask() unexpectedly changed the incumbent.")
        actual_candidate = getattr(trial_info, "config", None)
        if actual_candidate is None:
            raise ActionFeatureFidelityError("SMAC ask() returned TrialInfo without a configuration.")
        descriptors = calculate_candidate_semantic_descriptors(
            smbo,
            actual_candidate,
            include_xi=_uses_wei_xi(actual_env, entry.action_space),
            evaluated_configs=completed_before_ask,
        )
        hyperparameters = list(smbo._scenario.configspace.values())
        distance = configuration_distance(proxy_candidate, actual_candidate, hyperparameters)
        return _ActualCandidateProbe(
            candidate=actual_candidate,
            identity=actual_identity,
            descriptors=descriptors,
            distance=distance,
        )
    finally:
        _close_env(actual_env)


def inspect_action_feature_fidelity(
    entry: FidelityPanelEntry,
    env_factory: FidelityEnvironmentFactory,
    *,
    forbidden_task_ids: Collection[str],
    replay_tolerance: float = _REPLAY_TOLERANCE,
) -> tuple[ActionFeatureFidelityRecord, ...]:
    """Inspect every discrete action from one portable replay state.

    ``env_factory`` is invoked as ``factory(task_id, inner_seed,
    action_space_label)``.  One clone captures the existing proxy observation;
    each action then receives its own fresh replay clone.  Calling SMAC
    ``ask()`` mutates only that disposable clone and the objective runner is
    never invoked.
    """
    dimension = _validate_context(entry, forbidden_task_ids)
    if not np.isfinite(replay_tolerance) or replay_tolerance < 0:
        raise ValueError("replay_tolerance must be finite and non-negative.")

    fixed_factory = lambda task_id, inner_seed: env_factory(task_id, inner_seed, entry.action_space)
    proxy_state = _capture_proxy_state(entry, fixed_factory)

    unfinished_records: list[ActionFeatureFidelityRecord] = []
    proxy_serialized: list[str] = []
    actual_serialized: list[str] = []
    for action in range(proxy_state.n_actions):
        proxy_candidate = proxy_state.candidates[action]
        proxy_identity, proxy_descriptors = _feature_parts(proxy_state.feature_rows[action])
        actual = _inspect_actual_candidate(
            entry=entry,
            fixed_factory=fixed_factory,
            proxy_state=proxy_state,
            proxy_candidate=proxy_candidate,
            proxy_identity=proxy_identity,
            action=action,
            replay_tolerance=replay_tolerance,
        )

        proxy_json = canonical_candidate(proxy_candidate)
        actual_json = canonical_candidate(actual.candidate)
        proxy_serialized.append(proxy_json)
        actual_serialized.append(actual_json)
        unfinished_records.append(
            ActionFeatureFidelityRecord(
                schema_version=FIDELITY_SCHEMA_VERSION,
                snapshot_id=proxy_state.snapshot_id,
                observation_hash=proxy_state.observation_hash,
                task_id=entry.task_id,
                inner_seed=entry.inner_seed,
                action_space=entry.action_space,
                action=action,
                dimension=dimension,
                bo_evaluations=proxy_state.evaluations,
                budget_fraction=proxy_state.budget_fraction,
                budget_phase=proxy_state.budget_phase,
                history_policy=entry.history_policy,
                action_history=entry.action_history,
                proxy_candidate=proxy_json,
                actual_candidate=actual_json,
                exact_candidate_equality=proxy_json == actual_json,
                mixed_space_distance=float(actual.distance),
                proxy_control_identity=proxy_identity,
                actual_control_identity=actual.identity,
                control_identity_equality=bool(
                    len(proxy_identity) == len(actual.identity)
                    and np.allclose(proxy_identity, actual.identity, rtol=0.0, atol=1e-7)
                ),
                proxy_standardized_improvement=proxy_descriptors["standardized_improvement"],
                actual_standardized_improvement=actual.descriptors["standardized_improvement"],
                proxy_normalized_uncertainty=proxy_descriptors["normalized_uncertainty"],
                actual_normalized_uncertainty=actual.descriptors["normalized_uncertainty"],
                proxy_novelty=proxy_descriptors["novelty"],
                actual_novelty=actual.descriptors["novelty"],
                standardized_improvement_absolute_error=abs(
                    proxy_descriptors["standardized_improvement"] - actual.descriptors["standardized_improvement"]
                ),
                normalized_uncertainty_absolute_error=abs(
                    proxy_descriptors["normalized_uncertainty"] - actual.descriptors["normalized_uncertainty"]
                ),
                novelty_absolute_error=abs(proxy_descriptors["novelty"] - actual.descriptors["novelty"]),
                proxy_candidate_duplicate_count=0,
                actual_candidate_duplicate_count=0,
            )
        )

    proxy_counts = Counter(proxy_serialized)
    actual_counts = Counter(actual_serialized)
    return tuple(
        replace(
            record,
            proxy_candidate_duplicate_count=proxy_counts[record.proxy_candidate] - 1,
            actual_candidate_duplicate_count=actual_counts[record.actual_candidate] - 1,
        )
        for record in unfinished_records
    )


def run_fidelity_panel(
    entries: Sequence[FidelityPanelEntry],
    env_factory: FidelityEnvironmentFactory,
    *,
    forbidden_task_ids: Collection[str],
) -> tuple[ActionFeatureFidelityRecord, ...]:
    """Run a configurable panel, refusing forbidden IDs before each factory use."""
    if not entries:
        raise ValueError("A fidelity panel must contain at least one entry.")
    records: list[ActionFeatureFidelityRecord] = []
    for entry in entries:
        records.extend(
            inspect_action_feature_fidelity(
                entry,
                env_factory,
                forbidden_task_ids=forbidden_task_ids,
            )
        )
    return tuple(records)


def _correlation(left: Sequence[float], right: Sequence[float], *, rank: bool) -> float | None:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < _MIN_CORRELATION_POINTS or np.ptp(x) <= np.finfo(float).eps or np.ptp(y) <= np.finfo(float).eps:
        return None
    if rank:
        x = rankdata(x, method="average")
        y = rankdata(y, method="average")
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def _pairwise_order_agreement(
    proxy: np.ndarray,
    actual: np.ndarray,
    tolerance: float = 1e-12,
) -> float | None:
    agreements: list[float] = []
    for left in range(len(proxy)):
        for right in range(left + 1, len(proxy)):
            proxy_delta = proxy[left] - proxy[right]
            actual_delta = actual[left] - actual[right]
            proxy_sign = 0 if abs(proxy_delta) <= tolerance else int(np.sign(proxy_delta))
            actual_sign = 0 if abs(actual_delta) <= tolerance else int(np.sign(actual_delta))
            agreements.append(float(proxy_sign == actual_sign))
    return float(np.mean(agreements)) if agreements else None


def _record_descriptor(record: ActionFeatureFidelityRecord, side: str, descriptor: str) -> float:
    return float(getattr(record, f"{side}_{descriptor}"))


def _snapshot_order_metrics(records: Sequence[ActionFeatureFidelityRecord]) -> list[dict[str, Any]]:
    snapshots: dict[str, list[ActionFeatureFidelityRecord]] = defaultdict(list)
    for record in records:
        snapshots[record.snapshot_id].append(record)

    metrics: list[dict[str, Any]] = []
    for snapshot_id, snapshot_records in sorted(snapshots.items()):
        ordered = sorted(snapshot_records, key=lambda record: record.action)
        first = ordered[0]
        row: dict[str, Any] = {
            "snapshot_id": snapshot_id,
            "task_id": first.task_id,
            "inner_seed": first.inner_seed,
            "action_space": first.action_space,
            "dimension": first.dimension,
            "budget_phase": first.budget_phase,
            "history_policy": first.history_policy,
        }
        for descriptor in DESCRIPTOR_NAMES:
            proxy = np.asarray([_record_descriptor(record, "proxy", descriptor) for record in ordered])
            actual = np.asarray([_record_descriptor(record, "actual", descriptor) for record in ordered])
            row[f"{descriptor}_spearman"] = _correlation(proxy, actual, rank=True)
            row[f"{descriptor}_action_order_agreement"] = _pairwise_order_agreement(proxy, actual)
            row[f"{descriptor}_top_action_agreement"] = (
                bool(int(np.argmax(proxy)) == int(np.argmax(actual))) if len(ordered) > 1 else None
            )

        action_indices = np.asarray([record.action for record in ordered], dtype=float)
        actual_uncertainty = np.asarray([record.actual_normalized_uncertainty for record in ordered])
        action_label = first.action_space.lower()
        expected_direction: float | None = None
        if "lcb" in action_label:
            expected_direction = 1.0
        elif "wei" in action_label or "ucb" in action_label:
            expected_direction = -1.0
        ordering = _correlation(action_indices, actual_uncertainty, rank=True)
        row["nominal_exploration_ordering"] = (
            None if ordering is None or expected_direction is None else expected_direction * ordering
        )
        row["proxy_unique_candidate_count"] = len({record.proxy_candidate for record in ordered})
        row["actual_unique_candidate_count"] = len({record.actual_candidate for record in ordered})
        metrics.append(row)
    return metrics


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def _median_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _group_summary(records: Sequence[ActionFeatureFidelityRecord]) -> dict[str, Any]:
    distances = np.asarray([record.mixed_space_distance for record in records], dtype=float)
    snapshot_metrics = _snapshot_order_metrics(records)
    summary: dict[str, Any] = {
        "n_records": len(records),
        "n_snapshots": len({record.snapshot_id for record in records}),
        "exact_candidate_equality_rate": float(np.mean([record.exact_candidate_equality for record in records])),
        "control_identity_equality_rate": float(np.mean([record.control_identity_equality for record in records])),
        "median_candidate_distance": float(np.median(distances)),
        "p90_candidate_distance": float(np.quantile(distances, 0.9)),
        "proxy_unique_candidate_count": len({record.proxy_candidate for record in records}),
        "actual_unique_candidate_count": len({record.actual_candidate for record in records}),
        "proxy_duplicate_candidate_fraction": float(
            np.mean([record.proxy_candidate_duplicate_count > 0 for record in records])
        ),
        "actual_duplicate_candidate_fraction": float(
            np.mean([record.actual_candidate_duplicate_count > 0 for record in records])
        ),
        "median_proxy_unique_candidates_per_snapshot": _median_optional(
            [metric["proxy_unique_candidate_count"] for metric in snapshot_metrics]
        ),
        "median_actual_unique_candidates_per_snapshot": _median_optional(
            [metric["actual_unique_candidate_count"] for metric in snapshot_metrics]
        ),
    }
    for descriptor in DESCRIPTOR_NAMES:
        proxy = [_record_descriptor(record, "proxy", descriptor) for record in records]
        actual = [_record_descriptor(record, "actual", descriptor) for record in records]
        summary[f"{descriptor}_pearson"] = _correlation(proxy, actual, rank=False)
        summary[f"{descriptor}_spearman"] = _correlation(proxy, actual, rank=True)
        summary[f"{descriptor}_median_absolute_error"] = float(
            np.median(np.abs(np.asarray(proxy) - np.asarray(actual)))
        )
        summary[f"{descriptor}_mean_action_order_agreement"] = _mean_optional(
            [metric[f"{descriptor}_action_order_agreement"] for metric in snapshot_metrics]
        )
        summary[f"{descriptor}_top_action_agreement_rate"] = _mean_optional(
            [
                None
                if metric[f"{descriptor}_top_action_agreement"] is None
                else float(metric[f"{descriptor}_top_action_agreement"])
                for metric in snapshot_metrics
            ]
        )
    summary["median_nominal_exploration_ordering"] = _median_optional(
        [metric["nominal_exploration_ordering"] for metric in snapshot_metrics]
    )
    return summary


def _classification(rank_correlation: float | None) -> str:
    if rank_correlation is None:
        return "unavailable"
    if rank_correlation >= 0.7:  # noqa: PLR2004
        return "strong"
    if rank_correlation >= 0.5:  # noqa: PLR2004
        return "usable but imperfect"
    return "weak"


def _summaries_for_grouping(
    records: Sequence[ActionFeatureFidelityRecord],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[ActionFeatureFidelityRecord]] = defaultdict(list)
    for record in records:
        grouped[tuple(getattr(record, field) for field in fields)].append(record)
    output: list[dict[str, Any]] = []
    for key, group_records in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        output.append(
            {
                "group": dict(zip(fields, key, strict=True)),
                "metrics": _group_summary(group_records),
            }
        )
    return output


def summarize_action_feature_fidelity(records: Sequence[ActionFeatureFidelityRecord]) -> dict[str, Any]:
    """Return correlations, action ordering, duplicates, and fidelity class."""
    if not records:
        raise ValueError("At least one fidelity record is required.")
    snapshot_metrics = _snapshot_order_metrics(records)
    descriptor_medians = {
        descriptor: _median_optional([metric[f"{descriptor}_spearman"] for metric in snapshot_metrics])
        for descriptor in DESCRIPTOR_NAMES
    }
    median_rank = _median_optional(list(descriptor_medians.values()))
    grouping_fields = {
        "by_action_space": ("action_space",),
        "by_action_space_and_action": ("action_space", "action"),
        "by_action_space_and_dimension": ("action_space", "dimension"),
        "by_action_space_and_budget_phase": ("action_space", "budget_phase"),
        "by_action_space_action_dimension_budget_phase": (
            "action_space",
            "action",
            "dimension",
            "budget_phase",
        ),
    }
    return {
        "schema_version": FIDELITY_SCHEMA_VERSION,
        "replay_process_environment": replay_process_environment(),
        "n_records": len(records),
        "n_snapshots": len({record.snapshot_id for record in records}),
        "overall": _group_summary(records),
        "fidelity": {
            "median_rank_correlation": median_rank,
            "classification": _classification(median_rank),
            "descriptor_median_rank_correlations": descriptor_medians,
            "descriptor_classifications": {
                descriptor: _classification(correlation) for descriptor, correlation in descriptor_medians.items()
            },
            "thresholds": {
                "strong": ">= 0.7",
                "usable but imperfect": ">= 0.5 and < 0.7",
                "weak": "< 0.5",
            },
        },
        "snapshot_action_order": snapshot_metrics,
        "groupings": {name: _summaries_for_grouping(records, fields) for name, fields in grouping_fields.items()},
    }


def write_fidelity_csv(records: Sequence[ActionFeatureFidelityRecord], path: Path | str) -> None:
    """Write tidy fidelity records to CSV with JSON-encoded tuple columns."""
    if not records:
        raise ValueError("At least one fidelity record is required.")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]))
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            for key in ("action_history", "proxy_control_identity", "actual_control_identity"):
                row[key] = json.dumps(row[key], separators=(",", ":"))
            writer.writerow(row)


def write_fidelity_summary(summary: Mapping[str, Any], path: Path | str) -> None:
    """Write a machine-readable fidelity summary."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_task_ids(path: Path) -> set[str]:
    payload = (
        OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if path.suffix in {".yaml", ".yml"}
        else _load_json(path)
    )
    if isinstance(payload, Mapping):
        payload = payload.get("task_ids")
    if not isinstance(payload, list) or any(not isinstance(task_id, str) for task_id in payload):
        raise ValueError(f"{path} must contain a list or a manifest object with a task_ids list.")
    return set(payload)


def _load_panel(path: Path) -> list[FidelityPanelEntry]:
    payload = _load_json(path)
    if isinstance(payload, Mapping):
        payload = payload.get("entries")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list or an object with an entries list.")
    entries: list[FidelityPanelEntry] = []
    for item in payload:
        if not isinstance(item, Mapping):
            raise ValueError("Every fidelity-panel entry must be an object.")
        entries.append(
            FidelityPanelEntry(
                task_id=str(item["task_id"]),
                inner_seed=int(item["inner_seed"]),
                action_space=str(item["action_space"]),
                history_policy=str(item["history_policy"]),
                action_history=tuple(int(action) for action in item.get("action_history", ())),
            )
        )
    return entries


def _load_factory(specification: str) -> FidelityEnvironmentFactory:
    try:
        module_name, attribute = specification.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("Factory must use the form 'python.module:callable'.") from error
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Factory target {specification!r} is not callable.")
    return factory


def main(argv: Sequence[str] | None = None) -> int:
    """Run an explicitly supplied panel with an explicitly supplied factory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True, help="JSON fidelity-panel entries.")
    parser.add_argument(
        "--factory",
        required=True,
        help="Callable as module:attribute accepting (task_id, inner_seed, action_space).",
    )
    parser.add_argument(
        "--forbidden-task-ids",
        type=Path,
        required=True,
        help="JSON sealed/test task inventory rejected before replay.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    arguments = parser.parse_args(argv)
    require_deterministic_replay_process_environment()

    entries = _load_panel(arguments.panel)
    forbidden = _load_task_ids(arguments.forbidden_task_ids)
    factory = _load_factory(arguments.factory)
    records = run_fidelity_panel(entries, factory, forbidden_task_ids=forbidden)
    write_fidelity_csv(records, arguments.output_csv)
    write_fidelity_summary(summarize_action_feature_fidelity(records), arguments.output_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
