"""CARP-S optimizer and portable NPZ schema for offline DAC datasets."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from dacboenv.env.action import AcqParameterActionSpace, WEIDiscreteActionSpace, WEITempoRLActionSpace
from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.optimizer import DACBOEnvOptimizer

if TYPE_CHECKING:
    from carps.utils.trials import TrialInfo, TrialValue

    from dacboenv.env.observations.types import ObsType

OFFLINE_DATASET_SCHEMA_VERSION = "dacbo-offline-transitions-v1"
OFFLINE_OBSERVATION_KEYS = (
    "global_state",
    "action_features",
    "gp_hp_summary",
    "gp_hp_change",
    "gp_hp_raw",
    "gp_hp_raw_mask",
    "gp_hp_raw_roles",
)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _copy_observation(observation: ObsType, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    missing = sorted(set(keys) - set(observation))
    if missing:
        raise ValueError(f"Offline observation is missing required keys: {missing}.")
    copied = {key: np.asarray(observation[key], dtype=np.float32).copy() for key in keys}
    for key, value in copied.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Offline observation key {key!r} contains non-finite values.")
    return copied


@dataclass(frozen=True, slots=True)
class CanonicalWEIAction:
    """Common action representation across discrete, TempoRL, and SAWEI."""

    alpha: float
    requested_duration: int
    alpha_index: int
    duration_index: int


def canonical_wei_action(action_space: Any, action: Any, interaction_frequency: int) -> CanonicalWEIAction:
    """Map one public DAC action to ``[alpha, requested duration]``.

    Continuous SAWEI alphas use ``alpha_index=-1``.  A fixed interaction
    frequency has ``duration_index=-1`` because duration is part of the
    environment configuration rather than a policy-selected categorical axis.
    """
    if isinstance(action_space, WEITempoRLActionSpace):
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        if values.shape != (2,):
            raise ValueError(f"Expected a two-axis TempoRL action, got shape {values.shape}.")
        duration_index, alpha_index = (int(values[0]), int(values[1]))
        return CanonicalWEIAction(
            alpha=float(action_space._param_levels[alpha_index]),
            requested_duration=int(action_space._step_durations[duration_index]),
            alpha_index=alpha_index,
            duration_index=duration_index,
        )
    if isinstance(action_space, WEIDiscreteActionSpace):
        alpha_index = int(np.asarray(action).item())
        return CanonicalWEIAction(
            alpha=float(action_space._param_levels[alpha_index]),
            requested_duration=int(interaction_frequency),
            alpha_index=alpha_index,
            duration_index=-1,
        )
    if isinstance(action_space, AcqParameterActionSpace) and action_space._action.attr == "_alpha":
        return CanonicalWEIAction(
            alpha=float(np.asarray(action).item()),
            requested_duration=int(interaction_frequency),
            alpha_index=-1,
            duration_index=-1,
        )
    raise TypeError(f"Offline WEI datasets do not support action controller {type(action_space).__name__}.")


@dataclass(slots=True)
class _PendingTransition:
    observation: dict[str, np.ndarray]
    action: CanonicalWEIAction
    bo_evaluations_before: int
    reward: float = 0.0
    realized_duration: int = 0


class OfflineDatasetOptimizer(DACBOEnvOptimizer):
    """Record decision-level transitions while CARP-S owns the BO rollout.

    CARP-S still performs every objective evaluation through its ordinary
    ``Optimizer.run()`` loop.  The recorder only groups the per-evaluation
    rewards produced by :class:`DACBOEnvOptimizer` into one transition per
    policy decision, matching ``DACBOEnv.step()`` for fixed and TempoRL action
    durations.
    """

    def __init__(
        self,
        *args: Any,
        dataset_output_path: str | Path,
        dataset_policy_id: str,
        dataset_schema_version: str = OFFLINE_DATASET_SCHEMA_VERSION,
        stored_observation_keys: list[str] | tuple[str, ...] = OFFLINE_OBSERVATION_KEYS,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._dataset_output_path = Path(dataset_output_path).resolve()
        self._dataset_status_path = self._dataset_output_path.with_name("offline_episode_status.json")
        self._dataset_policy_id = str(dataset_policy_id)
        self._dataset_schema_version = str(dataset_schema_version)
        self._stored_observation_keys = tuple(str(key) for key in stored_observation_keys)
        if self._dataset_schema_version != OFFLINE_DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported offline dataset schema {self._dataset_schema_version!r}; "
                f"expected {OFFLINE_DATASET_SCHEMA_VERSION!r}."
            )
        if self._stored_observation_keys != OFFLINE_OBSERVATION_KEYS:
            raise ValueError(
                f"The v1 offline schema requires the canonical structured + GP key order {OFFLINE_OBSERVATION_KEYS!r}."
            )
        self._pending_transition: _PendingTransition | None = None
        self._transitions: list[dict[str, Any]] = []
        self._started_at = _utc_now()

    def _setup_optimizer(self) -> Any:
        solver = super()._setup_optimizer()
        outer_budget = int(self.task.optimization_resources.n_trials)
        inner_budget = int(self._dacboenv._smac_instance._scenario.n_trials)
        if outer_budget != inner_budget:
            raise ValueError(
                "Offline collection requires identical CARP-S and inner DACBO budgets; "
                f"received outer={outer_budget} and inner={inner_budget}."
            )
        initial = _copy_observation(self._state, self._stored_observation_keys)
        if not self._dacboenv.observation_space.contains(self._state):
            raise ValueError("Initial offline observation is not contained in the configured Gymnasium space.")
        del initial
        return solver

    def ask(self) -> TrialInfo:
        """Start one buffered transition whenever the policy makes a decision."""
        is_model_based = len(self.solver.runhistory) >= len(
            self.solver.intensifier.config_selector._initial_design_configs
        )
        is_new_decision = is_model_based and self._skip_duration == 0
        observation = _copy_observation(self._state, self._stored_observation_keys) if is_new_decision else None
        bo_evaluations_before = self._dacboenv.get_n_finished_trials()
        trial_info = super().ask()
        if is_new_decision:
            if self._pending_transition is not None:
                raise RuntimeError("A new policy decision began before the preceding transition was finalized.")
            if self.action is None:
                raise ValueError("Offline collection requires an explicit WEI action; received a no-op action.")
            self._pending_transition = _PendingTransition(
                observation=observation or {},
                action=canonical_wei_action(
                    self._dacboenv._action_space,
                    self.action,
                    self._dacboenv.interaction_frequency,
                ),
                bo_evaluations_before=bo_evaluations_before,
            )
        return trial_info

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Accumulate inner rewards and close a transition at its duration."""
        super().tell(trial_info, trial_value)
        pending = self._pending_transition
        if pending is None:
            raise RuntimeError("Received a model-based result without a pending offline transition.")
        pending.reward += float(self._dacboenv.get_reward())
        pending.realized_duration += 1
        bo_evaluations_after = self._dacboenv.get_n_finished_trials()
        terminal = bo_evaluations_after >= int(self._dacboenv._smac_instance._scenario.n_trials)
        duration_complete = pending.realized_duration >= pending.action.requested_duration
        if not (terminal or duration_complete):
            return
        if pending.realized_duration > pending.action.requested_duration:
            raise RuntimeError("The realized action duration exceeded its requested duration.")
        next_observation = _copy_observation(self._state, self._stored_observation_keys)
        if not self._dacboenv.observation_space.contains(self._state):
            raise ValueError("Offline next observation is not contained in the configured Gymnasium space.")
        self._transitions.append(
            {
                "observation": pending.observation,
                "next_observation": next_observation,
                "action": pending.action,
                "reward": pending.reward,
                "terminated": terminal,
                "truncated": False,
                "bo_evaluations_before": pending.bo_evaluations_before,
                "bo_evaluations_after": bo_evaluations_after,
                "realized_duration": pending.realized_duration,
            }
        )
        self._pending_transition = None

    def run(self) -> Any:
        """Run through CARP-S and persist failures from setup or rollout."""
        try:
            return super().run()
        except Exception as error:
            _atomic_json(
                self._dataset_status_path,
                {
                    "schema_version": self._dataset_schema_version,
                    "status": "failed",
                    "policy_id": self._dataset_policy_id,
                    "task_id": self.task.name,
                    "seed": self._seed,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                    "started_at": self._started_at,
                    "failed_at": _utc_now(),
                },
            )
            raise

    def _run(self) -> Any:
        """Materialize one successful episode after the normal CARP-S loop."""
        incumbent = super()._run()
        if self._pending_transition is not None:
            raise RuntimeError("CARP-S stopped with a partial offline transition still buffered.")
        self._write_episode()
        return incumbent

    def _write_episode(self) -> None:
        if not self._transitions:
            raise RuntimeError("Offline episode contains no model-based policy transitions.")
        arrays: dict[str, np.ndarray] = {}
        for key in self._stored_observation_keys:
            arrays[f"observations__{key}"] = np.stack(
                [transition["observation"][key] for transition in self._transitions]
            ).astype(np.float32, copy=False)
            arrays[f"next_observations__{key}"] = np.stack(
                [transition["next_observation"][key] for transition in self._transitions]
            ).astype(np.float32, copy=False)

        actions = [transition["action"] for transition in self._transitions]
        terminals = np.asarray([transition["terminated"] for transition in self._transitions], dtype=np.bool_)
        timeouts = np.asarray([transition["truncated"] for transition in self._transitions], dtype=np.bool_)
        arrays.update(
            {
                "actions": np.asarray(
                    [[action.alpha, action.requested_duration] for action in actions],
                    dtype=np.float32,
                ),
                "action_alpha_index": np.asarray([action.alpha_index for action in actions], dtype=np.int8),
                "action_duration_index": np.asarray([action.duration_index for action in actions], dtype=np.int8),
                "rewards": np.asarray([transition["reward"] for transition in self._transitions], dtype=np.float64),
                # Keep Gymnasium names and conventional offline-RL aliases.
                "terminated": terminals,
                "truncated": timeouts,
                "terminals": terminals.copy(),
                "timeouts": timeouts.copy(),
                "requested_duration": np.asarray([action.requested_duration for action in actions], dtype=np.int16),
                "realized_duration": np.asarray(
                    [transition["realized_duration"] for transition in self._transitions], dtype=np.int16
                ),
                "bo_evaluations_before": np.asarray(
                    [transition["bo_evaluations_before"] for transition in self._transitions], dtype=np.int32
                ),
                "bo_evaluations_after": np.asarray(
                    [transition["bo_evaluations_after"] for transition in self._transitions], dtype=np.int32
                ),
                "transition_index": np.arange(len(self._transitions), dtype=np.int32),
            }
        )
        reference = getattr(self._dacboenv, "_objective_reference", None)
        metadata = {
            "schema_version": self._dataset_schema_version,
            "task_id": self.task.name,
            "domain": self.task.name.split("/", maxsplit=1)[0].lower(),
            "seed": int(self._seed) if self._seed is not None else None,
            "policy_id": self._dataset_policy_id,
            "transition_count": len(self._transitions),
            "bo_budget": int(self._dacboenv._smac_instance._scenario.n_trials),
            "initial_design_size": len(
                self._dacboenv._smac_instance.intensifier.config_selector._initial_design_configs
            ),
            "observation_keys": list(self._stored_observation_keys),
            "observation_shapes": {
                key: list(arrays[f"observations__{key}"].shape[1:]) for key in self._stored_observation_keys
            },
            "action_columns": ["wei_alpha", "requested_duration"],
            "transition_unit": "one external acquisition-control decision",
            "reward_aggregation": "sum of reference-regret rewards over realized_duration BO evaluations",
            "action_convention": {
                "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
                "duration_grid": [1, 5, 10],
                "continuous_alpha_index": -1,
                "configured_duration_index": -1,
            },
            "reference_kind": None if reference is None else reference.kind,
            "reference_source_hash": None if reference is None else reference.source_hash,
            "gp_diagnostics": self._dacboenv.get_gp_hyperparameter_diagnostics(),
            "started_at": self._started_at,
            "completed_at": _utc_now(),
        }
        metadata["observation_schema_hash"] = canonical_sha256(
            {
                "keys": metadata["observation_keys"],
                "shapes": metadata["observation_shapes"],
                "dtypes": dict.fromkeys(self._stored_observation_keys, "float32"),
            }
        )
        arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))

        self._dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._dataset_output_path.with_name(f".{self._dataset_output_path.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self._dataset_output_path)
        sha256 = file_sha256(self._dataset_output_path)
        _atomic_json(
            self._dataset_status_path,
            {
                "schema_version": self._dataset_schema_version,
                "status": "success",
                "policy_id": self._dataset_policy_id,
                "task_id": self.task.name,
                "seed": self._seed,
                "transition_count": len(self._transitions),
                "episode_path": str(self._dataset_output_path),
                "episode_sha256": sha256,
                "completed_at": _utc_now(),
            },
        )


def validate_episode_npz(  # noqa: C901, PLR0912
    path: Path,
    *,
    expected_task_id: str | None = None,
    expected_seed: int | None = None,
    expected_policy_id: str | None = None,
) -> dict[str, Any]:
    """Validate one shard without permitting pickle-backed arrays."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        required = {
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "terminals",
            "timeouts",
            "requested_duration",
            "realized_duration",
            "metadata_json",
            *(f"observations__{key}" for key in OFFLINE_OBSERVATION_KEYS),
            *(f"next_observations__{key}" for key in OFFLINE_OBSERVATION_KEYS),
        }
        missing = sorted(required - set(payload.files))
        if missing:
            raise ValueError(f"Offline episode {path} is missing arrays: {missing}.")
        metadata = json.loads(str(payload["metadata_json"].item()))
        n_transitions = int(payload["rewards"].shape[0])
        if n_transitions <= 0:
            raise ValueError(f"Offline episode {path} has no transitions.")
        for key in payload.files:
            value = payload[key]
            if value.dtype.hasobject:
                raise ValueError(f"Offline array {key!r} uses forbidden object dtype.")
            if key != "metadata_json" and value.shape[0] != n_transitions:
                raise ValueError(f"Offline array {key!r} has {value.shape[0]} rows, expected {n_transitions}.")
        if payload["actions"].shape != (n_transitions, 2):
            raise ValueError(f"Offline actions must have shape ({n_transitions}, 2).")
        if not np.isfinite(payload["rewards"]).all():
            raise ValueError("Offline rewards contain non-finite values.")
        for key in OFFLINE_OBSERVATION_KEYS:
            if not np.isfinite(payload[f"observations__{key}"]).all():
                raise ValueError(f"Offline observations__{key} contains non-finite values.")
            if not np.isfinite(payload[f"next_observations__{key}"]).all():
                raise ValueError(f"Offline next_observations__{key} contains non-finite values.")

    checks = (
        ("task_id", expected_task_id),
        ("seed", expected_seed),
        ("policy_id", expected_policy_id),
    )
    for field, expected in checks:
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(f"Offline {field} mismatch: {metadata.get(field)!r} != {expected!r}.")
    if metadata.get("schema_version") != OFFLINE_DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unexpected offline schema {metadata.get('schema_version')!r}.")
    return metadata


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an arbitrary dataset file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "OFFLINE_DATASET_SCHEMA_VERSION",
    "OFFLINE_OBSERVATION_KEYS",
    "CanonicalWEIAction",
    "OfflineDatasetOptimizer",
    "canonical_wei_action",
    "validate_episode_npz",
]
