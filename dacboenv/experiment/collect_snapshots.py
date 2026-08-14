"""Portable collection of replayable DACBO/SMAC snapshots.

The scientific record is JSON Lines containing task/seed/action history and
auditable completed evaluations.  It deliberately contains no pickled SMAC
objects; branches reconstruct the optimizer from deterministic seed streams.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from omegaconf import OmegaConf

from dacboenv.experiment.protocol import load_manifest, require_runnable_manifest, sealed_final_test_task_ids
from dacboenv.experiment.snapshot_branch import (
    CANONICAL_ACTION_SPACE_NAMES,
    DEFAULT_REPLAY_TOLERANCE,
    BOSnapshot,
    CompletedBOEvaluation,
    SnapshotReplayError,
    assert_snapshot_action_space,
    replay_process_environment,
    replay_snapshot,
    require_deterministic_replay_process_environment,
)
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.experiment.task_metadata import parse_task_metadata
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider


class SnapshotPolicy(Protocol):
    """History-generating policy used by the collector."""

    name: str
    outer_seed: int | None

    def __call__(self, observation: Any, env: Any) -> int:
        """Return one discrete action for the current observation."""


def configured_structured_action_space(cfg: Any) -> str:  # noqa: PLR0911
    """Resolve one canonical structured action family from a Hydra config."""
    action_space_id = str(OmegaConf.select(cfg, "action_space_id", default="")).lower()
    if action_space_id.startswith("wei-discrete-"):
        return "wei"
    if action_space_id.startswith("lcb-quantile-"):
        return "lcb_quantile"
    if action_space_id.startswith("ucb-quantile-"):
        return "ucb_quantile"
    if action_space_id.startswith("af-select-"):
        return "af_selection"

    target = str(OmegaConf.select(cfg, "dacboenv.action_space_class._target_", default=""))
    if target.endswith("WEIDiscreteActionSpace"):
        return "wei"
    if target.endswith("PosteriorModeActionSpace"):
        return "af_selection"
    if target.endswith("PosteriorQuantileActionSpace"):
        acquisition_target = str(
            OmegaConf.select(
                cfg,
                "dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.acquisition_function._target_",
                default="",
            )
        )
        if acquisition_target.endswith(".LCB"):
            return "lcb_quantile"
        if acquisition_target.endswith(".UCB"):
            return "ucb_quantile"
    raise ValueError("Hydra config does not identify one supported structured action family.")


class StaticSnapshotPolicy:
    """Always select one discrete action."""

    def __init__(self, action: int) -> None:
        self.action = int(action)
        self.name = f"static_{self.action}"
        self.outer_seed: int | None = None

    def __call__(self, observation: Any, env: Any) -> int:  # noqa: ARG002
        """Return the configured static action."""
        return self.action


class InitialDesignOnlySnapshotPolicy:
    """Label a snapshot at reset without fabricating an action-generated history."""

    name = "initial_design_only"
    outer_seed: int | None = None

    def __call__(self, observation: Any, env: Any) -> int:  # noqa: ARG002
        """Refuse to advance because this policy represents reset state only."""
        raise RuntimeError("initial_design_only can collect only budget targets already reached by the initial design.")


class UniformRandomSnapshotPolicy:
    """Seeded state-independent uniform random action policy."""

    def __init__(self, seed: int) -> None:
        self.outer_seed = int(seed)
        self.name = "uniform_random"
        self._rng = np.random.default_rng(self.outer_seed)

    def __call__(self, observation: Any, env: Any) -> int:  # noqa: ARG002
        """Draw one action from the policy's reproducible RNG stream."""
        return int(self._rng.integers(int(env.action_space.n)))


class DefaultSMACEquivalentSnapshotPolicy:
    """Select the EI-equivalent discrete action for compatible controllers."""

    def __init__(self, action_space_name: str) -> None:
        if action_space_name not in {"wei", "af_selection"}:
            raise ValueError(
                "A native default-SMAC/EI history is meaningful only for WEI or acquisition-function selection."
            )
        self.action = 2
        self.name = "default_smac_equivalent"
        self.outer_seed: int | None = None

    def __call__(self, observation: Any, env: Any) -> int:  # noqa: ARG002
        """Return the controller action matching native expected improvement."""
        return self.action


class SB3SnapshotPolicy:
    """Deterministic action wrapper around an explicitly supplied SB3 model."""

    def __init__(self, model: Any, *, checkpoint: str, outer_seed: int | None) -> None:
        if checkpoint not in {"best", "final"}:
            raise ValueError("checkpoint must be 'best' or 'final'.")
        self.model = model
        self.checkpoint = checkpoint
        self.outer_seed = outer_seed
        self.name = f"sb3_{checkpoint}"

    def __call__(self, observation: Any, env: Any) -> int:  # noqa: ARG002
        """Return the model's deterministic action."""
        action, _state = self.model.predict(observation, deterministic=True)
        return int(np.asarray(action).item())


def observation_digest(observation: Any) -> str:
    """Hash a nested NumPy observation without exposing privileged metadata."""

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in sorted(value.items())}
        array = np.asarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "values": array.tolist(),
        }

    payload = json.dumps(canonical(observation), allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def portable_observation_json(observation: Any) -> str:
    """Serialize the exact policy-visible observation without privileged metadata."""

    def canonical(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): canonical(item) for key, item in sorted(value.items())}
        array = np.asarray(value)
        return {"dtype": str(array.dtype), "shape": list(array.shape), "values": array.tolist()}

    return json.dumps(canonical(observation), allow_nan=False, separators=(",", ":"), sort_keys=True)


def _task_metadata(task_id: str) -> dict[str, Any]:
    try:
        metadata = parse_task_metadata(task_id)
    except ValueError:
        # Portable replay is also used with synthetic unit-test/task-factory IDs.
        # Never reinterpret a malformed benchmark namespace, but preserve the
        # historical neutral metadata for explicitly non-benchmark identifiers.
        if task_id.startswith(("bbob/", "yahpo/")):
            raise
        return {"domain": "", "dimension": None, "native_instance": "", "scenario": ""}
    return {
        "domain": metadata.domain,
        "dimension": metadata.dimension,
        "native_instance": metadata.native_instance,
        "scenario": "" if metadata.scenario is None else metadata.scenario,
    }


def _initial_design_hash(evaluations: Sequence[CompletedBOEvaluation], initial_design_size: int) -> str:
    rows = [asdict(evaluation) for evaluation in evaluations[:initial_design_size]]
    payload = json.dumps(rows, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def current_git_commit(repository: Path | None = None) -> str:
    """Return a clean commit or a commit plus dirty-source digest."""
    return current_source_revision(repository)


def _scalar_cost(value: Any) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    return float(values[0]) if values.size else np.inf


def completed_evaluations(env: Any) -> tuple[CompletedBOEvaluation, ...]:
    """Extract a portable, insertion-ordered audit trail from a live SMAC run history."""
    runhistory = env._smac_instance.runhistory
    records: list[CompletedBOEvaluation] = []
    for trial, (trial_key, trial_value) in enumerate(runhistory._data.items()):
        configuration = runhistory.get_config(trial_key.config_id)
        configuration_json = json.dumps(dict(configuration), allow_nan=False, separators=(",", ":"), sort_keys=True)
        status = getattr(trial_value, "status", "UNKNOWN")
        records.append(
            CompletedBOEvaluation(
                trial=trial,
                configuration_json=configuration_json,
                cost=_scalar_cost(trial_value.cost),
                status=getattr(status, "name", str(status)),
                seed=None if trial_key.seed is None else int(trial_key.seed),
                budget=None if trial_key.budget is None else float(trial_key.budget),
            )
        )
    return tuple(records)


def _assert_completed_evaluations_match(
    expected: Sequence[CompletedBOEvaluation],
    actual: Sequence[CompletedBOEvaluation],
    *,
    tolerance: float,
) -> None:
    if len(expected) != len(actual):
        raise SnapshotReplayError(
            f"Completed-evaluation count changed during replay: {len(expected)} != {len(actual)}."
        )
    for index, (saved, replayed) in enumerate(zip(expected, actual, strict=True)):
        saved_identity = (
            saved.trial,
            saved.configuration_json,
            saved.status,
            saved.seed,
            saved.budget,
        )
        replayed_identity = (
            replayed.trial,
            replayed.configuration_json,
            replayed.status,
            replayed.seed,
            replayed.budget,
        )
        if saved_identity != replayed_identity or not np.isclose(
            saved.cost,
            replayed.cost,
            rtol=0.0,
            atol=tolerance,
        ):
            raise SnapshotReplayError(
                f"Completed evaluation {index} changed during deterministic replay: "
                f"saved={saved!r}, replayed={replayed!r}."
            )


def verify_portable_snapshot_replay(  # noqa: C901
    snapshot: BOSnapshot,
    env_factory: Callable[[str, int], Any],
    *,
    tolerance: float = DEFAULT_REPLAY_TOLERANCE,
) -> None:
    """Verify all supplied portable state fields against a deterministic replay.

    This complements :func:`replay_snapshot`, which validates context and the
    action prefix.  No objective evaluations beyond that saved prefix are
    performed.
    """
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}.")
    env = replay_snapshot(snapshot, env_factory)
    try:
        if snapshot.completed_evaluations:
            _assert_completed_evaluations_match(
                snapshot.completed_evaluations,
                completed_evaluations(env),
                tolerance=float(tolerance),
            )
        if snapshot.incumbent is not None and not np.isclose(
            float(env.get_incumbent_cost()),
            snapshot.incumbent,
            rtol=0.0,
            atol=tolerance,
        ):
            raise SnapshotReplayError(
                f"Snapshot incumbent changed during replay: {snapshot.incumbent} != {env.get_incumbent_cost()}."
            )
        if int(env.interaction_frequency) != snapshot.interaction_frequency:
            raise SnapshotReplayError(
                "Snapshot interaction frequency changed during replay: "
                f"{snapshot.interaction_frequency} != {env.interaction_frequency}."
            )
        if snapshot.budget_fraction is not None:
            budget = int(env._smac_instance._scenario.n_trials)
            replayed_fraction = float(env.get_n_finished_trials()) / float(budget)
            if not np.isclose(replayed_fraction, snapshot.budget_fraction, rtol=0.0, atol=tolerance):
                raise SnapshotReplayError(
                    "Snapshot budget fraction changed during replay: "
                    f"{snapshot.budget_fraction} != {replayed_fraction}."
                )
        if snapshot.observation_hash:
            get_observation = getattr(env, "get_observation", None)
            if not callable(get_observation):
                raise SnapshotReplayError("Snapshot has an observation hash but replay environment cannot expose it.")
            replayed_hash = observation_digest(get_observation())
            if replayed_hash != snapshot.observation_hash:
                raise SnapshotReplayError(
                    f"Snapshot observation hash changed during replay: {snapshot.observation_hash} != {replayed_hash}."
                )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _portable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _portable_json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (tuple, list)):
        return [_portable_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _reference_fields(reference: Any | None) -> dict[str, Any]:
    if reference is None:
        return {}
    return {
        "reference_kind": str(reference.kind),
        "reference_value": float(reference.value),
        "reference_source": str(reference.source),
        "reference_source_hash": str(reference.source_hash),
        "reference_runtime_objective_transform": str(reference.runtime_objective_transform),
        "reference_reporting_objective_transform": str(reference.reporting_objective_transform),
        "reference_fidelity_json": json.dumps(
            _portable_json_value(reference.fidelity),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
        "reference_tolerance": float(reference.tolerance),
        "reference_benchmark_code_version": str(reference.benchmark_code_version or ""),
        "reference_benchmark_data_version": str(reference.benchmark_data_version or ""),
    }


def _initial_design_incumbent(env: Any) -> float:
    n_initial = len(env._smac_instance.intensifier.config_selector._initial_design_configs)
    costs = [_scalar_cost(value.cost) for value in env._smac_instance.runhistory._data.values()]
    finite = np.asarray(costs[:n_initial], dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise RuntimeError("Cannot collect a replayable snapshot without a finite initial-design incumbent.")
    return float(np.min(finite))


def collect_context_snapshots(  # noqa: C901, PLR0912, PLR0913
    *,
    task_id: str,
    inner_seed: int,
    env_factory: Callable[[str, int, str], Any],
    policy: SnapshotPolicy,
    budget_fractions: Sequence[float],
    action_space_name: str,
    source_manifest: str,
    source_manifest_hash: str,
    reference: Any | None = None,
    reference_provider: Any | None = None,
    code_commit: str | None = None,
) -> tuple[BOSnapshot, ...]:
    """Collect snapshots at the first transitions reaching target budget fractions."""
    if reference is not None and reference_provider is not None:
        raise ValueError("Supply either reference or reference_provider, not both.")
    if task_id in sealed_final_test_task_ids():
        raise ValueError(f"Snapshot collection intrinsically refuses sealed/test task ID {task_id!r}.")
    targets = tuple(sorted({float(value) for value in budget_fractions}))
    if not targets or any(not np.isfinite(value) or not 0.0 <= value < 1.0 for value in targets):
        raise ValueError("budget_fractions must contain finite unique values in [0, 1).")
    try:
        env = env_factory(task_id, int(inner_seed), action_space_name)
    except TypeError as error:
        raise TypeError(
            "Snapshot environment factories must accept (task_id, inner_seed, action_space_name); "
            "a two-argument factory can silently replay the wrong action family."
        ) from error
    actions: list[int] = []
    snapshots: list[BOSnapshot] = []
    try:
        observation, info = env.reset()
        assert_snapshot_action_space(
            BOSnapshot(task_id, int(inner_seed), action_space=action_space_name),
            env,
        )
        if info.get("task_id", getattr(env, "current_task_id", None)) != task_id:
            raise RuntimeError("Snapshot factory returned a different task than requested.")
        if int(info.get("inner_seed", getattr(env, "current_seed", -1))) != int(inner_seed):
            raise RuntimeError("Snapshot factory returned a different inner seed than requested.")
        if reference_provider is not None:
            objective = env._carps_solver.task.objective_function
            reference = reference_provider.get_reference(task_id, objective, {})
        initial_incumbent = _initial_design_incumbent(env)

        for target in targets:
            while float(env.get_n_finished_trials()) / float(env._n_trials) < target:
                action = int(policy(observation, env))
                if not env.action_space.contains(action):
                    raise ValueError(f"History policy returned invalid action {action}.")
                observation, _reward, terminated, truncated, _step_info = env.step(action)
                actions.append(action)
                if terminated or truncated:
                    raise RuntimeError(f"Episode ended before requested snapshot budget fraction {target}.")

            budget_fraction = float(env.get_n_finished_trials()) / float(env._n_trials)
            evaluations = completed_evaluations(env)
            metadata = _task_metadata(task_id)
            initial_design_size = len(env._smac_instance.intensifier.config_selector._initial_design_configs)
            snapshot_id = hashlib.sha256(
                (
                    f"{source_manifest}|{task_id}|{inner_seed}|{action_space_name}|{policy.name}|{budget_fraction:.17g}"
                ).encode()
            ).hexdigest()
            snapshots.append(
                BOSnapshot(
                    task_id=task_id,
                    inner_seed=int(inner_seed),
                    action_history=tuple(actions),
                    action_space=action_space_name,
                    interaction_frequency=int(env.interaction_frequency),
                    completed_evaluations=evaluations,
                    budget_fraction=budget_fraction,
                    history_policy=policy.name,
                    outer_policy_seed=policy.outer_seed,
                    source_manifest=source_manifest,
                    source_manifest_hash=source_manifest_hash,
                    code_commit=code_commit or current_git_commit(),
                    observation_hash=observation_digest(observation),
                    observation_json=portable_observation_json(observation),
                    snapshot_id=snapshot_id,
                    history_seed=policy.outer_seed,
                    total_budget=int(env._n_trials),
                    initial_design_hash=_initial_design_hash(evaluations, initial_design_size),
                    deterministic_environment_json=json.dumps(
                        replay_process_environment(), separators=(",", ":"), sort_keys=True
                    ),
                    **metadata,
                    incumbent=float(env.get_incumbent_cost()),
                    initial_design_incumbent=initial_incumbent,
                    **_reference_fields(reference),
                )
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return tuple(snapshots)


def collect_snapshot_panel(
    contexts: Iterable[tuple[str, int]],
    *,
    env_factory: Callable[[str, int, str], Any],
    policy_factory: Callable[[str, int], SnapshotPolicy],
    budget_fractions: Sequence[float],
    action_space_name: str,
    source_manifest: str,
    source_manifest_hash: str,
    forbidden_task_ids: Collection[str],
    reference_provider: Any | None = None,
) -> tuple[BOSnapshot, ...]:
    """Collect a guarded non-test panel across fixed task/seed contexts."""
    frozen_contexts = tuple((str(task_id), int(seed)) for task_id, seed in contexts)
    forbidden = set(forbidden_task_ids) | set(sealed_final_test_task_ids())
    prohibited = sorted({task_id for task_id, _seed in frozen_contexts} & forbidden)
    if prohibited:
        raise ValueError(f"Snapshot collection refuses forbidden/test task IDs: {prohibited!r}.")
    snapshots: list[BOSnapshot] = []
    for task_id, inner_seed in frozen_contexts:
        snapshots.extend(
            collect_context_snapshots(
                task_id=task_id,
                inner_seed=inner_seed,
                env_factory=env_factory,
                policy=policy_factory(task_id, inner_seed),
                budget_fractions=budget_fractions,
                action_space_name=action_space_name,
                source_manifest=source_manifest,
                source_manifest_hash=source_manifest_hash,
                reference_provider=reference_provider,
            )
        )
    return tuple(snapshots)


def write_snapshots(path: Path, snapshots: Sequence[BOSnapshot]) -> None:
    """Write portable snapshots as deterministic JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for snapshot in snapshots:
            output.write(json.dumps(asdict(snapshot), allow_nan=False, sort_keys=True) + "\n")


def read_snapshots(path: Path) -> tuple[BOSnapshot, ...]:
    """Read and validate snapshots written by :func:`write_snapshots`."""
    snapshots: list[BOSnapshot] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                row["completed_evaluations"] = tuple(
                    CompletedBOEvaluation(**evaluation) for evaluation in row.get("completed_evaluations", [])
                )
                snapshots.append(BOSnapshot(**row))
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid snapshot at {path}:{line_number}: {error}") from error
    return tuple(snapshots)


def bind_snapshot_action_space_factory(
    snapshots: Sequence[BOSnapshot],
    env_factory: Callable[[str, int, str], Any],
) -> Callable[[str, int], Any]:
    """Bind each replay context to its recorded action family.

    A single model-analysis or branch file may contain many tasks/seeds, but a
    task/seed pair cannot represent two action families because the legacy
    two-argument replay interface could not distinguish them.
    """
    action_space_by_context: dict[tuple[str, int], str] = {}
    for snapshot in snapshots:
        if not snapshot.action_space:
            raise ValueError("Portable snapshots used for real replay must record action_space.")
        context = (snapshot.task_id, snapshot.inner_seed)
        previous = action_space_by_context.setdefault(context, snapshot.action_space)
        if previous != snapshot.action_space:
            raise ValueError(
                "One task/seed context cannot be dispatched to multiple action families: "
                f"context={context!r}, families={previous!r}/{snapshot.action_space!r}."
            )

    def bound_factory(task_id: str, inner_seed: int) -> Any:
        context = (task_id, int(inner_seed))
        try:
            action_space_name = action_space_by_context[context]
        except KeyError as error:
            raise KeyError(f"No recorded action family for replay context {context!r}.") from error
        try:
            return env_factory(task_id, int(inner_seed), action_space_name)
        except TypeError as error:
            raise TypeError(
                "Snapshot replay factories must accept (task_id, inner_seed, action_space_name); "
                "refusing an ambiguous two-argument factory."
            ) from error

    return bound_factory


def _load_callable(specification: str) -> Callable[..., Any]:
    try:
        module_name, attribute_name = specification.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("Factory must use the form 'python.module:callable'.") from error
    target = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(target):
        raise TypeError(f"Factory target {specification!r} is not callable.")
    return target


def _selected_contexts(
    manifest: Mapping[str, Any],
    task_ids: Sequence[str],
    seeds: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    selected_tasks = tuple(task_ids or manifest["task_ids"])
    unknown = sorted(set(selected_tasks) - set(manifest["task_ids"]))
    if unknown:
        raise ValueError(f"Requested tasks are absent from manifest {manifest['id']!r}: {unknown!r}.")
    selected_seeds: tuple[int, ...]
    if seeds:
        selected_seeds = tuple(int(seed) for seed in seeds)
    else:
        manifest_seeds = tuple(manifest["inner_seeds"])
        if any(seed is None for seed in manifest_seeds):
            raise ValueError("A streaming-seed training manifest requires at least one explicit --inner-seed.")
        selected_seeds = tuple(int(seed) for seed in manifest_seeds)
    return tuple((str(task_id), seed) for task_id in selected_tasks for seed in selected_seeds)


def _checkpoint_model(run_root: Path, checkpoint: str) -> tuple[Any, int | None, str]:
    from stable_baselines3 import PPO  # noqa: PLC0415

    candidates = (
        (run_root / "validation" / "best_balanced_model.zip", run_root / "validation" / "best_model.zip")
        if checkpoint == "best"
        else (run_root / "model.zip", run_root / "final_model.zip")
    )
    model_path = next((path for path in candidates if path.is_file()), None)
    if model_path is None:
        raise FileNotFoundError(f"{checkpoint} checkpoint is incomplete below {run_root}")
    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-A run is incomplete: missing {config_path}")
    cfg = OmegaConf.load(config_path)
    if bool(OmegaConf.select(cfg, "experiment.vecnormalize", default=False)):
        raise ValueError(
            "Snapshot collection from a VecNormalize checkpoint requires a checkpoint-specific observation "
            "normalizer and is deliberately unsupported by this raw-environment CLI."
        )
    outer_seed = OmegaConf.select(cfg, "seed")
    return (
        PPO.load(model_path),
        None if outer_seed is None else int(outer_seed),
        configured_structured_action_space(cfg),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Frozen runnable BBOB/YAHPO train/validation manifest.")
    parser.add_argument("--forbidden-task-ids", type=Path, help="Sealed task manifest rejected before factory use.")
    parser.add_argument(
        "--factory",
        help="Callable module:attribute accepting (task_id, inner_seed, action_space_name).",
    )
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--inner-seed", action="append", type=int, default=[])
    parser.add_argument("--budget-fraction", action="append", type=float, default=[])
    parser.add_argument("--action-space", choices=CANONICAL_ACTION_SPACE_NAMES, default="wei")
    parser.add_argument(
        "--policy",
        choices=("initial_design_only", "static", "uniform_random", "default_smac", "sb3"),
        default="static",
    )
    parser.add_argument("--static-action", type=int, default=0)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint", choices=("best", "final"), default="best")
    parser.add_argument("--reference-table", type=Path, help="Required provenance-complete YAHPO reference table.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901, PLR0912
    """Collect an explicitly guarded BBOB train/validation snapshot panel.

    The CLI never guesses a manifest, factory, or learned run. A missing or
    incomplete learned-policy run root is a clean skip.
    """
    args = _build_parser().parse_args(argv)
    if args.run_root is not None and not args.run_root.is_dir():
        print(f"SKIP: Stage-A run root does not exist: {args.run_root}")
        return 0
    if args.dry_run:
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "task_ids": args.task_id,
                    "inner_seeds": args.inner_seed,
                    "run_root": None if args.run_root is None else str(args.run_root),
                    "checkpoint": args.checkpoint,
                    "manifest": None if args.manifest is None else str(args.manifest),
                    "forbidden_task_ids": (None if args.forbidden_task_ids is None else str(args.forbidden_task_ids)),
                    "factory": args.factory,
                    "budget_fractions": args.budget_fraction,
                    "policy": args.policy,
                },
                sort_keys=True,
            )
        )
        return 0
    require_deterministic_replay_process_environment()
    if args.manifest is None or args.forbidden_task_ids is None or args.factory is None:
        raise SystemExit("Real collection requires --manifest, --forbidden-task-ids, and --factory.")
    if not args.budget_fraction:
        raise SystemExit("Real collection requires at least one explicit --budget-fraction.")
    manifest = load_manifest(args.manifest)
    require_runnable_manifest(manifest)
    if manifest["domain"] not in {"bbob", "yahpo"} or manifest["split"] not in {"train", "validation"}:
        raise PermissionError(
            f"Snapshot CLI accepts only runnable BBOB/YAHPO train/validation manifests, got "
            f"domain={manifest['domain']!r}, split={manifest['split']!r}."
        )
    forbidden_manifest = load_manifest(args.forbidden_task_ids)
    if forbidden_manifest["split"] != "test":
        raise ValueError("--forbidden-task-ids must name an explicit sealed test manifest.")
    contexts = _selected_contexts(manifest, args.task_id, args.inner_seed)
    factory = _load_callable(args.factory)
    model: Any | None = None
    outer_seed: int | None = None
    if args.policy == "sb3":
        if args.run_root is None:
            raise SystemExit("--policy sb3 requires an explicit --run-root.")
        try:
            model, outer_seed, model_action_space = _checkpoint_model(args.run_root, args.checkpoint)
        except FileNotFoundError as error:
            print(f"SKIP: {error}")
            return 0
        if model_action_space != args.action_space:
            raise ValueError(
                "SB3 checkpoint action family does not match --action-space: "
                f"checkpoint={model_action_space!r}, requested={args.action_space!r}."
            )

    def policy_factory(_task_id: str, _inner_seed: int) -> SnapshotPolicy:
        if args.policy == "initial_design_only":
            return InitialDesignOnlySnapshotPolicy()
        if args.policy == "static":
            return StaticSnapshotPolicy(args.static_action)
        if args.policy == "uniform_random":
            return UniformRandomSnapshotPolicy(args.policy_seed)
        if args.policy == "default_smac":
            return DefaultSMACEquivalentSnapshotPolicy(args.action_space)
        assert model is not None
        return SB3SnapshotPolicy(model, checkpoint=args.checkpoint, outer_seed=outer_seed)

    if manifest["domain"] == "bbob":
        reference_provider: Any = BBOBExactReferenceProvider()
    else:
        if args.reference_table is None:
            raise SystemExit("YAHPO snapshot collection requires --reference-table.")
        reference_provider = ManifestReferenceProvider(
            args.reference_table,
            expected_runtime_objective_transform="negative_accuracy",
            expected_reporting_objective_transform="one_minus_accuracy",
            expected_fidelity="fixed_maximum",
        )

    snapshots = collect_snapshot_panel(
        contexts,
        env_factory=factory,
        policy_factory=policy_factory,
        budget_fractions=args.budget_fraction,
        action_space_name=args.action_space,
        source_manifest=str(manifest["id"]),
        source_manifest_hash=str(manifest["manifest_hash"]),
        forbidden_task_ids=set(forbidden_manifest["task_ids"]),
        reference_provider=reference_provider,
    )
    write_snapshots(args.output, snapshots)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "snapshot_count": len(snapshots),
                "context_count": len(contexts),
                "source_manifest": manifest["id"],
                "source_manifest_hash": manifest["manifest_hash"],
                "policy": args.policy,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
