"""Compare an explicit Stage-A PPO checkpoint with controls on saved branches."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dacboenv.experiment.collect_snapshots import (
    bind_snapshot_action_space_factory,
    configured_structured_action_space,
    read_snapshots,
    verify_portable_snapshot_replay,
)
from dacboenv.experiment.protocol import (
    load_manifest,
    require_runnable_manifest,
    sealed_final_test_task_ids,
)
from dacboenv.experiment.run_snapshot_branches import validate_branch_row_provenance
from dacboenv.experiment.snapshot_branch import BOSnapshot, replay_snapshot


def resolve_checkpoint(run_root: Path, checkpoint: str) -> tuple[Path, Path | None]:
    """Resolve a complete best/final model and its checkpoint-specific normalizer."""
    candidates = {
        "best": (run_root / "validation" / "best_balanced_model.zip", run_root / "validation" / "best_model.zip"),
        "final": (run_root / "model.zip", run_root / "final_model.zip"),
    }
    if checkpoint not in candidates:
        raise ValueError("checkpoint must be 'best' or 'final'.")
    try:
        model_path = next(path for path in candidates[checkpoint] if path.is_file())
    except StopIteration as error:
        raise FileNotFoundError(f"No complete {checkpoint} PPO checkpoint exists below {run_root}.") from error
    normalization_candidates = (
        (
            run_root / "validation" / "best_balanced_vecnormalize.pkl",
            run_root / "validation" / "best_vecnormalize.pkl",
            run_root / "validation" / "vecnormalize.pkl",
        )
        if checkpoint == "best"
        else (run_root / "vecnormalize.pkl", run_root / "final_vecnormalize.pkl")
    )
    normalization_path = next((path for path in normalization_candidates if path.is_file()), None)
    return model_path, normalization_path


def _configured_vecnormalize(run_root: Path) -> bool:
    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-A run is incomplete: missing {config_path}.")
    cfg = OmegaConf.load(config_path)
    return bool(OmegaConf.select(cfg, "experiment.vecnormalize", default=False))


def _branch_values(
    path: Path,
    snapshots: Sequence[BOSnapshot],
) -> tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, int, int], float]]:
    values: dict[tuple[int, int, int], float] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            snapshot_index = int(row["snapshot_index"])
            action = int(row["action"])
            horizon = int(row["horizon"])
            trace = json.loads(row["configuration_trace"])
            if not isinstance(trace, list) or len(trace) not in {0, horizon}:
                raise ValueError(
                    "Branch CSV configuration trace must be empty for a lightweight test environment "
                    f"or contain exactly horizon={horizon} configurations."
                )
            if not 0 <= snapshot_index < len(snapshots):
                raise ValueError(f"Branch CSV has invalid snapshot index {snapshot_index}.")
            snapshot = snapshots[snapshot_index]
            validate_branch_row_provenance(row, snapshot, snapshot_index=snapshot_index)
            key = (snapshot_index, action, horizon)
            if key in values:
                raise ValueError(f"Duplicate branch outcome for {key!r}.")
            value = float(row["normalized_potential_improvement"])
            if not np.isfinite(value):
                raise ValueError(f"Branch CSV contains non-finite Q value for {key!r}.")
            values[key] = value
    actions = tuple(sorted({key[1] for key in values}))
    horizons = tuple(sorted({key[2] for key in values}))
    expected = {
        (index, action, horizon) for index in range(len(snapshots)) for action in actions for horizon in horizons
    }
    if set(values) != expected:
        raise ValueError("Branch CSV does not contain a complete snapshot/action/horizon matrix.")
    return actions, horizons, values


def _policy_probabilities(model: PPO, observation: Any) -> tuple[int, np.ndarray]:
    deterministic_action, _state = model.predict(observation, deterministic=True)
    observation_tensor, _vectorized = model.policy.obs_to_tensor(observation)
    distribution = model.policy.get_distribution(observation_tensor).distribution
    probabilities = getattr(distribution, "probs", None)
    if probabilities is None:
        raise TypeError("Stage-A dynamic-headroom analysis requires a categorical PPO actor.")
    vector = probabilities.detach().cpu().numpy().reshape(-1)
    return int(np.asarray(deterministic_action).item()), vector


def _validation_only(snapshots: Sequence[BOSnapshot], validation_manifest_path: Path) -> None:
    manifest = load_manifest(validation_manifest_path)
    require_runnable_manifest(manifest)
    if manifest["split"] != "validation":
        raise PermissionError(f"Learned controls require a validation manifest, got split={manifest['split']!r}.")
    expected_id = str(manifest["id"])
    expected_hash = str(manifest["manifest_hash"])
    invalid_sources = sorted(
        {
            (snapshot.source_manifest, snapshot.source_manifest_hash)
            for snapshot in snapshots
            if (snapshot.source_manifest, snapshot.source_manifest_hash) != (expected_id, expected_hash)
        }
    )
    if invalid_sources:
        raise PermissionError(
            "Snapshot source manifest identity/hash does not match the explicit frozen validation manifest: "
            f"{invalid_sources!r}."
        )
    unknown_tasks = sorted({snapshot.task_id for snapshot in snapshots} - set(manifest["task_ids"]))
    if unknown_tasks:
        raise PermissionError(f"Snapshots contain tasks outside the frozen validation manifest: {unknown_tasks!r}.")
    allowed_seeds = {int(seed) for seed in manifest["inner_seeds"] if seed is not None}
    unknown_seeds = sorted({snapshot.inner_seed for snapshot in snapshots} - allowed_seeds)
    if unknown_seeds:
        raise PermissionError(f"Snapshots contain seeds outside the frozen validation manifest: {unknown_seeds!r}.")


def analyze_snapshot_policy(  # noqa: C901, PLR0912, PLR0915
    *,
    run_root: Path,
    checkpoint: str,
    snapshot_path: Path,
    branch_csv: Path,
    env_factory: Callable[[str, int, str], Any],
    validation_manifest_path: Path,
    forbidden_task_ids: Collection[str],
) -> dict[str, Any]:
    """Compute learned, modal, marginal, static, and oracle values per horizon."""
    model_path, normalization_path = resolve_checkpoint(run_root, checkpoint)
    uses_normalization = _configured_vecnormalize(run_root)
    if uses_normalization and normalization_path is None:
        raise FileNotFoundError(
            f"Stage-A run used VecNormalize, but {checkpoint} checkpoint-specific normalization is missing."
        )
    if not uses_normalization:
        normalization_path = None
    snapshots = read_snapshots(snapshot_path)
    if not snapshots:
        raise ValueError("At least one portable snapshot is required.")
    forbidden = set(forbidden_task_ids) | set(sealed_final_test_task_ids())
    prohibited = sorted({snapshot.task_id for snapshot in snapshots} & forbidden)
    if prohibited:
        raise PermissionError(f"Learned-policy snapshot analysis refuses sealed/test task IDs: {prohibited!r}.")
    _validation_only(snapshots, validation_manifest_path)
    snapshot_action_spaces = {snapshot.action_space for snapshot in snapshots}
    if len(snapshot_action_spaces) != 1 or "" in snapshot_action_spaces:
        raise ValueError(
            f"Learned-policy analysis requires one recorded action family, got {snapshot_action_spaces!r}."
        )
    snapshot_action_space = next(iter(snapshot_action_spaces))
    configured_action_space = configured_structured_action_space(OmegaConf.load(run_root / ".hydra" / "config.yaml"))
    if configured_action_space != snapshot_action_space:
        raise ValueError(
            "Stage-A checkpoint action family does not match the portable snapshots: "
            f"checkpoint={configured_action_space!r}, snapshots={snapshot_action_space!r}."
        )
    actions, horizons, values = _branch_values(branch_csv, snapshots)
    replay_factory = bind_snapshot_action_space_factory(snapshots, env_factory)
    for snapshot in snapshots:
        verify_portable_snapshot_replay(snapshot, replay_factory)
    model = PPO.load(model_path)

    normalizer: VecNormalize | None = None
    normalizer_probe: Any | None = None
    if normalization_path is not None:
        normalizer_probe = replay_factory(snapshots[0].task_id, snapshots[0].inner_seed)
        normalizer_probe.reset()
        normalizer = VecNormalize.load(normalization_path, DummyVecEnv([lambda: normalizer_probe]))
        normalizer.training = False
        normalizer.norm_reward = False

    deterministic_actions: list[int] = []
    probabilities: list[np.ndarray] = []
    try:
        for snapshot in snapshots:
            env = replay_snapshot(snapshot, replay_factory)
            try:
                observation = env.get_observation()
                if normalizer is not None:
                    observation = normalizer.normalize_obs(
                        {key: np.asarray(value)[None, ...] for key, value in observation.items()}
                    )
                action, action_probabilities = _policy_probabilities(model, observation)
                if action not in actions or len(action_probabilities) != len(actions):
                    raise ValueError("PPO action space does not match the saved branch action space.")
                deterministic_actions.append(action)
                probabilities.append(action_probabilities)
            finally:
                env.close()
    finally:
        if normalizer is not None:
            normalizer.close()
        elif normalizer_probe is not None:
            normalizer_probe.close()

    probability_matrix = np.asarray(probabilities, dtype=float)
    deterministic_frequency = np.asarray(
        [deterministic_actions.count(action) / len(deterministic_actions) for action in actions],
        dtype=float,
    )
    modal_index = int(np.flatnonzero(np.isclose(deterministic_frequency, np.max(deterministic_frequency)))[0])
    modal_action = actions[modal_index]
    horizon_results: list[dict[str, Any]] = []
    for horizon in horizons:
        q_values = np.asarray(
            [[values[(index, action, horizon)] for action in actions] for index in range(len(snapshots))],
            dtype=float,
        )
        static_values = np.mean(q_values, axis=0)
        best_static_index = int(np.flatnonzero(np.isclose(static_values, np.max(static_values)))[0])
        best_static = float(static_values[best_static_index])
        dynamic_oracle = float(np.mean(np.max(q_values, axis=1)))
        policy_values = {
            "learned_deterministic": float(
                np.mean([q_values[index, actions.index(action)] for index, action in enumerate(deterministic_actions)])
            ),
            "learned_stochastic_expected": float(np.mean(np.sum(probability_matrix * q_values, axis=1))),
            "modal_static_clone": float(np.mean(q_values[:, modal_index])),
            "marginal_frequency_matched_random": float(np.mean(q_values @ deterministic_frequency)),
            "best_validation_static": best_static,
            "dynamic_oracle": dynamic_oracle,
        }
        denominator = max(dynamic_oracle - best_static, np.finfo(float).eps)
        horizon_results.append(
            {
                "horizon": horizon,
                "values": policy_values,
                "captured_headroom": {
                    name: (value - best_static) / denominator for name, value in policy_values.items()
                },
                "best_validation_static_action": actions[best_static_index],
                "dynamic_headroom": dynamic_oracle - best_static,
            }
        )
    cfg = OmegaConf.load(run_root / ".hydra" / "config.yaml")
    return {
        "run_root": str(run_root.resolve()),
        "outer_ppo_seed": OmegaConf.select(cfg, "seed"),
        "checkpoint": checkpoint,
        "model_path": str(model_path),
        "normalization_path": None if normalization_path is None else str(normalization_path),
        "source_validation_manifests": sorted({snapshot.source_manifest for snapshot in snapshots}),
        "source_validation_manifest_hashes": sorted({snapshot.source_manifest_hash for snapshot in snapshots}),
        "n_snapshots": len(snapshots),
        "actions": list(actions),
        "deterministic_actions": deterministic_actions,
        "deterministic_action_frequencies": {
            str(action): float(deterministic_frequency[index]) for index, action in enumerate(actions)
        },
        "mean_stochastic_action_probabilities": {
            str(action): float(np.mean(probability_matrix[:, index])) for index, action in enumerate(actions)
        },
        "modal_static_action": modal_action,
        "horizons": horizon_results,
    }


def _load_callable(specification: str) -> Callable[..., Any]:
    module_name, attribute_name = specification.split(":", maxsplit=1)
    target = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(target):
        raise TypeError(f"Factory target {specification!r} is not callable.")
    return target


def _load_task_ids(path: Path) -> set[str]:
    payload: Any = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if isinstance(payload, Mapping):
        payload = payload.get("task_ids")
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain task_ids.")
    return {str(task_id) for task_id in payload}


def main(argv: Sequence[str] | None = None) -> int:
    """Analyze one explicit, complete Stage-A checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "final"), required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--branches", type=Path, required=True)
    parser.add_argument("--factory", required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--forbidden-task-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        result = analyze_snapshot_policy(
            run_root=arguments.run_root,
            checkpoint=arguments.checkpoint,
            snapshot_path=arguments.snapshots,
            branch_csv=arguments.branches,
            env_factory=_load_callable(arguments.factory),
            validation_manifest_path=arguments.validation_manifest,
            forbidden_task_ids=_load_task_ids(arguments.forbidden_task_ids),
        )
    except FileNotFoundError as error:
        print(f"SKIP: incomplete Stage-A analysis input: {error}")
        return 0
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
