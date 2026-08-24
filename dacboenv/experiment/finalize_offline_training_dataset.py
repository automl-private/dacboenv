"""Freeze task-disjoint train/dev/holdout views of the clean f5 corpus."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.protocol import load_manifest, manifest_hash
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.offline.normalization import fit_observation_normalizer
from dacboenv.offline.provenance import headroom_provenance
from dacboenv.offline.schema import (
    ALPHA_GRID,
    BEHAVIOR_COMPONENT,
    INITIAL_BRANCH_COMPONENT,
    OFFLINE_FINAL_SCHEMA_VERSION,
    ensure_no_object_arrays,
    validate_behavior_arrays,
    validate_branch_arrays,
)

SPLITS = ("train", "dev", "holdout")
SCENARIOS = ("bbob", "lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost")
EXPECTED_SOURCE_SCHEMAS = frozenset({"dacbo-offline-training-f5-v2"})
SMALL_STRATUM_SIZE = 6
EXPECTED_EPISODES = 2400
EXPECTED_TRANSITIONS = 52980
INTERACTION_FREQUENCY = 5


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    ensure_no_object_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_split_manifest(
    output_root: Path,
    split: str,
    task_ids: list[str],
    task_split_hash: str,
    *,
    domain: str = "mixed",
) -> dict[str, Any]:
    """Write one protocol-valid offline task manifest.

    Offline development remains a validation role even though its tasks came
    from the original training inventory. Holdout is defined but non-runnable.
    """
    manifest_split = {"train": "train", "dev": "validation", "holdout": "test"}[split]
    suffix = "" if domain == "mixed" else f"-{domain}"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "id": f"offline-{split}{suffix}-v3",
        "domain": domain,
        "split": manifest_split,
        "status": "blocked" if split == "holdout" else "ready",
        "runnable": split != "holdout",
        "role": "sealed_offline_holdout" if split == "holdout" else f"offline_{split}",
        "task_ids": task_ids,
        "inner_seeds": [None],
        "source_context_split": "train",
        "task_split_hash": task_split_hash,
    }
    if split == "holdout":
        payload["blockers"] = ["Offline holdout is sealed; ordinary training and model selection must not access it."]
    payload["manifest_hash"] = manifest_hash(payload)
    filename = f"offline_{split}_v3{'' if domain == 'mixed' else f'_{domain}'}.yaml"
    destination = output_root / filename
    _atomic_text(destination, OmegaConf.to_yaml(OmegaConf.create(payload), sort_keys=False))
    config_destination = output_root / "hydra_configs" / "instance_sets" / filename
    _atomic_text(config_destination, destination.read_text(encoding="utf-8"))
    return {
        "path": str(destination),
        "hydra_config_path": str(config_destination),
        "sha256": file_sha256(destination),
        "manifest_hash": payload["manifest_hash"],
    }


def _write_validation_panel(
    output_root: Path,
    task_ids: list[str],
    task_split_hash: str,
    *,
    domain: str,
) -> dict[str, Any]:
    """Write a dev-only monitoring panel consumed by online fine-tuning."""
    suffix = "" if domain == "mixed" else f"_{domain}"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "id": f"offline-dev-v3-{domain}",
        "domain": domain,
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "role": "offline_development_monitoring",
        "task_ids": task_ids,
        "inner_seeds": [0, 1],
        "source_context_split": "train",
        "task_split_hash": task_split_hash,
        "panel": {
            "tier": "offline_dev",
            "episode_count": 2 * len(task_ids),
            "checkpoint_selection_eligible": False,
            "sealed_holdout_access": False,
        },
    }
    payload["manifest_hash"] = manifest_hash(payload)
    filename = f"offline_dev_v3{suffix}.yaml"
    destination = output_root / "hydra_configs" / "validation_panels" / filename
    _atomic_text(destination, OmegaConf.to_yaml(OmegaConf.create(payload), sort_keys=False))
    return {
        "hydra_config_path": str(destination),
        "sha256": file_sha256(destination),
        "manifest_hash": payload["manifest_hash"],
    }


def _unicode(values: list[str]) -> np.ndarray:
    width = max((len(item) for item in values), default=1)
    return np.asarray(values, dtype=f"U{width}")  # type: ignore[no-any-return]


def _episodes(source: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    rows = [json.loads(str(item)) for item in source["episode_metadata_json"]]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Episode metadata rows must be JSON objects.")
    return cast("list[dict[str, Any]]", rows)


def _validate_expected_corpus(
    source: Mapping[str, np.ndarray],
    episodes: list[dict[str, Any]],
    repository: Path,
) -> None:
    """Require the complete clean fixed-f5 behavior campaign described by protocol."""
    expected_tasks: set[str] = set()
    for filename in ("bbob_train.yaml", "yahpo_train.yaml"):
        manifest = load_manifest(repository / "dacboenv/configs/instance_sets" / filename)
        expected_tasks.update(map(str, manifest["task_ids"]))
    actual_tasks = {str(row["task_id"]) for row in episodes}
    if actual_tasks != expected_tasks:
        raise ValueError("Behavior corpus task identities do not exactly match the frozen BBOB/YAHPO train manifests.")
    if len(episodes) != EXPECTED_EPISODES or len(np.asarray(source["rewards"])) != EXPECTED_TRANSITIONS:
        raise ValueError(
            f"Expected {EXPECTED_EPISODES} episodes/{EXPECTED_TRANSITIONS} transitions, "
            f"received {len(episodes)}/{len(np.asarray(source['rewards']))}."
        )
    expected_policies = {*(f"static_f5_alpha{index}" for index in range(5)), "uniform_random_f5"}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        groups[str(row["task_id"])].append(row)
    for task_id, rows in groups.items():
        combinations = {(str(row["policy_id"]), int(row["seed"])) for row in rows}
        expected = {(policy, seed) for policy in expected_policies for seed in range(5)}
        if combinations != expected or len(rows) != len(expected):
            raise ValueError(f"Task {task_id!r} does not contain exactly six policies across seeds 0..4.")
        if any(int(row.get("interaction_frequency", INTERACTION_FREQUENCY)) != INTERACTION_FREQUENCY for row in rows):
            raise ValueError(f"Task {task_id!r} contains a non-f5 behavior episode.")


def _task_stratum(task_id: str, row: Mapping[str, Any]) -> str:
    if task_id.startswith("bbob/"):
        return f"bbob:d{int(task_id.split('/')[1])}"
    if not task_id.startswith("yahpo/so/"):
        raise ValueError(f"Offline finalization rejects unknown or sealed task namespace: {task_id!r}.")
    scenario = str(row.get("scenario", task_id.split("/")[2]))
    if scenario == "nb301" or scenario not in SCENARIOS:
        raise ValueError(f"Offline finalization rejects YAHPO scenario {scenario!r}.")
    return f"yahpo:{scenario}"


def _stable_order(tasks: list[str], seed: str) -> list[str]:
    return sorted(tasks, key=lambda task: canonical_sha256({"seed": seed, "task_id": task}))


def _allocation(size: int) -> tuple[int, int, int]:
    if size == SMALL_STRATUM_SIZE:
        return 4, 1, 1
    if size == 12:  # noqa: PLR2004
        return 8, 2, 2
    if size == 8:  # noqa: PLR2004
        return 4, 2, 2
    if size < 3:  # noqa: PLR2004
        raise ValueError(f"Every split stratum needs at least three tasks; received {size}.")
    dev = max(1, round(0.175 * size))
    holdout = max(1, round(0.175 * size))
    return size - dev - holdout, dev, holdout


def deterministic_task_split(episodes: list[dict[str, Any]], split_seed: str) -> dict[str, str]:
    """Assign whole tasks to deterministic, stratified disjoint splits."""
    metadata: dict[str, dict[str, Any]] = {}
    for row in episodes:
        task_id = str(row["task_id"])
        current = {"domain": str(row["domain"]), "scenario": str(row["scenario"])}
        if task_id in metadata and metadata[task_id] != current:
            raise ValueError(f"Inconsistent task metadata for {task_id!r}.")
        metadata[task_id] = current
    strata: dict[str, list[str]] = defaultdict(list)
    for task_id, row in metadata.items():
        strata[_task_stratum(task_id, row)].append(task_id)
    assignment: dict[str, str] = {}
    for stratum, tasks in sorted(strata.items()):
        ordered = _stable_order(tasks, f"{split_seed}|{stratum}")
        counts = _allocation(len(ordered))
        offset = 0
        for split, count in zip(SPLITS, counts, strict=True):
            for task in ordered[offset : offset + count]:
                assignment[task] = split
            offset += count
        if offset != len(ordered):
            raise RuntimeError(f"Split allocation failed for {stratum}.")
    if set(assignment) != set(metadata):
        raise RuntimeError("Task split does not cover the source corpus exactly.")
    return assignment


def _source_array(source: Mapping[str, np.ndarray], canonical: str, legacy: str) -> np.ndarray:
    key = canonical if canonical in source else legacy
    if key not in source:
        raise ValueError(f"Source dataset lacks {canonical!r}/{legacy!r}.")
    return np.asarray(source[key])  # type: ignore[no-any-return]


def _phase(before: np.ndarray, budgets: np.ndarray) -> np.ndarray:
    fraction = before.astype(np.float64) / np.maximum(budgets.astype(np.float64), 1.0)
    return np.asarray(np.minimum(np.floor(4 * fraction).astype(np.int8), 3), dtype=np.int8)  # type: ignore[no-any-return]


def build_behavior_split(
    source: Mapping[str, np.ndarray],
    episodes: list[dict[str, Any]],
    task_split: Mapping[str, str],
    split: str,
    source_hash: str,
    split_hash: str,
) -> dict[str, np.ndarray]:
    """Create one compact, row-addressable fixed-f5 behavior split."""
    offsets = np.asarray(source["episode_offsets"], dtype=np.int64)
    selected = [index for index, row in enumerate(episodes) if task_split[str(row["task_id"])] == split]
    if not selected:
        raise ValueError(f"Split {split!r} has no episodes.")
    slices = [np.arange(offsets[index], offsets[index + 1], dtype=np.int64) for index in selected]
    rows = np.concatenate(slices)
    counts = np.asarray([len(item) for item in slices], dtype=np.int64)
    new_episode_index: np.ndarray = np.repeat(np.arange(len(selected), dtype=np.int32), counts)
    selected_metadata = [episodes[index] for index in selected]
    tasks = sorted({str(row["task_id"]) for row in selected_metadata})
    task_index = {task: index for index, task in enumerate(tasks)}
    transition_metadata = [selected_metadata[int(index)] for index in new_episode_index]
    budgets = np.asarray([int(row["bo_budget"]) for row in transition_metadata], dtype=np.int32)
    before = _source_array(source, "bo_evaluations_before", "bo_evaluations_before")[rows].astype(np.int32)
    actions = _source_array(source, "action_index", "actions")[rows].astype(np.int8)
    scenario_index = {name: index for index, name in enumerate(SCENARIOS)}
    policy_values = [str(row["policy_id"]) for row in transition_metadata]
    task_values = [str(row["task_id"]) for row in transition_metadata]
    arrays = {
        "global_state": _source_array(source, "global_state", "observations__global_state")[rows].astype(np.float32),
        "action_features": _source_array(source, "action_features", "observations__action_features")[rows].astype(
            np.float32
        ),
        "next_global_state": _source_array(source, "next_global_state", "next_observations__global_state")[rows].astype(
            np.float32
        ),
        "next_action_features": _source_array(source, "next_action_features", "next_observations__action_features")[
            rows
        ].astype(np.float32),
        "action_index": actions,
        "alpha": ALPHA_GRID[actions].astype(np.float32),
        "reward": _source_array(source, "reward", "rewards")[rows].astype(np.float64),
        "terminated": _source_array(source, "terminated", "terminated")[rows].astype(np.bool_),
        "truncated": _source_array(source, "truncated", "truncated")[rows].astype(np.bool_),
        "behavior_probability": _source_array(source, "behavior_probability", "behavior_probability")[rows].astype(
            np.float32
        ),
        "behavior_log_probability": _source_array(source, "behavior_log_probability", "behavior_log_probability")[
            rows
        ].astype(np.float32),
        "task_id": _unicode(task_values),
        "task_index": np.asarray([task_index[task] for task in task_values], dtype=np.int16),
        "domain_id": np.asarray([0 if row["domain"] == "bbob" else 1 for row in transition_metadata], dtype=np.int8),
        "scenario_id": np.asarray([scenario_index[str(row["scenario"])] for row in transition_metadata], dtype=np.int8),
        "phase_bin": _phase(before, budgets),
        "episode_index": new_episode_index,
        "episode_offsets": np.concatenate(([0], np.cumsum(counts, dtype=np.int64))),
        "seed": np.asarray([int(row["seed"]) for row in transition_metadata], dtype=np.int32),
        "policy_id": _unicode(policy_values),
        "bo_evaluations_before": before,
        "bo_evaluations_after": _source_array(source, "bo_evaluations_after", "bo_evaluations_after")[rows].astype(
            np.int32
        ),
        "realized_duration": _source_array(source, "realized_duration", "realized_duration")[rows].astype(np.int16),
    }
    metadata = {
        "schema_version": OFFLINE_FINAL_SCHEMA_VERSION,
        "component": BEHAVIOR_COMPONENT,
        "split": split,
        "context_split": split,
        "data_role": "offline_training"
        if split == "train"
        else "offline_development"
        if split == "dev"
        else "sealed_holdout",
        "source_dataset_sha256": source_hash,
        "task_split_hash": split_hash,
        "task_ids": tasks,
        "task_count": len(tasks),
        "episode_count": len(selected),
        "transition_count": len(rows),
        "interaction_frequency": 5,
        "holdout_sealed": split == "holdout",
    }
    arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    validate_behavior_arrays(arrays)
    return arrays


def build_initial_counterfactual_split(
    source: Mapping[str, np.ndarray],
    episodes: list[dict[str, Any]],
    task_split: Mapping[str, str],
    split: str,
    source_hash: str,
    split_hash: str,
    tie_tolerance: float,
) -> dict[str, np.ndarray]:
    """Extract five static rewards from each identical first model state."""
    offsets = np.asarray(source["episode_offsets"], dtype=np.int64)
    groups: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for episode, row in enumerate(episodes):
        if task_split[str(row["task_id"])] != split or str(row.get("policy_kind")) != "static":
            continue
        action = int(row["configured_action"])
        key = (str(row["task_id"]), int(row["seed"]))
        if action in groups[key]:
            raise ValueError(f"Duplicate static action {action} for {key}.")
        groups[key][action] = episode
    globals_source = _source_array(source, "global_state", "observations__global_state")
    features_source = _source_array(source, "action_features", "observations__action_features")
    rewards = _source_array(source, "reward", "rewards")
    global_rows, feature_rows, q_rows, tasks, seeds, domains, scenarios, digests = [], [], [], [], [], [], [], []
    duplicate_groups: list[str] = []
    scenario_index = {name: index for index, name in enumerate(SCENARIOS)}
    for (task_id, seed), policies in sorted(groups.items()):
        if set(policies) != set(range(5)):
            raise ValueError(f"Incomplete static action group for {(task_id, seed)}: {sorted(policies)}.")
        starts = [int(offsets[policies[action]]) for action in range(5)]
        global_candidates = [np.asarray(globals_source[index]) for index in starts]
        feature_candidates = [np.asarray(features_source[index]) for index in starts]
        if any(not np.array_equal(global_candidates[0], candidate) for candidate in global_candidates[1:]):
            raise ValueError(f"Initial global states are not exactly paired for {(task_id, seed)}.")
        if any(not np.array_equal(feature_candidates[0], candidate) for candidate in feature_candidates[1:]):
            raise ValueError(f"Initial action features are not exactly paired for {(task_id, seed)}.")
        q = np.asarray([rewards[index] for index in starts], dtype=np.float64)
        row = episodes[policies[0]]
        global_rows.append(global_candidates[0])
        feature_rows.append(feature_candidates[0])
        candidate_groups: dict[bytes, int] = {}
        encoded_groups = []
        for feature_row in np.asarray(feature_candidates[0])[:, 1:]:
            key_bytes = np.asarray(feature_row, dtype=np.float32).tobytes(order="C")
            candidate_groups.setdefault(key_bytes, len(candidate_groups))
            encoded_groups.append(candidate_groups[key_bytes])
        duplicate_groups.append(json.dumps(encoded_groups, separators=(",", ":")))
        q_rows.append(q)
        tasks.append(task_id)
        seeds.append(seed)
        domains.append(0 if row["domain"] == "bbob" else 1)
        scenarios.append(scenario_index[str(row["scenario"])])
        digests.append(
            canonical_sha256(
                {
                    "task_id": task_id,
                    "seed": seed,
                    "global": global_candidates[0].tolist(),
                    "features": feature_candidates[0].tolist(),
                }
            )
        )
    q5 = np.asarray(q_rows, dtype=np.float64)
    maximum = q5.max(axis=1, keepdims=True)
    sorted_q = np.sort(q5, axis=1)
    arrays = {
        "global_state": np.asarray(global_rows, dtype=np.float32),
        "action_features": np.asarray(feature_rows, dtype=np.float32),
        "action_alpha": ALPHA_GRID.copy(),
        "q5": q5,
        "centered_advantage_q5": q5 - q5.mean(axis=1, keepdims=True),
        "valid_action_mask": np.ones_like(q5, dtype=np.bool_),
        "tie_mask_q5": (maximum - q5) <= tie_tolerance,
        "top1_top2_gap_q5": sorted_q[:, -1] - sorted_q[:, -2],
        "oracle_action": q5.argmax(axis=1).astype(np.int8),
        "task_id": _unicode(tasks),
        "domain_id": np.asarray(domains, dtype=np.int8),
        "scenario_id": np.asarray(scenarios, dtype=np.int8),
        "phase_bin": np.zeros(len(tasks), dtype=np.int8),
        "seed": np.asarray(seeds, dtype=np.int32),
        "source_policy_id": _unicode(["five_static_paired"] * len(tasks)),
        "source_state_digest": _unicode(digests),
        "source_replay_digest": _unicode(digests),
        "candidate_duplicate_groups": _unicode(duplicate_groups),
        "branch_protocol_hash": _unicode(
            [canonical_sha256({"horizon": 5, "interaction_frequency": 5, "source": "paired_static"})] * len(tasks)
        ),
        "reference_metadata_json": _unicode(
            [
                json.dumps(
                    {
                        "kind": "best_known" if task.startswith("yahpo/") else "exact",
                        "source_dataset_sha256": source_hash,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for task in tasks
            ]
        ),
    }
    metadata = {
        "schema_version": OFFLINE_FINAL_SCHEMA_VERSION,
        "component": INITIAL_BRANCH_COMPONENT,
        "split": split,
        "context_split": split,
        "data_role": "offline_training"
        if split == "train"
        else "offline_development"
        if split == "dev"
        else "sealed_holdout",
        "source_dataset_sha256": source_hash,
        "task_split_hash": split_hash,
        "state_count": len(tasks),
        "task_count": len(set(tasks)),
        "horizon": 5,
        "value_semantics": "fixed_action_normalized_potential_improvement_over_next_5_bo_evaluations",
        "not_long_return_q": True,
        "tie_tolerance": tie_tolerance,
    }
    arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    validate_branch_arrays(arrays)
    return arrays


def finalize(  # noqa: C901, PLR0912, PLR0915 - protocol assembly is linear and explicit
    config: DictConfig,
) -> dict[str, Any]:
    """Run validation, splitting, counterfactual extraction, and normalization."""
    source_path = Path(str(config.behavior_npz)).resolve()
    output_root = Path(str(config.output_root)).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_hash = file_sha256(source_path)
    with np.load(source_path, allow_pickle=False) as payload:
        source = {key: np.asarray(payload[key]) for key in payload.files}
    ensure_no_object_arrays(source)
    source_metadata = json.loads(str(source["dataset_metadata_json"].item()))
    if source_metadata.get("schema_version") not in EXPECTED_SOURCE_SCHEMAS:
        raise ValueError(
            f"Expected clean f5 training schema {sorted(EXPECTED_SOURCE_SCHEMAS)}, "
            f"got {source_metadata.get('schema_version')!r}."
        )
    if source_metadata.get("context_split") != "train" or source_metadata.get("data_role") != "training":
        raise ValueError("Source must be the clean training-manifest f5 corpus, not a test/evaluation archive.")
    episodes = _episodes(source)
    if bool(config.get("strict_expected_corpus", True)):
        _validate_expected_corpus(source, episodes, Path(__file__).resolve().parents[2])
    if any(str(row["task_id"]).startswith("yahpo/so/nb301/") for row in episodes):
        raise ValueError("nb301 is forbidden in offline training.")
    existing_manifest = output_root / "final_offline_dataset_manifest.json"
    if existing_manifest.is_file():
        existing = cast("dict[str, Any]", json.loads(existing_manifest.read_text(encoding="utf-8")))
        if existing.get("source_dataset_sha256") != source_hash:
            raise RuntimeError("Refusing to reuse a frozen output root for a different source dataset.")
        for split in SPLITS:
            for component in ("behavior", "initial_counterfactual"):
                record = existing["outputs"][split][component]
                path = Path(record["path"])
                if not path.is_file() or file_sha256(path) != record["sha256"]:
                    raise RuntimeError(f"Frozen offline output is missing or changed: {path}.")
        return existing
    task_split = deterministic_task_split(episodes, str(config.split_seed))
    split_hash = canonical_sha256(task_split)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    for split in SPLITS:
        behavior = build_behavior_split(source, episodes, task_split, split, source_hash, split_hash)
        branch = build_initial_counterfactual_split(
            source, episodes, task_split, split, source_hash, split_hash, float(config.tie_tolerance)
        )
        behavior_path = output_root / f"behavior_{split}.npz"
        branch_path = output_root / f"initial_counterfactual_{split}.npz"
        _atomic_npz(behavior_path, behavior)
        _atomic_npz(branch_path, branch)
        outputs[split] = {
            "behavior": {"path": str(behavior_path), "sha256": file_sha256(behavior_path)},
            "initial_counterfactual": {"path": str(branch_path), "sha256": file_sha256(branch_path)},
            "task_count": len({task for task, value in task_split.items() if value == split}),
            "episode_count": int(len(behavior["episode_offsets"]) - 1),
            "transition_count": len(behavior["reward"]),
            "counterfactual_state_count": len(branch["q5"]),
        }
    train_path = Path(outputs["train"]["behavior"]["path"])
    with np.load(train_path, allow_pickle=False) as train:
        normalizer = fit_observation_normalizer(
            train["global_state"], train["action_features"], train_dataset_sha256=file_sha256(train_path)
        )
        reward = np.asarray(train["reward"], dtype=np.float64)
    normalization_payload = normalizer.to_dict()
    normalization_payload["reward_diagnostics"] = {
        "count": int(reward.size),
        "mean": float(reward.mean()),
        "standard_deviation": float(reward.std()),
        "minimum": float(reward.min()),
        "maximum": float(reward.max()),
        "quantiles": {str(value): float(np.quantile(reward, value)) for value in (0.01, 0.1, 0.5, 0.9, 0.99)},
        "positive_fraction": float(np.mean(reward > 0)),
        "used_as_training_normalizer": False,
    }
    _atomic_json(output_root / "normalization_schema.json", normalization_payload)
    task_splits = {split: sorted(task for task, value in task_split.items() if value == split) for split in SPLITS}
    split_manifests: dict[str, Any] = {}
    for split in SPLITS:
        split_manifests[split] = _write_split_manifest(output_root, split, task_splits[split], split_hash)
        for domain in ("bbob", "yahpo"):
            selected = [task for task in task_splits[split] if task.startswith(f"{domain}/")]
            split_manifests[f"{split}_{domain}"] = _write_split_manifest(
                output_root,
                split,
                selected,
                split_hash,
                domain=domain,
            )
    validation_panels = {
        "dev_mixed": _write_validation_panel(output_root, task_splits["dev"], split_hash, domain="mixed"),
        "dev_yahpo": _write_validation_panel(
            output_root,
            [task for task in task_splits["dev"] if task.startswith("yahpo/")],
            split_hash,
            domain="yahpo",
        ),
    }
    manifest = {
        "schema_version": OFFLINE_FINAL_SCHEMA_VERSION,
        "source_dataset": str(source_path),
        "source_dataset_sha256": source_hash,
        "source_data_revision": source_metadata.get("source_revision"),
        "repository_revision": current_source_revision(Path(__file__).resolve().parents[2]),
        "split_seed": str(config.split_seed),
        "task_split_hash": split_hash,
        "task_splits": task_splits,
        "outputs": outputs,
        "task_manifests": split_manifests,
        "validation_panels": validation_panels,
        "normalization_sha256": file_sha256(output_root / "normalization_schema.json"),
        "external_learned_headroom": headroom_provenance(
            Path(str(config.learned_headroom_root)) if config.learned_headroom_root else None
        ),
        "scientific_rules": {
            "split_unit": "complete_task_id",
            "learned_headroom_rows_copied": False,
            "holdout_sealed": True,
            "test_contexts_allowed": False,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    _atomic_json(existing_manifest, manifest)
    stats = {
        "schema_version": OFFLINE_FINAL_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "task_counts": {split: len(task_splits[split]) for split in SPLITS},
        "episode_counts": {split: outputs[split]["episode_count"] for split in SPLITS},
        "transition_counts": {split: outputs[split]["transition_count"] for split in SPLITS},
        "counterfactual_state_counts": {split: outputs[split]["counterfactual_state_count"] for split in SPLITS},
    }
    _atomic_json(output_root / "final_offline_dataset_stats.json", stats)
    return manifest


@hydra.main(version_base=None, config_path="../configs/offline_finalizer", config_name="base")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Hydra entry point for deterministic finalization."""
    result = finalize(config)
    print(OmegaConf.to_yaml(OmegaConf.create({"manifest_hash": result["manifest_hash"], "outputs": result["outputs"]})))


if __name__ == "__main__":
    main()
