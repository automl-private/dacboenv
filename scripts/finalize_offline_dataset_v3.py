#!/usr/bin/env python3
"""Freeze task-grouped offline replay and counterfactual splits.

This script treats ``offline_training_f5_v2.npz`` as the behavior corpus and
never merges learned-policy validation headroom into training.  It creates:

* behavior_train/dev/holdout.npz
* initial_counterfactual_train/dev/holdout.npz
* final_offline_dataset_manifest.json
* final_offline_dataset_stats.json

Splits are by task ID (never by transition or inner seed), stratified by BBOB
dimension and YAHPO scenario.  All five static policies and the random policy,
and all five seeds, stay together for one task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "dacbo-offline-final-v3"
SPLIT_NAMES = ("train", "dev", "holdout")
SCENARIO_NAMES = (
    "bbob",
    "lcbench",
    "rbv2_glmnet",
    "rbv2_ranger",
    "rbv2_rpart",
    "rbv2_super",
    "rbv2_xgboost",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def stable_order(task_ids: list[str], split_seed: str) -> list[str]:
    return sorted(
        task_ids,
        key=lambda task: hashlib.sha256(f"{split_seed}|{task}".encode()).hexdigest(),
    )


def allocate_group(task_ids: list[str], split_seed: str) -> dict[str, str]:
    """Allocate one homogeneous stratum to train/dev/holdout.

    The current clean corpus has six BBOB tasks per dimension, twelve tasks per
    ordinary YAHPO scenario, and eight rbv2_super tasks.  The generic fallback
    retains at least one task in dev and holdout when the group has >=3 tasks.
    """
    ordered = stable_order(task_ids, split_seed)
    n = len(ordered)
    if n == 6:
        counts = (4, 1, 1)
    elif n == 12:
        counts = (8, 2, 2)
    elif n == 8:
        counts = (4, 2, 2)
    elif n >= 3:
        n_dev = max(1, round(0.15 * n))
        n_holdout = max(1, round(0.15 * n))
        n_train = n - n_dev - n_holdout
        if n_train < 1:
            raise ValueError(f"Cannot split stratum of size {n}")
        counts = (n_train, n_dev, n_holdout)
    else:
        raise ValueError(f"Need at least three tasks per stratum, got {n}: {ordered}")

    result: dict[str, str] = {}
    offset = 0
    for split, count in zip(SPLIT_NAMES, counts, strict=True):
        for task in ordered[offset : offset + count]:
            result[task] = split
        offset += count
    if offset != n:
        raise RuntimeError("Split allocation did not consume every task")
    return result


def build_task_split(episode_metadata: list[dict[str, Any]], split_seed: str) -> dict[str, str]:
    task_meta: dict[str, dict[str, Any]] = {}
    for row in episode_metadata:
        task_id = str(row["task_id"])
        candidate = {
            "domain": str(row["domain"]),
            "scenario": str(row["scenario"]),
        }
        if task_id in task_meta and task_meta[task_id] != candidate:
            raise ValueError(f"Inconsistent metadata for {task_id}")
        task_meta[task_id] = candidate

    strata: dict[str, list[str]] = defaultdict(list)
    for task_id, row in task_meta.items():
        if row["domain"] == "bbob":
            dimension = int(task_id.split("/")[1])
            stratum = f"bbob:d{dimension}"
        else:
            stratum = f"yahpo:{row['scenario']}"
        strata[stratum].append(task_id)

    result: dict[str, str] = {}
    for stratum, tasks in sorted(strata.items()):
        assigned = allocate_group(tasks, f"{split_seed}|{stratum}")
        overlap = set(result).intersection(assigned)
        if overlap:
            raise RuntimeError(f"Duplicate tasks across strata: {sorted(overlap)}")
        result.update(assigned)
    if set(result) != set(task_meta):
        raise RuntimeError("Task split is incomplete")
    return result


def phase_bin(before: np.ndarray, budget_by_transition: np.ndarray) -> np.ndarray:
    fraction = before.astype(np.float64) / budget_by_transition.astype(np.float64)
    return np.minimum((fraction * 4).astype(np.int8), 3)


def subset_behavior(
    payload: np.lib.npyio.NpzFile,
    episode_metadata: list[dict[str, Any]],
    selected_episodes: np.ndarray,
    split_name: str,
    source_sha256: str,
    task_split_hash: str,
) -> dict[str, np.ndarray]:
    old_offsets = payload["episode_offsets"].astype(np.int64)
    transition_indices = np.concatenate(
        [np.arange(old_offsets[e], old_offsets[e + 1], dtype=np.int64) for e in selected_episodes]
    )
    transition_counts = np.asarray(
        [old_offsets[e + 1] - old_offsets[e] for e in selected_episodes], dtype=np.int64
    )
    new_offsets = np.concatenate(([0], np.cumsum(transition_counts, dtype=np.int64)))
    new_episode_index = np.repeat(np.arange(len(selected_episodes), dtype=np.int32), transition_counts)

    arrays: dict[str, np.ndarray] = {}
    n_total = int(payload["rewards"].shape[0])
    for key in payload.files:
        if key in {"episode_index", "episode_offsets", "episode_metadata_json", "dataset_metadata_json"}:
            continue
        value = payload[key]
        if value.ndim >= 1 and value.shape[0] == n_total:
            arrays[key] = value[transition_indices]

    selected_metadata = [episode_metadata[int(e)] for e in selected_episodes]
    max_meta = max(len(json.dumps(row, sort_keys=True, separators=(",", ":"))) for row in selected_metadata)
    arrays["episode_metadata_json"] = np.asarray(
        [json.dumps(row, sort_keys=True, separators=(",", ":")) for row in selected_metadata],
        dtype=f"U{max_meta}",
    )
    arrays["episode_offsets"] = new_offsets
    arrays["episode_index"] = new_episode_index

    task_ids = np.asarray([str(row["task_id"]) for row in selected_metadata])
    unique_tasks = sorted(set(task_ids.tolist()))
    task_to_index = {task: index for index, task in enumerate(unique_tasks)}
    episode_task_index = np.asarray([task_to_index[task] for task in task_ids], dtype=np.int16)
    arrays["task_index"] = episode_task_index[new_episode_index]
    arrays["domain_id"] = np.asarray(
        [0 if selected_metadata[e]["domain"] == "bbob" else 1 for e in new_episode_index],
        dtype=np.int8,
    )
    scenario_to_index = {name: index for index, name in enumerate(SCENARIO_NAMES)}
    arrays["scenario_id"] = np.asarray(
        [scenario_to_index[str(selected_metadata[e]["scenario"])] for e in new_episode_index],
        dtype=np.int8,
    )
    budget_by_transition = np.asarray(
        [int(selected_metadata[e]["bo_budget"]) for e in new_episode_index], dtype=np.int32
    )
    arrays["phase_bin"] = phase_bin(arrays["bo_evaluations_before"], budget_by_transition)

    metadata = {
        "schema_version": SCHEMA,
        "component": "behavior_replay",
        "split": split_name,
        "source_dataset_sha256": source_sha256,
        "task_split_hash": task_split_hash,
        "episode_count": int(len(selected_episodes)),
        "transition_count": int(len(transition_indices)),
        "task_count": int(len(unique_tasks)),
        "task_ids": unique_tasks,
        "scenario_names": list(SCENARIO_NAMES),
        "phase_bins": ["early_0_25", "middle_25_50", "middle_50_75", "late_75_100"],
        "observation_keys": ["global_state", "action_features"],
        "interaction_frequency": 5,
        "training_allowed": split_name == "train",
        "model_selection_allowed": split_name == "dev",
        "final_offline_holdout": split_name == "holdout",
    }
    arrays["dataset_metadata_json"] = np.asarray(
        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
    )
    return arrays


def build_initial_counterfactuals(
    payload: np.lib.npyio.NpzFile,
    episode_metadata: list[dict[str, Any]],
    task_split: dict[str, str],
    split_name: str,
    source_sha256: str,
    task_split_hash: str,
) -> dict[str, np.ndarray]:
    offsets = payload["episode_offsets"].astype(np.int64)
    # NPZ members are compressed zip entries.  Materialize these three arrays
    # once; scalar indexing the lazy NpzFile repeatedly would decompress the
    # same member thousands of times.
    global_array = np.asarray(payload["observations__global_state"])
    action_feature_array = np.asarray(payload["observations__action_features"])
    reward_array = np.asarray(payload["rewards"])
    contexts: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for episode, row in enumerate(episode_metadata):
        if task_split[str(row["task_id"])] != split_name or row["policy_kind"] != "static":
            continue
        action = int(row["configured_action"])
        key = (str(row["task_id"]), int(row["seed"]))
        if action in contexts[key]:
            raise RuntimeError(f"Duplicate action {action} for context {key}")
        contexts[key][action] = episode

    global_states, action_features, q_values = [], [], []
    task_ids, seeds, domains, scenarios = [], [], [], []
    for (task_id, seed), action_episodes in sorted(contexts.items()):
        if set(action_episodes) != set(range(5)):
            raise RuntimeError(f"Incomplete static actions for {(task_id, seed)}: {sorted(action_episodes)}")
        first_indices = [int(offsets[action_episodes[action]]) for action in range(5)]
        first_global = [global_array[index] for index in first_indices]
        first_action_features = [action_feature_array[index] for index in first_indices]
        if not all(np.array_equal(first_global[0], value) for value in first_global[1:]):
            raise RuntimeError(f"Static policies do not share the initial global state: {(task_id, seed)}")
        if not all(np.array_equal(first_action_features[0], value) for value in first_action_features[1:]):
            raise RuntimeError(f"Static policies do not share initial action features: {(task_id, seed)}")
        global_states.append(first_global[0])
        action_features.append(first_action_features[0])
        q_values.append([float(reward_array[index]) for index in first_indices])
        sample_meta = episode_metadata[action_episodes[0]]
        task_ids.append(task_id)
        seeds.append(seed)
        domains.append(0 if sample_meta["domain"] == "bbob" else 1)
        scenarios.append(SCENARIO_NAMES.index(str(sample_meta["scenario"])))

    q = np.asarray(q_values, dtype=np.float64)
    ordered = np.sort(q, axis=1)
    best = np.argmax(q, axis=1).astype(np.int8)
    gap = (ordered[:, -1] - ordered[:, -2]).astype(np.float64)
    max_task_len = max(map(len, task_ids), default=1)
    metadata = {
        "schema_version": SCHEMA,
        "component": "initial_same_state_q5",
        "split": split_name,
        "source_dataset_sha256": source_sha256,
        "task_split_hash": task_split_hash,
        "context_count": len(task_ids),
        "task_count": len(set(task_ids)),
        "horizon": 5,
        "actions": [0, 1, 2, 3, 4],
        "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "training_allowed": split_name == "train",
        "model_selection_allowed": split_name == "dev",
        "final_offline_holdout": split_name == "holdout",
        "important_limitation": "Only the first model-based state is same-state across static trajectories.",
    }
    return {
        "observations__global_state": np.asarray(global_states, dtype=np.float32),
        "observations__action_features": np.asarray(action_features, dtype=np.float32),
        "q5": q,
        "centered_advantage_q5": q - q.mean(axis=1, keepdims=True),
        "best_action": best,
        "top1_top2_gap": gap,
        "task_id": np.asarray(task_ids, dtype=f"U{max_task_len}"),
        "seed": np.asarray(seeds, dtype=np.int32),
        "domain_id": np.asarray(domains, dtype=np.int8),
        "scenario_id": np.asarray(scenarios, dtype=np.int8),
        "dataset_metadata_json": np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="offline_training_f5_v2.npz")
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--split-seed", default="dacbo-offline-v3-split-2026-08-23")
    parser.add_argument("--headroom-root", type=Path, default=None)
    args = parser.parse_args()

    source = args.source.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_sha = sha256_file(source)

    with np.load(source, allow_pickle=False) as payload:
        source_metadata = json.loads(str(payload["dataset_metadata_json"].item()))
        if source_metadata.get("schema_version") != "dacbo-offline-training-f5-v2":
            raise ValueError(f"Unexpected source schema: {source_metadata}")
        episode_metadata = [json.loads(str(value)) for value in payload["episode_metadata_json"]]
        task_split = build_task_split(episode_metadata, args.split_seed)
        task_split_hash = canonical_sha256(task_split)

        outputs: dict[str, dict[str, Any]] = {}
        for split_name in SPLIT_NAMES:
            selected_episodes = np.asarray(
                [
                    index
                    for index, row in enumerate(episode_metadata)
                    if task_split[str(row["task_id"])] == split_name
                ],
                dtype=np.int32,
            )
            behavior_path = output_root / f"behavior_{split_name}.npz"
            counterfactual_path = output_root / f"initial_counterfactual_{split_name}.npz"
            atomic_npz(
                behavior_path,
                subset_behavior(
                    payload,
                    episode_metadata,
                    selected_episodes,
                    split_name,
                    source_sha,
                    task_split_hash,
                ),
            )
            atomic_npz(
                counterfactual_path,
                build_initial_counterfactuals(
                    payload,
                    episode_metadata,
                    task_split,
                    split_name,
                    source_sha,
                    task_split_hash,
                ),
            )
            with np.load(behavior_path, allow_pickle=False) as behavior:
                behavior_meta = json.loads(str(behavior["dataset_metadata_json"].item()))
            with np.load(counterfactual_path, allow_pickle=False) as counterfactual:
                counterfactual_meta = json.loads(str(counterfactual["dataset_metadata_json"].item()))
            outputs[split_name] = {
                "behavior_path": str(behavior_path),
                "behavior_sha256": sha256_file(behavior_path),
                "behavior": behavior_meta,
                "initial_counterfactual_path": str(counterfactual_path),
                "initial_counterfactual_sha256": sha256_file(counterfactual_path),
                "initial_counterfactual": counterfactual_meta,
            }

    external_headroom = None
    if args.headroom_root is not None:
        root = args.headroom_root.resolve()
        status = root / "headroom_status.json"
        comparisons = root / "policy_comparisons.csv"
        external_headroom = {
            "root": str(root),
            "status_sha256": sha256_file(status) if status.is_file() else None,
            "policy_comparisons_sha256": sha256_file(comparisons) if comparisons.is_file() else None,
            "training_allowed": False,
            "role": "external_validation_diagnostic_only",
        }

    split_tasks = {
        split: sorted(task for task, assigned in task_split.items() if assigned == split)
        for split in SPLIT_NAMES
    }
    manifest = {
        "schema_version": SCHEMA,
        "created_from": str(source),
        "source_dataset_sha256": source_sha,
        "source_dataset_metadata": source_metadata,
        "split_seed": args.split_seed,
        "task_split_hash": task_split_hash,
        "task_splits": split_tasks,
        "outputs": outputs,
        "external_headroom": external_headroom,
        "rules": {
            "split_unit": "task_id",
            "no_seed_leakage": True,
            "no_policy_leakage": True,
            "validation_headroom_merged_into_training": False,
            "final_test_contexts_used": False,
        },
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    atomic_json(output_root / "final_offline_dataset_manifest.json", manifest)

    stats = {
        "schema_version": SCHEMA,
        "source_dataset_sha256": source_sha,
        "task_split_hash": task_split_hash,
        "split_task_counts": {split: len(tasks) for split, tasks in split_tasks.items()},
        "split_behavior_episode_counts": {
            split: outputs[split]["behavior"]["episode_count"] for split in SPLIT_NAMES
        },
        "split_behavior_transition_counts": {
            split: outputs[split]["behavior"]["transition_count"] for split in SPLIT_NAMES
        },
        "split_counterfactual_context_counts": {
            split: outputs[split]["initial_counterfactual"]["context_count"] for split in SPLIT_NAMES
        },
    }
    atomic_json(output_root / "final_offline_dataset_stats.json", stats)
    print(json.dumps({**stats, "manifest": str(output_root / "final_offline_dataset_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
