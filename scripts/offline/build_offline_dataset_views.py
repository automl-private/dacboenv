"""Audit a consolidated DACBO offline dataset and derive compact H=5/H=10 views.

The supplied v1 campaign was collected on broad BBOB-2D/8D and official YAHPO
SO evaluation contexts.  Those data are valuable for diagnostics and engineering,
but they must not be used to train a method that is later evaluated on the same
contexts.  This utility therefore requires an explicit acknowledgement and marks
all derived files as diagnostic-only unless ``--data-role training`` is supplied
for a separately collected training-manifest dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

COMPACT_KEYS = ("global_state", "action_features", "gp_hp_summary", "gp_hp_change")
HORIZONS = (1, 5, 10)
STATIC_POLICY_PREFIX = "static_wei_alpha"
RANDOM_POLICY_ID = "double_random_wei_tempo"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario(task_id: str) -> str:
    parts = task_id.split("/")
    return parts[2] if len(parts) >= 4 and parts[0].lower() == "yahpo" else "bbob"


def _policy_alpha_index(policy_id: str) -> int:
    suffix = policy_id.removeprefix(STATIC_POLICY_PREFIX)
    mapping = {"000": 0, "025": 1, "050": 2, "075": 3, "100": 4}
    try:
        return mapping[suffix]
    except KeyError as error:
        raise ValueError(f"Unknown static policy ID {policy_id!r}.") from error


def _unicode(values: list[str]) -> np.ndarray:
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"U{width}")


def _episode_rows(payload: Any) -> list[dict[str, Any]]:
    return [json.loads(str(value)) for value in payload["episode_metadata_json"]]


def _dataset_audit(payload: Any, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray(payload["rewards"], dtype=np.float64)
    requested = np.asarray(payload["requested_duration"], dtype=np.int64)
    realized = np.asarray(payload["realized_duration"], dtype=np.int64)
    action_indices = np.asarray(payload["action_alpha_index"], dtype=np.int64)
    episode_counts = Counter(str(row["policy_id"]) for row in episodes)
    transition_counts: Counter[str] = Counter()
    domain_transitions: Counter[str] = Counter()
    offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
    for episode_index, row in enumerate(episodes):
        count = int(offsets[episode_index + 1] - offsets[episode_index])
        transition_counts[str(row["policy_id"])] += count
        domain_transitions[str(row["domain"])] += count

    action_features = np.asarray(payload["observations__action_features"], dtype=np.float32)
    consequence_rows = action_features[:, :, 1:]
    unique_counts = []
    for rows in consequence_rows:
        unique_counts.append(len({np.asarray(row).tobytes() for row in rows}))

    gp_summary = np.asarray(payload["observations__gp_hp_summary"], dtype=np.float32)
    gp_change = np.asarray(payload["observations__gp_hp_change"], dtype=np.float32)
    return {
        "schema_version": str(json.loads(str(payload["dataset_metadata_json"].item()))["schema_version"]),
        "transition_count": int(rewards.size),
        "episode_count": len(episodes),
        "task_count": len({str(row["task_id"]) for row in episodes}),
        "policy_count": len(episode_counts),
        "seed_count": len({int(row["seed"]) for row in episodes}),
        "episode_counts_by_policy": dict(sorted(episode_counts.items())),
        "transition_counts_by_policy": dict(sorted(transition_counts.items())),
        "transition_counts_by_domain": dict(sorted(domain_transitions.items())),
        "requested_duration_counts": {str(k): int(v) for k, v in sorted(Counter(requested.tolist()).items())},
        "realized_duration_counts": {str(k): int(v) for k, v in sorted(Counter(realized.tolist()).items())},
        "action_index_counts": {str(k): int(v) for k, v in sorted(Counter(action_indices.tolist()).items())},
        "reward": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "zero_fraction": float(np.mean(rewards == 0.0)),
            "positive_fraction": float(np.mean(rewards > 0.0)),
            "q50": float(np.quantile(rewards, 0.50)),
            "q95": float(np.quantile(rewards, 0.95)),
            "q99": float(np.quantile(rewards, 0.99)),
            "maximum": float(np.max(rewards)),
        },
        "action_features": {
            "mean_unique_consequence_rows": float(np.mean(unique_counts)),
            "mean_duplicate_fraction": float(np.mean(1.0 - np.asarray(unique_counts, dtype=float) / 5.0)),
            "alpha_column_exact": bool(
                np.array_equal(
                    action_features[:, :, 0],
                    np.broadcast_to(np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32), action_features[:, :, 0].shape),
                )
            ),
        },
        "gp": {
            "availability_rate": float(np.mean(gp_summary[:, 0])),
            "previous_available_rate": float(np.mean(gp_change[:, 0])),
        },
    }


def _block_replay(
    payload: Any,
    episodes: list[dict[str, Any]],
    horizon: int,
    *,
    include_direct_random: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
    rewards = np.asarray(payload["rewards"], dtype=np.float64)
    terminals = np.asarray(payload["terminals"], dtype=np.bool_)
    requested = np.asarray(payload["requested_duration"], dtype=np.int64)
    realized = np.asarray(payload["realized_duration"], dtype=np.int64)
    bo_before = np.asarray(payload["bo_evaluations_before"], dtype=np.int64)
    bo_after = np.asarray(payload["bo_evaluations_after"], dtype=np.int64)
    actions = np.asarray(payload["action_alpha_index"], dtype=np.int64)

    output: dict[str, list[np.ndarray | float | int | str | bool]] = defaultdict(list)
    static_blocks = 0
    random_blocks = 0

    def append(start: int, end: int, episode_index: int, source: str, action_index: int) -> None:
        for key in COMPACT_KEYS:
            output[f"observations__{key}"].append(np.asarray(payload[f"observations__{key}"][start]).copy())
            output[f"next_observations__{key}"].append(np.asarray(payload[f"next_observations__{key}"][end - 1]).copy())
        row = episodes[episode_index]
        output["actions"].append(action_index)
        output["rewards"].append(float(np.sum(rewards[start:end])))
        output["terminals"].append(bool(terminals[end - 1]))
        output["timeouts"].append(False)
        output["requested_duration"].append(horizon)
        output["realized_duration"].append(int(bo_after[end - 1] - bo_before[start]))
        output["bo_evaluations_before"].append(int(bo_before[start]))
        output["bo_evaluations_after"].append(int(bo_after[end - 1]))
        output["source_episode_index"].append(episode_index)
        output["source_policy_id"].append(str(row["policy_id"]))
        output["source_kind"].append(source)
        output["task_id"].append(str(row["task_id"]))
        output["domain"].append(str(row["domain"]))
        output["scenario"].append(_scenario(str(row["task_id"])))
        output["seed"].append(int(row["seed"]))

    for episode_index, row in enumerate(episodes):
        start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
        policy_id = str(row["policy_id"])
        if policy_id.startswith(STATIC_POLICY_PREFIX):
            action_index = _policy_alpha_index(policy_id)
            if not np.all(requested[start:stop] == 1) or not np.all(realized[start:stop] == 1):
                raise ValueError(f"Static f1 episode has non-unit duration: {policy_id} {row['task_id']}.")
            if not np.all(actions[start:stop] == action_index):
                raise ValueError(f"Static episode action mismatch: {policy_id} {row['task_id']}.")
            for block_start in range(start, stop, horizon):
                block_end = block_start + horizon
                if block_end > stop:
                    break
                append(block_start, block_end, episode_index, "static_f1_aggregated", action_index)
                static_blocks += 1
        elif include_direct_random and policy_id == RANDOM_POLICY_ID:
            indices = np.arange(start, stop)
            selected = indices[(requested[start:stop] == horizon) & (realized[start:stop] == horizon)]
            for index in selected:
                append(int(index), int(index + 1), episode_index, f"random_direct_f{horizon}", int(actions[index]))
                random_blocks += 1

    arrays: dict[str, np.ndarray] = {}
    for key, values in output.items():
        if key.startswith("observations__") or key.startswith("next_observations__"):
            arrays[key] = np.stack(values).astype(np.float32, copy=False)
        elif key in {"rewards"}:
            arrays[key] = np.asarray(values, dtype=np.float64)
        elif key in {"terminals", "timeouts"}:
            arrays[key] = np.asarray(values, dtype=np.bool_)
        elif key in {"source_policy_id", "source_kind", "task_id", "domain", "scenario"}:
            arrays[key] = _unicode([str(value) for value in values])
        else:
            arrays[key] = np.asarray(values, dtype=np.int64)

    # Persist a real length-10 decision history for every aggregated static
    # block. This fixes the old portable-snapshot limitation where nine GRU
    # positions were only masked sentinels. Direct random TempoRL rows are not
    # contiguous after selecting one duration, so their history availability is
    # explicitly zero rather than fabricated.
    history_length = 10
    n_rows = int(arrays["rewards"].shape[0])
    arrays["history_global_state"] = np.zeros(
        (n_rows, history_length, *arrays["observations__global_state"].shape[1:]), dtype=np.float32
    )
    arrays["history_action_features"] = np.zeros(
        (n_rows, history_length, *arrays["observations__action_features"].shape[1:]), dtype=np.float32
    )
    arrays["history_actions"] = np.full((n_rows, history_length), -1, dtype=np.int8)
    arrays["history_rewards"] = np.zeros((n_rows, history_length), dtype=np.float64)
    arrays["history_mask"] = np.zeros((n_rows, history_length), dtype=np.bool_)
    arrays["history_available"] = np.zeros(n_rows, dtype=np.bool_)
    by_episode: dict[int, list[int]] = defaultdict(list)
    for row_index, episode_index in enumerate(arrays["source_episode_index"]):
        if str(arrays["source_kind"][row_index]) == "static_f1_aggregated":
            by_episode[int(episode_index)].append(row_index)
    for indices in by_episode.values():
        for local_index, row_index in enumerate(indices):
            previous = indices[max(0, local_index - history_length) : local_index]
            if previous:
                destination = slice(history_length - len(previous), history_length)
                arrays["history_global_state"][row_index, destination] = arrays["observations__global_state"][previous]
                arrays["history_action_features"][row_index, destination] = arrays["observations__action_features"][previous]
                arrays["history_actions"][row_index, destination] = arrays["actions"][previous].astype(np.int8)
                arrays["history_rewards"][row_index, destination] = arrays["rewards"][previous]
                arrays["history_mask"][row_index, destination] = True
            arrays["history_available"][row_index] = True
    metadata = {
        "schema_version": "dacbo-offline-derived-blocks-v1",
        "horizon": horizon,
        "transition_count": int(len(output["rewards"])),
        "static_aggregated_count": static_blocks,
        "direct_random_count": random_blocks,
        "observation_keys": list(COMPACT_KEYS),
        "reward_definition": "sum of original telescoping rewards over underlying BO evaluations",
        "source_unit": "static f1 nonoverlapping blocks plus directly collected random duration blocks",
        "history_length": 10,
        "history_semantics": "previous decision blocks only; right-aligned; direct random rows unavailable",
    }
    arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return arrays, metadata


def _initial_counterfactuals(payload: Any, episodes: list[dict[str, Any]]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    offsets = np.asarray(payload["episode_offsets"], dtype=np.int64)
    rewards = np.asarray(payload["rewards"], dtype=np.float64)
    groups: dict[tuple[str, int], dict[int, int]] = defaultdict(dict)
    for episode_index, row in enumerate(episodes):
        policy_id = str(row["policy_id"])
        if policy_id.startswith(STATIC_POLICY_PREFIX):
            groups[(str(row["task_id"]), int(row["seed"]))][_policy_alpha_index(policy_id)] = episode_index

    task_ids: list[str] = []
    domains: list[str] = []
    scenarios: list[str] = []
    seeds: list[int] = []
    obs: dict[str, list[np.ndarray]] = {key: [] for key in COMPACT_KEYS}
    q = np.zeros((len(groups), len(HORIZONS), 5), dtype=np.float64)
    next_obs: dict[tuple[int, str], list[np.ndarray]] = {
        (horizon, key): [] for horizon in (5, 10) for key in COMPACT_KEYS
    }
    gaps = np.zeros((len(groups), len(HORIZONS)), dtype=np.float64)
    oracle_actions = np.zeros((len(groups), len(HORIZONS)), dtype=np.int8)

    for group_index, ((task_id, seed), policies) in enumerate(sorted(groups.items())):
        if set(policies) != set(range(5)):
            raise ValueError(f"Incomplete static action group for {(task_id, seed)}: {sorted(policies)}")
        first_observations = []
        for action_index in range(5):
            episode_index = policies[action_index]
            start, stop = int(offsets[episode_index]), int(offsets[episode_index + 1])
            first_observations.append(np.asarray(payload["observations__global_state"][start]))
            for horizon_index, horizon in enumerate(HORIZONS):
                if stop - start < horizon:
                    raise ValueError(f"Episode shorter than horizon {horizon}: {(task_id, seed, action_index)}")
                q[group_index, horizon_index, action_index] = float(np.sum(rewards[start : start + horizon]))
                if horizon in (5, 10):
                    for key in COMPACT_KEYS:
                        next_obs[(horizon, key)].append(
                            np.asarray(payload[f"next_observations__{key}"][start + horizon - 1]).copy()
                        )
        if any(not np.array_equal(first_observations[0], item) for item in first_observations[1:]):
            raise RuntimeError(f"Static policies do not share the same initial state for {(task_id, seed)}.")
        first_episode = policies[0]
        first_start = int(offsets[first_episode])
        for key in COMPACT_KEYS:
            obs[key].append(np.asarray(payload[f"observations__{key}"][first_start]).copy())
        ordered = np.sort(q[group_index], axis=1)
        gaps[group_index] = ordered[:, -1] - ordered[:, -2]
        oracle_actions[group_index] = np.argmax(q[group_index], axis=1)
        task_ids.append(task_id)
        domains.append("yahpo" if task_id.startswith("yahpo/") else "bbob")
        scenarios.append(_scenario(task_id))
        seeds.append(seed)

    arrays: dict[str, np.ndarray] = {
        **{f"observations__{key}": np.stack(values).astype(np.float32, copy=False) for key, values in obs.items()},
        "q_values": q,
        "horizons": np.asarray(HORIZONS, dtype=np.int16),
        "action_alpha": np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32),
        "oracle_action": oracle_actions,
        "top1_top2_gap": gaps,
        "task_id": _unicode(task_ids),
        "domain": _unicode(domains),
        "scenario": _unicode(scenarios),
        "seed": np.asarray(seeds, dtype=np.int64),
    }
    for (horizon, key), values in next_obs.items():
        shape = (len(groups), 5, *np.asarray(values[0]).shape)
        arrays[f"next_observations_h{horizon}__{key}"] = np.asarray(values, dtype=np.float32).reshape(shape)
    metadata = {
        "schema_version": "dacbo-offline-initial-counterfactuals-v1",
        "state_count": len(groups),
        "actions_per_state": 5,
        "horizons": list(HORIZONS),
        "observation_keys": list(COMPACT_KEYS),
        "construction": "five static trajectories paired only at their bit-identical first model-based state",
    }
    arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    return arrays, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--data-role",
        choices=("diagnostic_evaluation", "training"),
        default="diagnostic_evaluation",
    )
    parser.add_argument("--acknowledge-evaluation-contexts", action="store_true")
    args = parser.parse_args()
    if args.data_role == "diagnostic_evaluation" and not args.acknowledge_evaluation_contexts:
        raise SystemExit(
            "This dataset was collected on broad evaluation/test contexts. Re-run with "
            "--acknowledge-evaluation-contexts to create diagnostic-only views."
        )
    dataset = args.dataset.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = {
        "dataset_metadata_json",
        "episode_metadata_json",
        "episode_offsets",
        "rewards",
        "requested_duration",
        "realized_duration",
        "action_alpha_index",
        "terminals",
        "bo_evaluations_before",
        "bo_evaluations_after",
        *(f"observations__{key}" for key in COMPACT_KEYS),
        *(f"next_observations__{key}" for key in COMPACT_KEYS),
    }
    # NPZ members are individually compressed. Materialize every required
    # member once so repeated random indexing does not repeatedly decompress
    # the same zip member. The raw 64-parameter GP arrays are intentionally not
    # loaded for these compact views.
    with np.load(dataset, allow_pickle=False) as source:
        missing = sorted(required - set(source.files))
        if missing:
            raise ValueError(f"Offline dataset is missing required arrays: {missing}")
        payload = {key: np.asarray(source[key]) for key in required}
    episodes = _episode_rows(payload)
    audit = _dataset_audit(payload, episodes)
    f5, f5_meta = _block_replay(payload, episodes, 5, include_direct_random=True)
    f10, f10_meta = _block_replay(payload, episodes, 10, include_direct_random=True)
    counterfactuals, cf_meta = _initial_counterfactuals(payload, episodes)

    provenance = {
        "source_dataset": str(dataset),
        "source_dataset_sha256": _sha256(dataset),
        "data_role": args.data_role,
        "scientific_use": (
            "diagnostic_only_not_for_final_training" if args.data_role == "diagnostic_evaluation" else "training"
        ),
    }
    for arrays, metadata, name in (
        (f5, f5_meta, "compact_f5_replay.npz"),
        (f10, f10_meta, "compact_f10_replay.npz"),
        (counterfactuals, cf_meta, "initial_state_counterfactuals.npz"),
    ):
        metadata.update(provenance)
        arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        _atomic_npz(output / name, arrays)

    audit.update(provenance)
    audit["derived_views"] = {
        "compact_f5_replay": f5_meta,
        "compact_f10_replay": f10_meta,
        "initial_state_counterfactuals": cf_meta,
    }
    _atomic_json(output / "offline_dataset_audit.json", audit)

    with (output / "derived_view_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("view", "transitions_or_states", "notes"))
        writer.writeheader()
        writer.writerow({"view": "compact_f5_replay", "transitions_or_states": f5_meta["transition_count"], "notes": "static H5 blocks plus direct random H5"})
        writer.writerow({"view": "compact_f10_replay", "transitions_or_states": f10_meta["transition_count"], "notes": "static H10 blocks plus direct random H10"})
        writer.writerow({"view": "initial_state_counterfactuals", "transitions_or_states": cf_meta["state_count"], "notes": "all five actions at one identical initial state"})
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
