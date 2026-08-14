"""Reconstruct genuine deployable decision histories from portable snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd

from dacboenv.experiment.collect_snapshots import completed_evaluations, observation_digest, portable_observation_json
from dacboenv.experiment.real_env import real_headroom_train_env, real_headroom_validation_env
from dacboenv.experiment.snapshot_branch import BOSnapshot, CompletedBOEvaluation
from dacboenv.experiment.task_metadata import parse_task_metadata

HISTORY_LENGTH = 10


def snapshot_from_row(row: dict[str, object]) -> BOSnapshot:
    """Restore a typed portable snapshot from its Parquet representation."""
    allowed = {field.name for field in fields(BOSnapshot)}
    payload = {key: value for key, value in row.items() if key in allowed}
    actions = payload.get("action_history", ())
    payload["action_history"] = tuple(
        actions if isinstance(actions, (list, tuple, np.ndarray)) else json.loads(str(actions))
    )
    raw = row["completed_evaluations_json"]
    payload["completed_evaluations"] = tuple(CompletedBOEvaluation(**item) for item in json.loads(str(raw)))
    for key, value in tuple(payload.items()):
        if pd.isna(value) if not isinstance(value, (tuple, list, dict, np.ndarray)) else False:
            payload[key] = None
    return BOSnapshot(**payload)


def _runhistory_hash(env: object) -> str:
    records = [evaluation.__dict__ for evaluation in completed_evaluations(env)]
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def reconstruct(snapshot: BOSnapshot, split: str) -> dict[str, object]:
    """Replay one action prefix and retain exactly the last ten real states."""
    metadata = parse_task_metadata(snapshot.task_id)
    factory = real_headroom_train_env if split == "train" else real_headroom_validation_env
    env = factory(snapshot.task_id, snapshot.inner_seed, snapshot.action_space)
    observations, hashes, actions, rewards, evaluations, fractions = [], [], [], [], [], []
    try:
        observation, _info = env.reset()
        for action in snapshot.action_history:
            observations.append(portable_observation_json(observation))
            hashes.append(observation_digest(observation))
            actions.append(int(action))
            evaluations.append(int(env.get_n_finished_trials()))
            fractions.append(float(env.get_n_finished_trials()) / float(env._smac_instance._scenario.n_trials))
            observation, reward, terminated, truncated, _info = env.step(int(action))
            rewards.append(float(reward))
            if terminated or truncated:
                raise RuntimeError("Portable snapshot action prefix terminated before its recorded endpoint.")
        final_hash = observation_digest(env.get_observation())
        if final_hash != snapshot.observation_hash:
            raise RuntimeError(f"final observation hash mismatch: {final_hash} != {snapshot.observation_hash}")
        replayed = completed_evaluations(env)
        if replayed != snapshot.completed_evaluations:
            raise RuntimeError("completed BO evaluation history mismatch")
        if not np.isclose(float(env.get_incumbent_cost()), float(snapshot.incumbent), rtol=0, atol=1e-12):
            raise RuntimeError("incumbent mismatch")
        start = max(0, len(actions) - HISTORY_LENGTH)
        selected_observations = observations[start:]
        length = len(selected_observations)
        padding = HISTORY_LENGTH - length
        return {
            "snapshot_id": snapshot.snapshot_id,
            "trajectory_id": (
                f"{snapshot.task_id}|{snapshot.inner_seed}|{snapshot.history_policy}|{snapshot.action_space}"
            ),
            "split": split,
            "domain": metadata.domain,
            "scenario": metadata.scenario,
            "dataset_instance": metadata.dataset_instance,
            "sequence_length": length,
            "observation_sequence_json": json.dumps([None] * padding + selected_observations),
            "observation_hashes_json": json.dumps([None] * padding + hashes[start:]),
            "action_sequence_json": json.dumps([-1] * padding + actions[start:]),
            "reward_sequence_json": json.dumps([0.0] * padding + rewards[start:]),
            "evaluation_count_sequence_json": json.dumps([-1] * padding + evaluations[start:]),
            "budget_fraction_sequence_json": json.dumps([-1.0] * padding + fractions[start:]),
            "mask_json": json.dumps([0] * padding + [1] * length),
            "final_observation_hash": final_hash,
            "runhistory_hash": _runhistory_hash(env),
            "replay_status": "success",
            "source_revision": snapshot.code_commit,
            "manifest_hash": snapshot.source_manifest_hash,
        }
    finally:
        env.close()


def main() -> int:
    """Reconstruct one indexed row or consolidate all completed shards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = pd.read_parquet(args.snapshots)
    if args.consolidate:
        shards = sorted(args.output_root.glob(f"{args.split}-*.json"))
        payloads = [json.loads(path.read_text()) for path in shards]
        if len(payloads) != len(rows):
            raise RuntimeError(f"Expected {len(rows)} {args.split} shards, found {len(payloads)}.")
        pd.DataFrame(payloads).sort_values("campaign_snapshot_id").to_parquet(
            Path("artifacts/headroom_predictability_v3") / f"history_sequences_{args.split}.parquet", index=False
        )
        fingerprint_path = Path("artifacts/headroom_predictability_v3/history_replay_fingerprints.json")
        existing = json.loads(fingerprint_path.read_text()) if fingerprint_path.is_file() else {}
        existing[args.split] = {
            "requested": len(rows),
            "successful": len(payloads),
            "failed": 0,
            "full_ten_step_histories": sum(int(item["sequence_length"]) == HISTORY_LENGTH for item in payloads),
            "observation_hashes": {item["campaign_snapshot_id"]: item["final_observation_hash"] for item in payloads},
            "runhistory_hashes": {item["campaign_snapshot_id"]: item["runhistory_hash"] for item in payloads},
        }
        fingerprint_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        return 0
    if args.index is None or not 1 <= args.index <= len(rows):
        raise ValueError(f"--index must be in [1, {len(rows)}]")
    started = time.monotonic()
    source_row = rows.iloc[args.index - 1].to_dict()
    result = reconstruct(snapshot_from_row(source_row), args.split)
    result["campaign_snapshot_id"] = source_row["campaign_snapshot_id"]
    result["runtime_seconds"] = time.monotonic() - started
    (args.output_root / f"{args.split}-{args.index:04d}.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
