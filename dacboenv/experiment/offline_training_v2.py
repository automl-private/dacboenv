"""Training-manifest-only offline replay campaign for fixed-frequency WEI.

This campaign is intentionally separate from the broad evaluation-context v1
archive.  It uses the frozen BBOB/YAHPO *training* manifests, interaction
frequency five, all five static WEI levels, and a uniform-random WEI policy.
The resulting replay transitions can prefill DDQN or support fitted-Q/CQL
experiments without contaminating held-out evaluation tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.protocol import load_manifest, require_runnable_manifest
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.source_provenance import current_source_revision

SCHEMA_VERSION = "dacbo-offline-training-f5-v2"
OBSERVATION_KEYS = ("global_state", "action_features")
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


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


def _task_slug(task_id: str) -> str:
    return task_id.replace("/", "__").replace("None", "none")


def _scenario(task_id: str) -> str:
    parts = task_id.split("/")
    return parts[2] if task_id.startswith("yahpo/") else "bbob"


def _copy_observation(observation: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {key: np.asarray(observation[key], dtype=np.float32).copy() for key in OBSERVATION_KEYS}
    for key, value in result.items():
        if not np.isfinite(value).all():
            raise ValueError(f"Observation {key!r} contains non-finite values.")
    return result


def _policy_rows() -> list[dict[str, Any]]:
    policies = [
        {"policy_id": f"static_f5_alpha{index}", "kind": "static", "action": index, "alpha": ALPHAS[index]}
        for index in range(5)
    ]
    policies.append({"policy_id": "uniform_random_f5", "kind": "random", "action": None, "alpha": None})
    return policies


def build_inventory(repository: Path, output_root: Path, seeds: list[int]) -> dict[str, Any]:
    bbob_path = repository / "dacboenv/configs/instance_sets/bbob_train.yaml"
    yahpo_path = repository / "dacboenv/configs/instance_sets/yahpo_train.yaml"
    bbob = load_manifest(bbob_path)
    yahpo = load_manifest(yahpo_path)
    require_runnable_manifest(bbob)
    require_runnable_manifest(yahpo)
    if bbob["split"] != "train" or yahpo["split"] != "train":
        raise ValueError("Offline training v2 requires train manifests.")
    tasks = [str(value) for value in bbob["task_ids"]] + [str(value) for value in yahpo["task_ids"]]
    policies = _policy_rows()
    rows = []
    for policy in policies:
        for task_id in tasks:
            for seed in seeds:
                row = {
                    "job_index": len(rows),
                    "task_id": task_id,
                    "domain": "yahpo" if task_id.startswith("yahpo/") else "bbob",
                    "scenario": _scenario(task_id),
                    "seed": int(seed),
                    **policy,
                    "interaction_frequency": 5,
                    "context_split": "train",
                    "output_path": str(
                        output_root
                        / "episodes"
                        / str(policy["policy_id"])
                        / _task_slug(task_id)
                        / f"seed{seed}"
                        / "offline_episode.npz"
                    ),
                }
                row["job_hash"] = canonical_sha256(row)
                rows.append(row)
    reference = repository / "dacboenv/experiment/analysis/yahpo_best_known_references.json"
    scientific_manifest = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "training",
        "context_split": "train",
        "bbob_manifest_id": bbob["id"],
        "bbob_manifest_hash": bbob["manifest_hash"],
        "yahpo_manifest_id": yahpo["id"],
        "yahpo_manifest_hash": yahpo["manifest_hash"],
        "task_ids": tasks,
        "reference_table_sha256": file_sha256(reference),
        "observation_keys": list(OBSERVATION_KEYS),
        "action_grid": list(ALPHAS),
        "interaction_frequency": 5,
        "seeds": seeds,
        "policies": policies,
        "source_revision": current_source_revision(repository),
    }
    payload = {
        **scientific_manifest,
        "reference_table": str(reference.resolve()),
        "task_count": len(tasks),
        "job_count": len(rows),
        "jobs": rows,
    }
    payload["manifest_hash"] = canonical_sha256(scientific_manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "offline_training_manifest.json"
    if destination.exists():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != payload["manifest_hash"]:
            raise RuntimeError(f"Refusing to replace a different campaign at {destination}.")
    else:
        _atomic_json(destination, payload)
    return payload


def _policy_rng(row: dict[str, Any]) -> np.random.Generator:
    digest = canonical_sha256(
        {
            "stream": "offline_training_policy",
            "task_id": row["task_id"],
            "seed": row["seed"],
            "policy_id": row["policy_id"],
        }
    )
    return np.random.default_rng(int(digest[:16], 16))


def run_job(manifest_path: Path, job_index: int) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    row = manifest["jobs"][job_index]
    output = Path(row["output_path"])
    status_path = output.with_name("offline_episode_status.json")
    if output.exists():
        with np.load(output, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata_json"].item()))
        if metadata.get("job_hash") == row["job_hash"]:
            return {"status": "already_complete", "output": str(output), "job_index": job_index}
        raise RuntimeError(f"Another/corrupt shard exists: {output}")

    reference = Path(manifest["reference_table"])
    env = real_structured_mixed_env(
        str(row["task_id"]),
        int(row["seed"]),
        "wei",
        context_split="train",
        reference_table=reference if row["domain"] == "yahpo" else None,
        interaction_frequency=5,
    )
    rng = _policy_rng(row)
    records: list[dict[str, Any]] = []
    try:
        observation, info = env.reset()
        while True:
            before = int(env.get_n_finished_trials())
            current = _copy_observation(observation)
            if row["kind"] == "static":
                action = int(row["action"])
            else:
                action = int(rng.integers(5))
            observation, reward, terminated, truncated, step_info = env.step(action)
            after = int(env.get_n_finished_trials())
            records.append(
                {
                    "observation": current,
                    "next_observation": _copy_observation(observation),
                    "action": action,
                    "reward": float(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "before": before,
                    "after": after,
                }
            )
            if terminated or truncated:
                break
    except Exception as error:
        _atomic_json(
            status_path,
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed",
                "job_index": job_index,
                "job_hash": row["job_hash"],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            },
        )
        raise
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    arrays: dict[str, np.ndarray] = {}
    for key in OBSERVATION_KEYS:
        arrays[f"observations__{key}"] = np.stack([record["observation"][key] for record in records])
        arrays[f"next_observations__{key}"] = np.stack([record["next_observation"][key] for record in records])
    arrays.update(
        {
            "actions": np.asarray([record["action"] for record in records], dtype=np.int8),
            "action_alpha": np.asarray([ALPHAS[record["action"]] for record in records], dtype=np.float32),
            "behavior_probability": np.full(
                len(records), 1.0 if row["kind"] == "static" else 0.2, dtype=np.float32
            ),
            "behavior_log_probability": np.full(
                len(records), 0.0 if row["kind"] == "static" else np.log(0.2), dtype=np.float32
            ),
            "rewards": np.asarray([record["reward"] for record in records], dtype=np.float64),
            "terminated": np.asarray([record["terminated"] for record in records], dtype=np.bool_),
            "truncated": np.asarray([record["truncated"] for record in records], dtype=np.bool_),
            "bo_evaluations_before": np.asarray([record["before"] for record in records], dtype=np.int32),
            "bo_evaluations_after": np.asarray([record["after"] for record in records], dtype=np.int32),
            "realized_duration": np.asarray([record["after"] - record["before"] for record in records], dtype=np.int16),
        }
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "job_hash": row["job_hash"],
        "job_index": job_index,
        "task_id": row["task_id"],
        "domain": row["domain"],
        "scenario": row["scenario"],
        "seed": row["seed"],
        "policy_id": row["policy_id"],
        "policy_kind": row["kind"],
        "configured_action": row["action"],
        "interaction_frequency": 5,
        "transition_count": len(records),
        "bo_budget": int(env._n_trials),
        "initial_design_size": len(env._smac_instance.intensifier.config_selector._initial_design_configs),
        "behavior_probability_kind": "deterministic" if row["kind"] == "static" else "uniform_discrete",
        "data_role": "training",
        "context_split": "train",
        "observation_keys": list(OBSERVATION_KEYS),
        "reference_table_sha256": manifest["reference_table_sha256"],
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    _atomic_npz(output, arrays)
    _atomic_json(status_path, {"status": "success", **metadata, "episode_sha256": file_sha256(output)})
    return {"status": "success", "output": str(output), "transitions": len(records), "job_index": job_index}


def status(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = {"success": 0, "failed": 0, "missing": 0, "corrupt": 0}
    details = []
    for row in manifest["jobs"]:
        path = Path(row["output_path"])
        state = "missing"
        message = ""
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as payload:
                    metadata = json.loads(str(payload["metadata_json"].item()))
                    if metadata["job_hash"] != row["job_hash"]:
                        raise ValueError("job hash mismatch")
                    if payload["rewards"].shape[0] <= 0:
                        raise ValueError("empty transition array")
                state = "success"
            except Exception as error:  # noqa: BLE001
                state = "corrupt"
                message = f"{type(error).__name__}: {error}"
        else:
            status_path = path.with_name("offline_episode_status.json")
            if status_path.is_file():
                failed = json.loads(status_path.read_text(encoding="utf-8"))
                if failed.get("status") == "failed":
                    state = "failed"
                    message = str(failed.get("exception_message", ""))
        counts[state] += 1
        if state != "success":
            details.append({"job_index": row["job_index"], "state": state, "message": message})
    result = {"expected": manifest["job_count"], "counts": counts, "complete": counts["success"] == manifest["job_count"], "non_success": details}
    _atomic_json(manifest_path.parent / "offline_training_status.json", result)
    return result


def consolidate(manifest_path: Path, destination: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = status(manifest_path)
    if not audit["complete"]:
        raise RuntimeError(f"Offline training campaign is incomplete: {audit['counts']}")
    keys: list[str] | None = None
    chunks: dict[str, list[np.ndarray]] = {}
    episode_metadata = []
    episode_index = []
    episode_offsets = [0]
    for index, row in enumerate(manifest["jobs"]):
        path = Path(row["output_path"])
        with np.load(path, allow_pickle=False) as payload:
            current = sorted(key for key in payload.files if key != "metadata_json")
            if keys is None:
                keys = current
                chunks = {key: [] for key in keys}
            elif current != keys:
                raise ValueError(f"Shard schema mismatch: {path}")
            n = int(payload["rewards"].shape[0])
            for key in keys:
                chunks[key].append(np.asarray(payload[key]))
            episode_index.append(np.full(n, index, dtype=np.int32))
            episode_offsets.append(episode_offsets[-1] + n)
            episode_metadata.append(str(payload["metadata_json"].item()))
    arrays = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    arrays["episode_index"] = np.concatenate(episode_index)
    arrays["episode_offsets"] = np.asarray(episode_offsets, dtype=np.int64)
    width = max(map(len, episode_metadata))
    arrays["episode_metadata_json"] = np.asarray(episode_metadata, dtype=f"U{width}")
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "data_role": "training",
        "context_split": "train",
        "episode_count": len(episode_metadata),
        "transition_count": int(arrays["rewards"].shape[0]),
        "observation_keys": list(OBSERVATION_KEYS),
    }
    arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
    if destination.exists():
        raise FileExistsError(destination)
    _atomic_npz(destination, arrays)
    result = {**metadata, "dataset_path": str(destination), "dataset_sha256": file_sha256(destination), "dataset_bytes": destination.stat().st_size}
    _atomic_json(destination.with_suffix(".json"), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--repository", type=Path, default=Path.cwd())
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--seeds", default="0,1,2,3,4")
    worker = sub.add_parser("worker")
    worker.add_argument("--manifest", type=Path, required=True)
    worker.add_argument("--job-index", type=int, required=True)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--manifest", type=Path, required=True)
    consolidate_parser = sub.add_parser("consolidate")
    consolidate_parser.add_argument("--manifest", type=Path, required=True)
    consolidate_parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        seeds = [int(value) for value in args.seeds.split(",")]
        result = build_inventory(args.repository.resolve(), args.output_root.resolve(), seeds)
        print(json.dumps({key: value for key, value in result.items() if key != "jobs"}, indent=2))
    elif args.command == "worker":
        print(json.dumps(run_job(args.manifest.resolve(), args.job_index), indent=2))
    elif args.command == "status":
        print(json.dumps(status(args.manifest.resolve()), indent=2))
    else:
        print(json.dumps(consolidate(args.manifest.resolve(), args.destination.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
