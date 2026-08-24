"""Hydra manager for train/dev random-history same-state branch collection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.offline.schema import MIDRUN_BRANCH_SCHEMA_VERSION, ensure_no_object_arrays, validate_branch_arrays

BRANCH_EXECUTION_SEMANTICS_VERSION = "task-split-environment-v2"


def environment_context_split(data_context_split: str) -> str:
    """Map scientific data splits to supported environment context semantics."""
    try:
        return {"train": "train", "dev": "validation"}[data_context_split]
    except KeyError as error:
        raise ValueError(f"Offline branch jobs forbid data split {data_context_split!r}.") from error


if TYPE_CHECKING:
    from omegaconf import DictConfig


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(config: DictConfig) -> dict[str, Any]:
    """Freeze one job per task, seed, and phase for train and dev only."""
    final_root = Path(str(config.final_dataset_root)).resolve()
    output_root = Path(str(config.output_root)).resolve()
    final_manifest_path = final_root / "final_offline_dataset_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    phases = [float(item) for item in config.phases]
    seeds = [int(item) for item in config.seeds]
    source_policies = [str(item) for item in config.source_policies]
    if phases != [0.25, 0.5, 0.75] or seeds != [0, 1, 2, 3, 4]:
        raise ValueError("Primary offline branch campaign requires phases .25/.5/.75 and seeds 0..4.")
    allowed = {"uniform_random", "static_0", "static_2", "static_4"}
    if not source_policies or not set(source_policies) <= allowed:
        raise ValueError(f"Unsupported branch history source: {source_policies}.")
    jobs: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        for task_id in final_manifest["task_splits"][split]:
            for seed in seeds:
                for phase in phases:
                    for source_policy in source_policies:
                        environment_split = environment_context_split(split)
                        scientific = {
                            "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
                            "branch_execution_semantics": BRANCH_EXECUTION_SEMANTICS_VERSION,
                            "final_manifest_hash": final_manifest["manifest_hash"],
                            "context_split": split,
                            "data_context_split": split,
                            "environment_context_split": environment_split,
                            "task_id": task_id,
                            "seed": seed,
                            "phase": phase,
                            "source_policy": source_policy,
                            "actions": [0, 1, 2, 3, 4],
                            "maximum_horizon": 10,
                            "prefix_horizons": [5, 10],
                            "interaction_frequency": 5,
                        }
                        index = len(jobs)
                        jobs.append(
                            {
                                "job_index": index,
                                "job_hash": canonical_sha256(scientific),
                                **scientific,
                                "output_path": str(output_root / "jobs" / f"{index:05d}.json"),
                            }
                        )
    manifest = {
        "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
        "final_dataset_root": str(final_root),
        "final_manifest_path": str(final_manifest_path),
        "final_manifest_sha256": file_sha256(final_manifest_path),
        "final_manifest_hash": final_manifest["manifest_hash"],
        "job_count": len(jobs),
        "expected_primary_counts": {"train": 780, "dev": 210},
        "jobs": jobs,
    }
    manifest["manifest_hash"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "jobs"}
        | {"job_hashes": [row["job_hash"] for row in jobs]}
    )
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "offline_branch_job_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != manifest["manifest_hash"]:
            raise RuntimeError(f"Refusing to replace a different branch campaign at {path}.")
    else:
        _atomic_json(path, manifest)
    return manifest


def status(config: DictConfig) -> dict[str, Any]:
    """Audit every expected atomic job shard."""
    root = Path(str(config.output_root)).resolve()
    manifest = json.loads((root / "offline_branch_job_manifest.json").read_text(encoding="utf-8"))
    counts = {"success": 0, "failed": 0, "missing": 0, "corrupt": 0}
    missing_indices = []
    for row in manifest["jobs"]:
        path = Path(row["output_path"])
        state = "missing"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "success" or payload.get("job_hash") != row["job_hash"]:
                    raise ValueError("identity/status mismatch")
                if not payload.get("all_actions_share_source") or not payload.get("reward_telescoping_valid"):
                    raise ValueError("branch scientific checks failed")
                state = "success"
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                state = "corrupt"
        elif path.with_suffix(".failed.json").is_file():
            state = "failed"
        counts[state] += 1
        if state != "success":
            missing_indices.append(int(row["job_index"]))
    result = {
        "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "expected": len(manifest["jobs"]),
        "counts": counts,
        "complete": counts["success"] == len(manifest["jobs"]),
        "missing_indices": missing_indices,
    }
    _atomic_json(root / "offline_branch_status.json", result)
    return result


def consolidate(config: DictConfig) -> dict[str, Any]:
    """Consolidate verified JSON branches into train/dev NPZ datasets."""
    root = Path(str(config.output_root)).resolve()
    audit = status(config)
    if not audit["complete"]:
        raise RuntimeError(f"Branch campaign is incomplete: {audit['counts']}.")
    manifest = json.loads((root / "offline_branch_job_manifest.json").read_text(encoding="utf-8"))
    outputs = {}
    for split in ("train", "dev"):
        payloads = [
            json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))
            for row in manifest["jobs"]
            if row["data_context_split"] == split
        ]
        arrays: dict[str, np.ndarray] = {}
        state_keys = (
            "global_state",
            "action_features",
            "q5",
            "q10",
            "valid_action_mask",
            "tie_mask_q5",
            "tie_mask_q10",
            "top1_top2_gap_q5",
            "top1_top2_gap_q10",
            "domain_id",
            "scenario_id",
            "phase_bin",
            "seed",
        )
        for key in state_keys:
            arrays[key] = np.asarray([payload["branch_record"][key] for payload in payloads])
        arrays["global_state"] = arrays["global_state"].astype(np.float32)
        arrays["action_features"] = arrays["action_features"].astype(np.float32)
        arrays["q5"] = arrays["q5"].astype(np.float64)
        arrays["q10"] = arrays["q10"].astype(np.float64)
        for key in ("valid_action_mask", "tie_mask_q5", "tie_mask_q10"):
            arrays[key] = arrays[key].astype(np.bool_)
        for key in ("top1_top2_gap_q5", "top1_top2_gap_q10"):
            arrays[key] = arrays[key].astype(np.float64)
        arrays["domain_id"] = arrays["domain_id"].astype(np.int8)
        arrays["scenario_id"] = arrays["scenario_id"].astype(np.int8)
        arrays["phase_bin"] = arrays["phase_bin"].astype(np.int8)
        arrays["seed"] = arrays["seed"].astype(np.int32)
        arrays["action_alpha"] = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
        for key in (
            "task_id",
            "source_policy_id",
            "source_state_digest",
            "source_replay_digest",
            "candidate_duplicate_groups",
            "branch_protocol_hash",
            "data_context_split",
            "environment_context_split",
        ):
            values = [str(payload["branch_record"][key]) for payload in payloads]
            arrays[key] = np.asarray(values, dtype=f"U{max(map(len, values), default=1)}")
        reference_values = [
            json.dumps(payload["branch_record"]["reference_metadata"], separators=(",", ":"), sort_keys=True)
            for payload in payloads
        ]
        arrays["reference_metadata_json"] = np.asarray(
            reference_values,
            dtype=f"U{max(map(len, reference_values), default=1)}",
        )
        metadata = {
            "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
            "component": "midrun_same_state_q5_q10",
            "split": split,
            "context_split": split,
            "environment_context_split": "train" if split == "train" else "validation",
            "branch_execution_semantics": BRANCH_EXECUTION_SEMANTICS_VERSION,
            "data_role": "offline_training" if split == "train" else "offline_development",
            "manifest_hash": manifest["manifest_hash"],
            "state_count": len(payloads),
            "source_policy": sorted({payload["branch_record"]["source_policy_id"] for payload in payloads}),
            "q5_semantics": "fixed_action_normalized_potential_improvement_over_next_5_bo_evaluations",
            "q10_semantics": "fixed_action_normalized_potential_improvement_over_next_10_bo_evaluations",
            "not_long_return_q": True,
        }
        arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        ensure_no_object_arrays(arrays)
        validate_branch_arrays(arrays)
        destination = root / f"branches_{split}.npz"
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        temporary.replace(destination)
        outputs[split] = {"path": str(destination), "sha256": file_sha256(destination), "states": len(payloads)}
    result = {
        "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
        "manifest_hash": manifest["manifest_hash"],
        "outputs": outputs,
    }
    _atomic_json(root / "offline_branch_consolidation.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs/offline_branch_campaign", config_name="base")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Dispatch the requested management operation."""
    operation = str(config.operation)
    if operation == "prepare":
        result = prepare(config)
    elif operation == "status":
        result = status(config)
    elif operation == "consolidate":
        result = consolidate(config)
    else:
        raise ValueError(f"Unknown offline branch operation {operation!r}.")
    printable = {key: value for key, value in result.items() if key not in {"jobs"}}
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
