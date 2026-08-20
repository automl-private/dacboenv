"""Hydra management workflow for the CARP-S offline dataset campaign."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import carps
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.offline_dataset import (
    OFFLINE_DATASET_SCHEMA_VERSION,
    OFFLINE_OBSERVATION_KEYS,
    validate_episode_npz,
)
from dacboenv.experiment.source_provenance import current_source_revision

POLICY_COUNT = 7
TASK_CONFIG_PATTERN = re.compile(r"cfg_(\d+)_(\d+)_(\d+)\.yaml")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_revision() -> tuple[str, str, bool]:
    repository = Path(__file__).resolve().parents[2]
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("git is required to freeze offline dataset provenance.")
    revision = subprocess.run(  # noqa: S603
        [git, "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(  # noqa: S603
            [git, "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return revision, current_source_revision(repository), dirty


def _task_root() -> Path:
    return Path(carps.__file__).resolve().parent / "configs" / "task"


def _read_task(path: Path, task_group: str) -> dict[str, Any]:
    config = OmegaConf.load(path)
    return {
        "task_group": task_group,
        "config_name": path.stem,
        "task_id": str(config.task.name),
        "benchmark_id": str(config.benchmark_id),
        "native_n_trials": int(config.task.optimization_resources.n_trials),
        "task_config_path": str(path.resolve()),
        "task_config_sha256": file_sha256(path),
    }


def discover_tasks(config: DictConfig) -> list[dict[str, Any]]:
    """Resolve the frozen BBOB and installed packaged YAHPO task inventory."""
    root = _task_root()
    tasks: list[dict[str, Any]] = []
    for dimension in map(int, config.bbob_dimensions):
        for function_id in map(int, config.bbob_functions):
            for instance in map(int, config.bbob_instances):
                path = root / "BBOB" / f"cfg_{dimension}_{function_id}_{instance}.yaml"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing packaged CARP-S BBOB task config: {path}")
                match = TASK_CONFIG_PATTERN.fullmatch(path.name)
                if match is None:
                    raise ValueError(f"Unexpected BBOB task filename: {path.name}")
                tasks.append(_read_task(path, "BBOB"))

    if not bool(config.include_all_packaged_yahpo_so):
        raise ValueError("The v1 campaign requires all packaged YAHPO/SO tasks.")
    yahpo_paths = sorted((root / "YAHPO" / "SO").glob("*.yaml"), key=lambda path: path.name)
    if len(yahpo_paths) != 20:  # noqa: PLR2004
        raise RuntimeError(f"Expected 20 packaged YAHPO/SO tasks, found {len(yahpo_paths)}.")
    tasks.extend(_read_task(path, "YAHPO/SO") for path in yahpo_paths)
    task_ids = [task["task_id"] for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise RuntimeError("Offline task inventory contains duplicate task IDs.")
    return tasks


def build_policies(config: DictConfig) -> list[dict[str, Any]]:
    """Create five static, one double-random, and one SAWEI policy row."""
    alphas = [float(value) for value in config.alpha_levels]
    durations = [int(value) for value in config.durations]
    if alphas != [0.0, 0.25, 0.5, 0.75, 1.0] or durations != [1, 5, 10]:
        raise ValueError("The v1 action schema requires alpha=[0,.25,.5,.75,1] and duration=[1,5,10].")

    policies: list[dict[str, Any]] = []
    for alpha_index, alpha in enumerate(alphas):
        policy_id = f"static_wei_alpha{round(100 * alpha):03d}"
        policies.append(
            {
                "policy_id": policy_id,
                "policy_kind": "static",
                "alpha": alpha,
                "duration": 1,
                "action_config": "wei_alpha_discrete",
                "interaction_config": "f1",
                "observation_config": "structured_gp_all",
                "policy_override": f"+policy/static/discrete_action=action_{alpha_index}",
                "extra_overrides": [],
            }
        )

    policies.append(
        {
            "policy_id": "double_random_wei_tempo",
            "policy_kind": "double_random",
            "alpha": None,
            "duration": None,
            "action_config": "wei_alpha_skip",
            "interaction_config": "f1",
            "observation_config": "structured_gp_all",
            "policy_override": "+policy=double_random",
            "extra_overrides": [],
        }
    )
    sawei = config.sawei
    policies.append(
        {
            "policy_id": "sawei_native_duration1",
            "policy_kind": "sawei",
            "alpha": None,
            "duration": 1,
            "action_config": "wei_alpha_continuous",
            "interaction_config": "f1",
            "observation_config": "structured_gp_all_sawei",
            "policy_override": "+policy=sawei",
            "extra_overrides": [
                f"++optimizer.policy_kwargs.alpha={float(sawei.alpha)}",
                f"++optimizer.policy_kwargs.delta={float(sawei.delta)}",
                f"++optimizer.policy_kwargs.window_size={int(sawei.window_size)}",
                f"++optimizer.policy_kwargs.atol_rel={float(sawei.atol_rel)}",
                f"++optimizer.policy_kwargs.track_attitude={sawei.track_attitude!s}",
                f"++optimizer.policy_kwargs.bounds=[{float(sawei.bounds[0])},{float(sawei.bounds[1])}]",
                f"++optimizer.policy_kwargs.auto_alpha={str(bool(sawei.auto_alpha)).lower()}",
            ],
        }
    )
    if len(policies) != POLICY_COUNT or len({policy["policy_id"] for policy in policies}) != POLICY_COUNT:
        raise RuntimeError("The offline policy registry must contain exactly seven unique policies.")
    return policies


def _episode_path(output_root: Path, policy_id: str, task: dict[str, Any], seed: int) -> Path:
    return output_root / "runs" / policy_id / task["benchmark_id"] / task["task_id"] / str(seed) / "offline_episode.npz"


def build_inventory(config: DictConfig) -> dict[str, Any]:
    """Freeze all scientific episode rows and Hydra launch groups."""
    if str(config.schema_version) != OFFLINE_DATASET_SCHEMA_VERSION:
        raise ValueError("Campaign and writer schema versions do not match.")
    if tuple(config.observation_keys) != OFFLINE_OBSERVATION_KEYS:
        raise ValueError("Campaign observation keys do not match the fixed v1 schema.")
    reference_table = Path(str(config.reference_table)).resolve()
    if not reference_table.is_file():
        raise FileNotFoundError(reference_table)
    output_root = Path(str(config.output_root)).resolve()
    tasks = discover_tasks(config)
    policies = build_policies(config)
    seeds = [int(seed) for seed in config.seeds]
    commit, revision, dirty = _git_revision()
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for task in tasks:
            for seed in seeds:
                scientific = {
                    "schema_version": str(config.schema_version),
                    "task_id": task["task_id"],
                    "task_config_sha256": task["task_config_sha256"],
                    "seed": seed,
                    "policy": policy,
                    "observation_keys": list(config.observation_keys),
                    "reference_table_sha256": file_sha256(reference_table),
                    "source_revision": revision,
                }
                rows.append(
                    {
                        "job_index": len(rows) + 1,
                        "scientific_id": canonical_sha256(scientific),
                        **task,
                        "seed": seed,
                        "policy_id": policy["policy_id"],
                        "policy_kind": policy["policy_kind"],
                        "alpha": policy["alpha"],
                        "duration": policy["duration"],
                        "output_path": str(_episode_path(output_root, policy["policy_id"], task, seed)),
                    }
                )

    expected_jobs = len(tasks) * len(seeds) * len(policies)
    if len(rows) != expected_jobs:
        raise RuntimeError("Offline inventory size is inconsistent.")
    scientific_manifest = {
        "schema_version": str(config.schema_version),
        "source_revision": revision,
        "source_commit": commit,
        "carps_version": version("carps"),
        "smac_version": version("smac"),
        "reference_table_sha256": file_sha256(reference_table),
        "observation_keys": list(config.observation_keys),
        "seeds": seeds,
        "tasks": tasks,
        "policies": policies,
        "scientific_ids": [row["scientific_id"] for row in rows],
    }
    manifest_hash = canonical_sha256(scientific_manifest)
    payload = {
        **scientific_manifest,
        "campaign_id": "carps-offline-bbob2d8d-yahpo-so-v1",
        "created_at": _utc_now(),
        "source_worktree_dirty": dirty,
        "manifest_hash": manifest_hash,
        "output_root": str(output_root),
        "reference_table": str(reference_table),
        "task_count": len(tasks),
        "bbob_task_count": sum(task["task_group"] == "BBOB" for task in tasks),
        "yahpo_task_count": sum(task["task_group"] == "YAHPO/SO" for task in tasks),
        "policy_count": len(policies),
        "seed_count": len(seeds),
        "job_count": len(rows),
        "native_budgets_preserved": True,
        "rows": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "inventory.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != manifest_hash:
            raise RuntimeError(f"Refusing to replace a different offline inventory at {destination}.")
    else:
        _atomic_json(destination, payload)
    _write_launch_groups(output_root / "launch_groups.tsv", tasks, policies)
    return payload


def _write_launch_groups(path: Path, tasks: list[dict[str, Any]], policies: list[dict[str, Any]]) -> None:
    task_groups: dict[str, list[str]] = {"BBOB": [], "YAHPO/SO": []}
    for task in tasks:
        task_groups[task["task_group"]].append(task["config_name"])
    rows = []
    for policy in policies:
        for task_group, config_names in task_groups.items():
            rows.append(
                {
                    "policy_id": policy["policy_id"],
                    "policy_kind": policy["policy_kind"],
                    "task_group": task_group,
                    "task_configs": ",".join(config_names),
                    "action_config": policy["action_config"],
                    "interaction_config": policy["interaction_config"],
                    "observation_config": policy["observation_config"],
                    "policy_override": policy["policy_override"],
                    "extra_overrides_json": json.dumps(policy["extra_overrides"], separators=(",", ":")),
                }
            )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fieldnames = list(rows[0])
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write("\t".join(fieldnames) + "\n")
        for row in rows:
            values = [str(row[field]) for field in fieldnames]
            if any("\t" in value or "\n" in value for value in values):
                raise ValueError("Launch-group fields must not contain tabs or newlines.")
            stream.write("\t".join(values) + "\n")
    temporary.replace(path)


def load_inventory(config: DictConfig) -> dict[str, Any]:
    """Load and minimally validate the frozen campaign inventory."""
    path = Path(str(config.output_root)).resolve() / "inventory.json"
    if not path.is_file():
        raise FileNotFoundError(f"Build the offline inventory first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != OFFLINE_DATASET_SCHEMA_VERSION:
        raise ValueError("Offline inventory schema mismatch.")
    return payload


def audit_inventory(config: DictConfig) -> dict[str, Any]:
    """Validate shards and write complete/missing/failed/corrupt status."""
    inventory = load_inventory(config)
    counts = {"success": 0, "failed": 0, "corrupt": 0, "missing": 0}
    details: list[dict[str, Any]] = []
    for row in inventory["rows"]:
        path = Path(row["output_path"])
        status_path = path.with_name("offline_episode_status.json")
        state = "missing"
        message = ""
        if path.is_file():
            try:
                metadata = validate_episode_npz(
                    path,
                    expected_task_id=row["task_id"],
                    expected_seed=row["seed"],
                    expected_policy_id=row["policy_id"],
                )
                state = "success"
                message = f"{metadata['transition_count']} transitions"
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                state = "corrupt"
                message = f"{type(error).__name__}: {error}"
        elif status_path.is_file():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "failed":
                state = "failed"
                message = str(status.get("exception_message", ""))
        counts[state] += 1
        if state != "success":
            details.append({"job_index": row["job_index"], "state": state, "message": message, **row})
    result = {
        "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
        "manifest_hash": inventory["manifest_hash"],
        "audited_at": _utc_now(),
        "expected": len(inventory["rows"]),
        "counts": counts,
        "complete": counts["success"] == len(inventory["rows"]),
        "non_success_rows": details,
    }
    _atomic_json(Path(str(config.output_root)).resolve() / "status.json", result)
    return result


def _stage_consolidated_arrays(
    rows: list[dict[str, Any]], staging: Path
) -> tuple[list[Path], int, list[str], dict[str, Any]]:
    metadata_rows: list[str] = []
    transition_counts: list[int] = []
    array_spec: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    data_keys: list[str] = []
    for row in rows:
        path = Path(row["output_path"])
        metadata = validate_episode_npz(
            path,
            expected_task_id=row["task_id"],
            expected_seed=row["seed"],
            expected_policy_id=row["policy_id"],
        )
        metadata.update(
            {
                "job_index": row["job_index"],
                "scientific_id": row.get("scientific_id"),
                "task_group": row.get("task_group"),
                "benchmark_id": row.get("benchmark_id"),
                "task_config_sha256": row.get("task_config_sha256"),
                "native_n_trials": row.get("native_n_trials"),
                "policy_kind": row.get("policy_kind"),
                "configured_alpha": row.get("alpha"),
                "configured_duration": row.get("duration"),
            }
        )
        metadata_rows.append(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        with np.load(path, allow_pickle=False) as payload:
            count = int(payload["rewards"].shape[0])
            transition_counts.append(count)
            current_keys = sorted(key for key in payload.files if key != "metadata_json")
            if not data_keys:
                data_keys = current_keys
                array_spec = {key: (payload[key].dtype, payload[key].shape[1:]) for key in data_keys}
            elif current_keys != data_keys:
                raise ValueError(f"Shard {path} does not share the canonical array set.")
            else:
                for key in data_keys:
                    expected_dtype, expected_shape = array_spec[key]
                    if payload[key].dtype != expected_dtype or payload[key].shape[1:] != expected_shape:
                        raise ValueError(f"Shard {path} has an incompatible {key!r} array.")

    total = sum(transition_counts)
    memmaps: dict[str, np.memmap[Any, Any]] = {}
    staged_paths: list[Path] = []
    for key in data_keys:
        dtype, trailing_shape = array_spec[key]
        staged = staging / f"{key}.npy"
        memmaps[key] = np.lib.format.open_memmap(staged, mode="w+", dtype=dtype, shape=(total, *trailing_shape))
        staged_paths.append(staged)
    episode_index_path = staging / "episode_index.npy"
    memmaps["episode_index"] = np.lib.format.open_memmap(episode_index_path, mode="w+", dtype=np.int32, shape=(total,))
    staged_paths.append(episode_index_path)

    offset = 0
    for episode_index, (row, count) in enumerate(zip(rows, transition_counts, strict=True)):
        with np.load(Path(row["output_path"]), allow_pickle=False) as payload:
            for key in data_keys:
                memmaps[key][offset : offset + count] = payload[key]
        memmaps["episode_index"][offset : offset + count] = episode_index
        offset += count
    for array in memmaps.values():
        array.flush()
    del memmaps

    maximum_metadata_length = max(map(len, metadata_rows), default=1)
    episode_metadata = np.asarray(metadata_rows, dtype=f"U{maximum_metadata_length}")
    episode_metadata_path = staging / "episode_metadata_json.npy"
    np.save(episode_metadata_path, episode_metadata, allow_pickle=False)
    staged_paths.append(episode_metadata_path)
    episode_offsets = np.concatenate(([0], np.cumsum(transition_counts, dtype=np.int64)))
    offsets_path = staging / "episode_offsets.npy"
    np.save(offsets_path, episode_offsets, allow_pickle=False)
    staged_paths.append(offsets_path)
    return (
        staged_paths,
        total,
        metadata_rows,
        {key: [str(dtype), list(shape)] for key, (dtype, shape) in array_spec.items()},
    )


def consolidate(config: DictConfig) -> dict[str, Any]:
    """Create one standard NPZ using disk-backed staging instead of RAM."""
    inventory = load_inventory(config)
    audit = audit_inventory(config)
    if not audit["complete"] and not bool(config.allow_incomplete_consolidation):
        raise RuntimeError(f"Refusing incomplete consolidation: {audit['counts']}.")
    successful_indices = {row["job_index"] for row in inventory["rows"] if Path(row["output_path"]).is_file()}
    rows = [row for row in inventory["rows"] if row["job_index"] in successful_indices]
    output_root = Path(str(config.output_root)).resolve()
    destination = output_root / str(config.consolidated_filename)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite consolidated dataset: {destination}")
    with tempfile.TemporaryDirectory(prefix="offline-consolidate-", dir=output_root) as temporary_root:
        staging = Path(temporary_root)
        staged_paths, transition_count, metadata_rows, array_spec = _stage_consolidated_arrays(rows, staging)
        dataset_metadata = {
            "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
            "manifest_hash": inventory["manifest_hash"],
            "source_revision": inventory["source_revision"],
            "reference_table_sha256": inventory["reference_table_sha256"],
            "episode_count": len(rows),
            "transition_count": transition_count,
            "observation_keys": inventory["observation_keys"],
            "array_spec": array_spec,
            "episode_metadata_count": len(metadata_rows),
            "created_at": _utc_now(),
        }
        metadata_path = staging / "dataset_metadata_json.npy"
        np.save(
            metadata_path,
            np.asarray(json.dumps(dataset_metadata, sort_keys=True, separators=(",", ":"))),
            allow_pickle=False,
        )
        staged_paths.append(metadata_path)
        if str(config.compression) not in {"deflated", "stored"}:
            raise ValueError("Offline compression must be 'deflated' or 'stored'.")
        compression = zipfile.ZIP_DEFLATED if str(config.compression) == "deflated" else zipfile.ZIP_STORED
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with zipfile.ZipFile(temporary, mode="w", compression=compression, allowZip64=True) as archive:
            for staged_path in sorted(staged_paths, key=lambda path: path.name):
                archive.write(staged_path, arcname=staged_path.name)
        temporary.replace(destination)
    result = {
        **dataset_metadata,
        "dataset_path": str(destination),
        "dataset_sha256": file_sha256(destination),
        "dataset_bytes": destination.stat().st_size,
    }
    _atomic_json(output_root / "consolidation.json", result)
    return result


def record_submission(config: DictConfig) -> dict[str, Any]:
    """Claim one immutable submission mode before any Hydra launcher runs."""
    inventory = load_inventory(config)
    output_root = Path(str(config.output_root)).resolve()
    mode = str(config.launch_mode)
    path = output_root / "submissions" / f"{mode}.json"
    if path.exists():
        raise FileExistsError(f"Submission marker already exists: {path}")
    result = {
        "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
        "manifest_hash": inventory["manifest_hash"],
        "launch_mode": mode,
        "submission_claimed_at": _utc_now(),
        "note": "Immutable pre-launch claim. Missing completion marker means launcher expansion was interrupted.",
    }
    _atomic_json(path, result)
    return result


def record_launch_completion(config: DictConfig) -> dict[str, Any]:
    """Record that every Hydra launcher command returned successfully."""
    inventory = load_inventory(config)
    output_root = Path(str(config.output_root)).resolve()
    mode = str(config.launch_mode)
    claim = output_root / "submissions" / f"{mode}.json"
    if not claim.is_file():
        raise FileNotFoundError(f"Submission completion has no immutable claim: {claim}")
    path = output_root / "submissions" / f"{mode}.complete.json"
    if path.exists():
        raise FileExistsError(f"Submission completion marker already exists: {path}")
    result = {
        "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
        "manifest_hash": inventory["manifest_hash"],
        "launch_mode": mode,
        "launcher_expansion_completed_at": _utc_now(),
        "note": "Every Hydra Submitit launcher invocation returned; scientific completion is audited from NPZ shards.",
    }
    _atomic_json(path, result)
    return result


@hydra.main(
    config_path="../configs",
    config_name="offline_dataset/carps_bbob_yahpo_v1",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Execute one management operation selected by Hydra configuration."""
    config = cfg.offline_dataset
    operation = str(config.operation)
    if operation == "build":
        result = build_inventory(config)
    elif operation == "status":
        result = audit_inventory(config)
    elif operation == "consolidate":
        result = consolidate(config)
    elif operation == "record_submission":
        result = record_submission(config)
    elif operation == "record_launch_completion":
        result = record_launch_completion(config)
    else:
        raise ValueError(f"Unknown offline dataset operation: {operation!r}.")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key not in {"rows", "tasks", "policies", "scientific_ids"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
