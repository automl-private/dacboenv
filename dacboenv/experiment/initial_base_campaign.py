"""Hydra management of paired initial-base CARP-S context jobs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.protocol import sealed_final_test_task_ids
from dacboenv.experiment.real_env import _template, real_structured_mixed_env
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.selective_wei.context import canonical_hash, context_from_task_id
from dacboenv.selective_wei.initial_collection import collect_static_context, read_verified, write_verified
from dacboenv.selective_wei.initial_data_reuse import audit_initial_data_reuse
from dacboenv.selective_wei.initial_features import InitialFeatureSettings


def prepare(config: DictConfig) -> dict[str, Any]:
    """Freeze explicit train/dev tasks, five seeds and the execution source hash."""
    path = Path(str(config.partition_manifest)).resolve()
    partition = json.loads(path.read_text())
    declared = partition.pop("manifest_hash")
    if canonical_sha256(partition) != declared:
        raise ValueError("Task partition manifest hash mismatch.")
    splits = partition["task_splits"]
    all_tasks = [task for tasks in splits.values() for task in tasks]
    if len(all_tasks) != len(set(all_tasks)):
        raise ValueError("Canonical tasks overlap across supplied partitions.")
    selected = splits["train"] + splits["dev"]
    if set(selected) & sealed_final_test_task_ids():
        raise ValueError("Protected test task requested for initial-base collection.")
    source = current_source_revision()
    root = Path(str(config.output_root)).resolve()
    feature_values = OmegaConf.to_container(config.features, resolve=True)
    if not isinstance(feature_values, dict):
        raise TypeError("Initial feature settings must be a mapping.")
    settings = InitialFeatureSettings(**{str(key): value for key, value in feature_values.items()})
    jobs: list[dict[str, Any]] = []
    for split in ("train", "dev"):
        for task in splits[split]:
            context = context_from_task_id(task)
            if context.domain not in {"bbob", "yahpo"}:
                raise ValueError("Initial-base campaign currently supports verified BBOB/YAHPO factories only.")
            for seed in config.seeds:
                recipe = {
                    "task_id": task,
                    "split": split,
                    "seed": int(seed),
                    "source_revision": source,
                    "partition_manifest_hash": declared,
                    "continuation_replicate": 0,
                    "reference_table_sha256": file_sha256(Path(str(config.reference_table))),
                }
                jobs.append(
                    {"index": len(jobs), "recipe": recipe, "output": str(root / "contexts" / f"{len(jobs):05d}")}
                )
    result = {
        "schema_version": "initial-base-campaign-v1",
        "jobs": jobs,
        "features": asdict(settings),
        "partition_manifest_hash": declared,
        "source_revision": source,
        "holdout_opened": False,
    }
    destination = root / "manifest.json"
    if destination.exists() and read_verified(destination) != result:
        raise FileExistsError("A different initial-base campaign already exists at this root.")
    write_verified(destination, result)
    return {"manifest": str(destination), "contexts": len(jobs), "static_arms": 5 * len(jobs)}


def run(config: DictConfig) -> dict[str, Any]:
    """Execute one frozen context with all five static continuations."""
    manifest = read_verified(Path(str(config.output_root)) / "manifest.json")
    if manifest["source_revision"] != current_source_revision():
        raise RuntimeError("Queued initial-base worker source changed after preparation.")
    job = manifest["jobs"][int(config.job_index)]
    recipe = job["recipe"]
    reference = Path(str(config.reference_table))
    if file_sha256(reference) != recipe["reference_table_sha256"]:
        raise ValueError("Initial-base reference table changed after preparation.")

    def factory() -> Any:
        return real_structured_mixed_env(
            recipe["task_id"],
            recipe["seed"],
            context_split={"train": "train", "dev": "validation"}[recipe["split"]],
            reference_table=reference,
            interaction_frequency=5,
            initial_context_settings=manifest["features"],
        )

    return collect_static_context(
        factory, Path(job["output"]), recipe=recipe, settings=InitialFeatureSettings(**manifest["features"])
    )


def audit(config: DictConfig, *, consolidate: bool = False) -> dict[str, Any]:
    """Audit content hashes and collect only complete five-arm rows."""
    root = Path(str(config.output_root))
    manifest = read_verified(root / "manifest.json")
    missing, rows = [], []
    for job in manifest["jobs"]:
        try:
            result = read_verified(Path(job["output"]) / "complete.json")
            if result["recipe"] != job["recipe"]:
                raise ValueError("Context recipe changed.")
            for action, digest in enumerate(result["arm_hashes"]):
                if canonical_hash(read_verified(Path(job["output"]) / f"arm_{action}.json")) != digest:
                    raise ValueError("Completed context has a corrupt arm.")
            rows.append(result)
        except (OSError, ValueError, KeyError):
            missing.append(job["index"])
    status = {"expected": len(manifest["jobs"]), "complete": len(rows), "missing_indices": missing}
    write_verified(root / "status.json", status)
    if consolidate:
        if missing:
            raise RuntimeError("Cannot consolidate an incomplete or corrupt initial-base campaign.")
        for split in ("train", "dev"):
            write_verified(
                root / f"initial_base_{split}.json",
                {
                    "schema_version": "initial-base-dataset-v1",
                    "target_semantics": "full_static_terminal_loss",
                    "split": split,
                    "features": manifest["features"],
                    "partition_manifest_hash": manifest["partition_manifest_hash"],
                    "rows": [row for row in rows if row["recipe"]["split"] == split],
                },
            )
    return status


@hydra.main(version_base=None, config_path="../configs", config_name="initial_base_campaign")
def main(config: DictConfig) -> None:
    """Prepare, audit, consolidate or run one explicit initial-base context."""
    if config.operation == "preflight":
        template = _template("wei", 5)
        result = {
            "status": "template_ready",
            "interaction_frequency": int(template.dacboenv.interaction_frequency),
            "source_revision": current_source_revision(),
            "objective_evaluations": 0,
        }
    elif config.operation == "audit_reuse":
        result = audit_initial_data_reuse(Path(str(config.reuse_source)).resolve())
    elif config.operation == "prepare":
        result = prepare(config)
    elif config.operation == "run":
        result = run(config)
    elif config.operation in {"status", "consolidate"}:
        result = audit(config, consolidate=config.operation == "consolidate")
    else:
        raise ValueError("Unknown initial-base operation.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
