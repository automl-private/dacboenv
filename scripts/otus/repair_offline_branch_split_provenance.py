#!/usr/bin/env python3
"""Repair train-shard split metadata and archive dev shards for clean replay.

The Hydra re-entry emergency worker accidentally hard-coded
``context_split='train'`` and omitted split fields.  Existing train shards are
scientifically correct and can be annotated deterministically.  Existing dev
shards must be rerun with validation environment semantics, so this utility
archives them and leaves their manifest rows missing for resubmission.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _compress_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    values = sorted(set(indices))
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    archive_root = root / "legacy_dev_shards_executed_as_train"

    repaired_train = 0
    already_repaired_train = 0
    archived_dev = 0
    missing_dev = 0
    dev_indices: list[int] = []

    for row in manifest["jobs"]:
        split = str(row["context_split"])
        index = int(row["job_index"])
        output = Path(row["output_path"])
        failed = output.with_suffix(".failed.json")

        if split == "train":
            if not output.is_file():
                raise FileNotFoundError(
                    f"Expected completed train shard before repair: {output}"
                )
            payload = json.loads(output.read_text(encoding="utf-8"))
            if payload.get("status") != "success":
                raise RuntimeError(f"Train shard is not successful: {output}")
            if payload.get("job_hash") != row["job_hash"]:
                raise RuntimeError(f"Train shard job hash mismatch: {output}")
            record = payload["branch_record"]
            old_data = record.get("data_context_split")
            old_environment = record.get("environment_context_split")
            if old_data not in {None, "train"}:
                raise RuntimeError(
                    f"Conflicting train data_context_split in {output}: {old_data!r}"
                )
            if old_environment not in {None, "train"}:
                raise RuntimeError(
                    "Conflicting train environment_context_split in "
                    f"{output}: {old_environment!r}"
                )
            if old_data == "train" and old_environment == "train":
                already_repaired_train += 1
                continue
            record["data_context_split"] = "train"
            record["environment_context_split"] = "train"
            payload["split_provenance_repair"] = {
                "repair_version": "offline-branch-split-provenance-r1",
                "repaired_at": datetime.now(UTC).isoformat(),
                "reason": (
                    "Emergency Hydra leaf-worker fix omitted split fields; "
                    "train rows were nevertheless executed with the correct "
                    "train environment semantics."
                ),
            }
            _atomic_json(output, payload)
            repaired_train += 1
            continue

        if split != "dev":
            raise RuntimeError(f"Unexpected branch data split {split!r} at row {index}")

        dev_indices.append(index)
        if output.is_file():
            destination = archive_root / "jobs" / output.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(output), str(destination))
            archived_dev += 1
        else:
            missing_dev += 1
        if failed.is_file():
            destination = archive_root / "failed" / failed.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError(destination)
            shutil.move(str(failed), str(destination))

    report = {
        "repair_version": "offline-branch-split-provenance-r1",
        "manifest": str(manifest_path),
        "manifest_hash": manifest.get("manifest_hash"),
        "repaired_at": datetime.now(UTC).isoformat(),
        "repaired_train_shards": repaired_train,
        "already_repaired_train_shards": already_repaired_train,
        "archived_dev_shards": archived_dev,
        "already_missing_dev_shards": missing_dev,
        "dev_indices": dev_indices,
        "slurm_array": _compress_indices(dev_indices),
        "archive_root": str(archive_root),
        "note": (
            "Archived dev shards were generated with train environment semantics "
            "and must not enter the consolidated development dataset."
        ),
    }
    _atomic_json(root / "split_provenance_repair.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
