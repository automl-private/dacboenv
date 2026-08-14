"""Safely merge repair successes into a new evaluation namespace."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dacboenv.experiment.audit_evaluation_matrix import audit_matrix, load_expected_protocol
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.evaluation_status import atomic_json, evaluation_cell_hash


def _successes(root: Path) -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("*.status.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") != "success":
            continue
        record["_status_path"] = str(path.resolve())
        records.setdefault(str(record.get("cell_hash") or evaluation_cell_hash(record)), []).append(record)
    return records


def merge_repairs(original: Path, repairs: Path, expected_protocol: Path, output: Path) -> dict[str, Any]:
    """Merge only cells absent/failed in the original, rejecting conflicts."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite populated merged evaluation root: {output}")
    expected = load_expected_protocol(expected_protocol)
    original_records = _successes(original)
    repair_records = _successes(repairs)
    output.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, Any]] = []
    for cell in expected:
        cell_hash = evaluation_cell_hash(cell)
        old = original_records.get(cell_hash, [])
        new = repair_records.get(cell_hash, [])
        if len(old) > 1 or len(new) > 1:
            raise RuntimeError(f"Cell {cell_hash} has duplicate successes; refusing silent source preference.")
        if old and new:
            old_identity = (old[0].get("context_hash"), old[0].get("result_sha256"))
            new_identity = (new[0].get("context_hash"), new[0].get("result_sha256"))
            if old_identity != new_identity:
                raise RuntimeError(f"Conflicting successful original/repair cell {cell_hash}.")
            raise RuntimeError(f"Repair namespace reran already-successful cell {cell_hash}.")
        source = old or new
        if not source:
            continue
        status = source[0]
        expected_initial = cell.get("initial_design_hash_expected")
        if expected_initial and status.get("initial_design_hash") != expected_initial:
            raise RuntimeError(
                f"Initial-design fingerprint mismatch for cell {cell_hash}: "
                f"expected {expected_initial}, observed {status.get('initial_design_hash')}."
            )
        result = Path(status["result_path"])
        if not result.is_file():
            raise FileNotFoundError(f"Successful status references missing result: {result}")
        if status.get("result_sha256") and file_sha256(result) != status["result_sha256"]:
            raise RuntimeError(f"Result hash changed for cell {cell_hash}.")
        destination_dir = output / "cells" / cell_hash
        destination_dir.mkdir(parents=True)
        copied_status = dict(status)
        copied_status.pop("_status_path", None)
        # Imported CARP-S statuses intentionally point into one shared
        # logs.parquet.  Copying that file once per successful cell would turn
        # a small metadata merge into hundreds of gigabytes.  Preserve the
        # content-addressed source reference; the merged namespace is a
        # scientific index, not a second physical copy of immutable results.
        copied_status["source_namespace"] = "original" if old else "repair"
        copied_status["source_result_path"] = str(result.resolve())
        atomic_json(destination_dir / "episode.status.json", copied_status)
        merged.append({"cell_hash": cell_hash, "source": "original" if old else "repair"})
    audit = audit_matrix(output, expected_protocol)
    manifest = {"schema_version": "evaluation-merged-v1", "cells": merged, "audit": audit}
    atomic_json(output / "merge_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Run the safe repair merge CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--expected-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = merge_repairs(args.original_root, args.repair_root, args.expected_protocol, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audit"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
