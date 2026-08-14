"""Plan only missing or failed cells from an explicit evaluation matrix."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dacboenv.experiment.audit_evaluation_matrix import audit_matrix, load_expected_protocol
from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.experiment.evaluation_status import atomic_json, evaluation_cell_hash


def plan_repairs(evaluation_root: Path, expected_protocol: Path) -> dict[str, Any]:
    """Return an immutable repair-only subset; successful cells are excluded."""
    expected = load_expected_protocol(expected_protocol)
    expected_by_hash = {evaluation_cell_hash(cell): cell for cell in expected}
    audit = audit_matrix(evaluation_root, expected_protocol)
    repair_hashes = {evaluation_cell_hash(cell) for cell in audit["missing"]}
    repair_hashes.update(str(record["cell_hash"]) for record in audit["failed"])
    cells = [expected_by_hash[cell_hash] for cell_hash in sorted(repair_hashes)]
    return {
        "schema_version": "evaluation-repairs-v1",
        "source_evaluation_root": str(evaluation_root.resolve()),
        "expected_protocol": str(expected_protocol.resolve()),
        "repair_namespace": "evaluation_repairs_v1",
        "cells": cells,
        "repair_cell_count": len(cells),
        "repair_manifest_hash": canonical_sha256(cells),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Write a repair-only manifest for one explicit expected protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-protocol", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.training_root.is_dir():
        raise FileNotFoundError(f"Stage-B training root does not exist: {args.training_root}")
    expected = args.expected_protocol or args.evaluation_root / "expected_protocol.json"
    plan = plan_repairs(args.evaluation_root, expected)
    plan["training_root"] = str(args.training_root.resolve())
    output = args.output if args.output.suffix == ".json" else args.output / "evaluation_repairs.json"
    atomic_json(output, plan)
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
