"""Audit expected evaluation cells against atomic terminal status records."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dacboenv.experiment.evaluation_status import atomic_json, evaluation_cell_hash


def load_expected_protocol(path: Path) -> list[dict[str, Any]]:
    """Load and validate an explicit expected-cell inventory."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"Expected protocol has no nonempty 'cells' list: {path}")
    hashes = [evaluation_cell_hash(cell) for cell in cells]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Expected evaluation protocol contains duplicate cell keys.")
    return cells


def audit_matrix(evaluation_root: Path, expected_protocol: Path) -> dict[str, Any]:
    """Classify every expected, duplicate, conflicting, and unexpected cell."""
    expected = load_expected_protocol(expected_protocol)
    expected_by_hash = {evaluation_cell_hash(cell): cell for cell in expected}
    observed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(evaluation_root.rglob("*.status.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["status_path"] = str(path.resolve())
        observed[str(record.get("cell_hash") or evaluation_cell_hash(record))].append(record)

    successful, failed, missing, duplicates, conflicts = [], [], [], [], []
    for cell_hash, cell in expected_by_hash.items():
        records = observed.get(cell_hash, [])
        successes = [record for record in records if record.get("status") == "success"]
        failures = [record for record in records if record.get("status") == "failed"]
        if not records:
            missing.append(cell)
        elif len(successes) == 1:
            successful.append(successes[0])
        elif not successes:
            failed.append(failures[-1] if failures else records[-1])
        else:
            result_identities = {
                (record.get("context_hash"), record.get("result_sha256"), record.get("result_path"))
                for record in successes
            }
            target = conflicts if len(result_identities) > 1 else duplicates
            target.append({"cell": cell, "records": successes})
    unexpected = [record for key, records in observed.items() if key not in expected_by_hash for record in records]
    return {
        "schema_version": "evaluation-matrix-audit-v1",
        "evaluation_root": str(evaluation_root.resolve()),
        "expected_protocol": str(expected_protocol.resolve()),
        "expected_cells": len(expected),
        "successful_cells": len(successful),
        "failed_cells": len(failed),
        "missing_cells": len(missing),
        "duplicate_cells": len(duplicates),
        "conflicting_duplicate_cells": len(conflicts),
        "unexpected_cells": len(unexpected),
        "successful": successful,
        "failed": failed,
        "missing": missing,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "unexpected": unexpected,
        "complete": not failed and not missing and not conflicts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the matrix audit CLI and fail while required cells are absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--expected-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = audit_matrix(args.evaluation_root, args.expected_protocol)
    output = args.output if args.output.suffix == ".json" else args.output / "evaluation_matrix_audit.json"
    atomic_json(output, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if audit["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
