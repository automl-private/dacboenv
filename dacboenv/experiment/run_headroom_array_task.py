"""Run one restart-safe CARP-S snapshot-and-branch array task."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dacboenv.experiment.collect_snapshots import main as collect_main
from dacboenv.experiment.run_snapshot_branches import main as branch_main
from dacboenv.experiment.snapshot_branch import require_deterministic_replay_process_environment

SUPPORTED_HISTORIES = {"default_smac", "uniform_random", "family_static"}


def _load_row(inventory: Path, index: int, expected_hash: str) -> tuple[dict[str, Any], str]:
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not isinstance(payload.get("snapshots"), list):
        raise ValueError(f"Invalid snapshot inventory: {inventory}")
    actual_hash = str(payload.get("manifest_hash", ""))
    if actual_hash != expected_hash:
        raise ValueError(f"Inventory hash mismatch: expected {expected_hash}, got {actual_hash}.")
    rows = payload["snapshots"]
    if not 1 <= index <= len(rows):
        raise ValueError(f"Array index must be in 1..{len(rows)}, got {index}.")
    row = dict(rows[index - 1])
    history = str(row["history_generator"])
    if history not in SUPPORTED_HISTORIES:
        raise ValueError(f"Array inventory contains unsupported history {history!r}.")
    return row, actual_hash


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    """Collect one portable state, verify replay, and branch every action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-hash", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    args = parser.parse_args(argv)
    require_deterministic_replay_process_environment()
    row, inventory_hash = _load_row(args.inventory, args.index, args.inventory_hash)

    split = str(row["split"])
    domain = str(row["domain"])
    snapshot_id = str(row["snapshot_id"])
    output = args.output_root / split / str(row["action_family"]) / snapshot_id
    snapshot_path = output / "snapshot.jsonl"
    branch_path = output / "branches.csv"
    summary_path = output / "summary.json"
    metadata_path = output / "array_task.json"
    if all(path.is_file() and path.stat().st_size > 0 for path in (snapshot_path, branch_path, summary_path)):
        print(json.dumps({"status": "already_complete", "snapshot_id": snapshot_id}, sort_keys=True))
        return 0

    root = Path("dacboenv/configs/instance_sets")
    manifest = root / f"{domain}_{split}.yaml"
    forbidden = root / ("bbob_test_strict.yaml" if domain == "bbob" else "yahpo_test_official_so.yaml")
    factory = f"dacboenv.experiment.real_env:real_headroom_{split}_env"
    history = str(row["history_generator"])
    policy = {"default_smac": "default_smac", "uniform_random": "uniform_random", "family_static": "static"}[history]
    output.mkdir(parents=True, exist_ok=True)
    collect_arguments = [
        "--output",
        str(snapshot_path),
        "--manifest",
        str(manifest),
        "--forbidden-task-ids",
        str(forbidden),
        "--factory",
        factory,
        "--task-id",
        str(row["task_id"]),
        "--inner-seed",
        str(row["inner_seed"]),
        "--budget-fraction",
        str(row["budget_fraction"]),
        "--action-space",
        str(row["action_family"]),
        "--policy",
        policy,
        "--policy-seed",
        str(row["history_seed"]),
        "--static-action",
        str(row["static_action"]),
        "--reference-table",
        str(args.reference_table),
    ]
    collect_main(collect_arguments)
    branch_main(
        [
            "--snapshots",
            str(snapshot_path),
            "--factory",
            factory,
            "--forbidden-task-ids",
            str(forbidden),
            "--output-csv",
            str(branch_path),
            "--output-summary",
            str(summary_path),
            "--horizon",
            "1",
            "--horizon",
            "5",
            "--horizon",
            "10",
        ]
    )
    metadata_path.write_text(
        json.dumps(
            {
                "array_index": args.index,
                "inventory": str(args.inventory.resolve()),
                "inventory_hash": inventory_hash,
                "snapshot_id": snapshot_id,
                "snapshot_sha256": _sha256(snapshot_path),
                "branches_sha256": _sha256(branch_path),
                "summary_sha256": _sha256(summary_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "snapshot_id": snapshot_id, "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
