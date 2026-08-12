"""Validate and consolidate completed baseline-history headroom array outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dacboenv.experiment.headroom_predictability import write_parquet_atomic

EXPECTED_ACTIONS = frozenset(range(5))
EXPECTED_HORIZONS = frozenset({1, 5, 10})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_inventory(path: Path) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("snapshots") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list) or int(payload.get("snapshot_count", -1)) != len(rows):
        raise ValueError(f"Invalid campaign inventory {path}.")
    return str(payload["manifest_hash"]), [dict(row) for row in rows]


def _expected_history(row: Mapping[str, Any]) -> str:
    if row["history_generator"] == "family_static":
        return f"static_{int(row['static_action'])}"
    if row["history_generator"] == "default_smac":
        return "default_smac_equivalent"
    return str(row["history_generator"])


def _validate_one(  # noqa: C901
    root: Path, inventory_hash: str, row: Mapping[str, Any]
) -> tuple[dict[str, Any], pd.DataFrame]:
    directory = root / str(row["split"]) / str(row["action_family"]) / str(row["snapshot_id"])
    paths = {name: directory / name for name in ("snapshot.jsonl", "branches.csv", "summary.json", "array_task.json")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete array row {row['snapshot_id']}: {missing!r}.")
    metadata = json.loads(paths["array_task.json"].read_text(encoding="utf-8"))
    if metadata["inventory_hash"] != inventory_hash:
        raise ValueError(f"Inventory hash mismatch for {row['snapshot_id']}.")
    for field, filename in (
        ("snapshot_sha256", "snapshot.jsonl"),
        ("branches_sha256", "branches.csv"),
        ("summary_sha256", "summary.json"),
    ):
        if metadata[field] != _sha256(paths[filename]):
            raise ValueError(f"Output hash mismatch for {row['snapshot_id']}/{filename}.")

    snapshot_lines = [line for line in paths["snapshot.jsonl"].read_text(encoding="utf-8").splitlines() if line]
    if len(snapshot_lines) != 1:
        raise ValueError(f"Expected one portable snapshot for {row['snapshot_id']}.")
    snapshot = json.loads(snapshot_lines[0])
    expected = {
        "task_id": str(row["task_id"]),
        "inner_seed": int(row["inner_seed"]),
        "action_space": str(row["action_family"]),
        "history_policy": _expected_history(row),
    }
    if any(snapshot[name] != value for name, value in expected.items()):
        raise ValueError(f"Portable snapshot context differs from inventory row {row['snapshot_id']}.")
    if not np.isclose(float(snapshot["budget_fraction"]), float(row["budget_fraction"]), atol=0.02, rtol=0):
        raise ValueError(f"Snapshot phase differs from target for {row['snapshot_id']}.")

    branches = pd.read_csv(paths["branches.csv"])
    if len(branches) != len(EXPECTED_ACTIONS) * len(EXPECTED_HORIZONS):
        raise ValueError(f"Incomplete branch matrix for {row['snapshot_id']}.")
    if set(branches["action"]) != EXPECTED_ACTIONS or set(branches["horizon"]) != EXPECTED_HORIZONS:
        raise ValueError(f"Wrong actions/horizons for {row['snapshot_id']}.")
    if branches.groupby(["action", "horizon"]).size().ne(1).any():
        raise ValueError(f"Duplicate branch cells for {row['snapshot_id']}.")
    numeric = branches[["final_incumbent", "normalized_potential_improvement", "raw_regret_improvement"]]
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError(f"Non-finite scientific outcome for {row['snapshot_id']}.")
    branches.insert(0, "campaign_snapshot_id", str(row["snapshot_id"]))
    branches["split"] = str(row["split"])
    branches["domain"] = str(row["domain"])
    branches["history_generator"] = str(row["history_generator"])
    branches["target_budget_fraction"] = float(row["budget_fraction"])
    branches["q_value"] = branches["normalized_potential_improvement"]
    snapshot["campaign_snapshot_id"] = str(row["snapshot_id"])
    snapshot["split"] = str(row["split"])
    snapshot["history_generator"] = str(row["history_generator"])
    snapshot["inventory_hash"] = inventory_hash
    snapshot["completed_evaluations_json"] = json.dumps(snapshot.pop("completed_evaluations"), sort_keys=True)
    return snapshot, branches


def _domain_balanced(values: pd.DataFrame, column: str) -> float:
    domain_values: list[float] = []
    for domain, domain_rows in values.groupby("domain", sort=True):
        stratum = "dimension" if domain == "bbob" else "scenario"
        domain_values.append(float(domain_rows.groupby(stratum, sort=True)[column].mean().mean()))
    return float(np.mean(domain_values))


def _headroom_summary(branches: pd.DataFrame) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    for (family, horizon), train in branches[branches["split"] == "train"].groupby(
        ["action_space", "horizon"], sort=True
    ):
        validation = branches[
            (branches["split"] == "validation")
            & (branches["action_space"] == family)
            & (branches["horizon"] == horizon)
        ]
        train_action_values = {
            int(action): _domain_balanced(rows, "q_value") for action, rows in train.groupby("action", sort=True)
        }
        best_train_value = max(train_action_values.values())
        global_action = min(action for action, value in train_action_values.items() if value == best_train_value)
        state = validation.groupby(
            ["campaign_snapshot_id", "domain", "scenario", "dimension"], dropna=False, sort=True
        )["q_value"]
        oracle_rows = state.max().reset_index(name="oracle")
        static_rows = validation[validation["action"] == global_action].rename(columns={"q_value": "static"})
        oracle = _domain_balanced(oracle_rows, "oracle")
        static = _domain_balanced(static_rows, "static")
        summaries.append(
            {
                "action_family": family,
                "horizon": int(horizon),
                "train_selected_global_action": global_action,
                "validation_dynamic_oracle": oracle,
                "validation_global_static": static,
                "intrinsic_headroom": oracle - static,
                "relative_headroom": (oracle - static) / max(abs(oracle), np.finfo(float).eps),
            }
        )
    return {
        "scope": "baseline_histories_without_sawei",
        "snapshot_count": int(branches["campaign_snapshot_id"].nunique()),
        "branch_row_count": len(branches),
        "results": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate every row and write consolidated Parquet/JSON artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--train-inventory", type=Path, required=True)
    parser.add_argument("--validation-inventory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    snapshots: list[dict[str, Any]] = []
    branch_frames: list[pd.DataFrame] = []
    for inventory in (args.train_inventory, args.validation_inventory):
        inventory_hash, rows = _load_inventory(inventory)
        for row in rows:
            snapshot, branches = _validate_one(args.campaign_root, inventory_hash, row)
            snapshots.append(snapshot)
            branch_frames.append(branches)
    snapshot_frame = pd.DataFrame(snapshots)
    branches = pd.concat(branch_frames, ignore_index=True)
    if snapshot_frame["campaign_snapshot_id"].duplicated().any():
        raise ValueError("Duplicate campaign snapshot IDs across inventories.")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    write_parquet_atomic(
        snapshot_frame[snapshot_frame["split"] == "train"], args.output_directory / "headroom_train_snapshots.parquet"
    )
    write_parquet_atomic(
        snapshot_frame[snapshot_frame["split"] == "validation"],
        args.output_directory / "headroom_validation_snapshots.parquet",
    )
    write_parquet_atomic(branches, args.output_directory / "branch_results.parquet")
    replicates = branches.copy()
    replicates["branch_replicate"] = 0
    write_parquet_atomic(replicates, args.output_directory / "branch_replicates.parquet")
    summary = _headroom_summary(branches)
    (args.output_directory / "headroom_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
