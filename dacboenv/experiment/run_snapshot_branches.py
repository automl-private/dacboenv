"""Run guarded real-action branches from portable DACBO snapshot JSONL files."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from dacboenv.experiment.collect_snapshots import (
    bind_snapshot_action_space_factory,
    read_snapshots,
    verify_portable_snapshot_replay,
)
from dacboenv.experiment.protocol import sealed_final_test_task_ids
from dacboenv.experiment.snapshot_branch import (
    DEFAULT_BRANCH_HORIZONS,
    SnapshotBranchReport,
    SnapshotBranchResult,
    replay_process_environment,
    require_deterministic_replay_process_environment,
    run_snapshot_branch_diagnostic,
    snapshot_record_digest,
)

BRANCH_SCHEMA_VERSION = 3
EARLY_MIDDLE_PHASE_BOUNDARY = 0.375
MIDDLE_LATE_PHASE_BOUNDARY = 0.625


def _budget_phase(fraction: float | None) -> str:
    if fraction is None:
        return "unknown"
    if fraction < EARLY_MIDDLE_PHASE_BOUNDARY:
        return "25_percent"
    if fraction < MIDDLE_LATE_PHASE_BOUNDARY:
        return "50_percent"
    return "75_percent"


def _dimension(task_id: str) -> int | None:
    parts = task_id.split("/")
    if len(parts) == 4 and parts[0].lower() == "bbob":  # noqa: PLR2004
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def _scenario(task_id: str) -> str:
    parts = task_id.split("/")
    if parts[0].lower() == "bbob":
        return "bbob"
    if len(parts) >= 3 and parts[0].lower() == "yahpo":  # noqa: PLR2004
        return parts[2]
    return parts[0]


def branch_rows(report: SnapshotBranchReport) -> list[dict[str, Any]]:
    """Flatten branch outcomes and portable provenance into tidy records."""
    rows: list[dict[str, Any]] = []
    for branch in report.branches:
        snapshot = branch.snapshot
        rows.append(
            {
                "schema_version": BRANCH_SCHEMA_VERSION,
                "snapshot_index": branch.snapshot_index,
                "snapshot_record_hash": snapshot_record_digest(snapshot),
                "task_id": snapshot.task_id,
                "scenario": _scenario(snapshot.task_id),
                "dimension": _dimension(snapshot.task_id),
                "inner_seed": snapshot.inner_seed,
                "action_space": snapshot.action_space,
                "interaction_frequency": snapshot.interaction_frequency,
                "action_history": json.dumps(snapshot.action_history, separators=(",", ":")),
                "completed_evaluation_count": len(snapshot.completed_evaluations),
                "budget_fraction": snapshot.budget_fraction,
                "budget_phase": _budget_phase(snapshot.budget_fraction),
                "history_policy": snapshot.history_policy,
                "outer_policy_seed": snapshot.outer_policy_seed,
                "source_manifest": snapshot.source_manifest,
                "source_manifest_hash": snapshot.source_manifest_hash,
                "code_commit": snapshot.code_commit,
                "observation_hash": snapshot.observation_hash,
                "reference_kind": snapshot.reference_kind,
                "reference_source": snapshot.reference_source,
                "reference_source_hash": snapshot.reference_source_hash,
                "reference_runtime_objective_transform": snapshot.reference_runtime_objective_transform,
                "reference_reporting_objective_transform": snapshot.reference_reporting_objective_transform,
                "reference_fidelity_json": snapshot.reference_fidelity_json,
                "reference_tolerance": snapshot.reference_tolerance,
                "reference_benchmark_code_version": snapshot.reference_benchmark_code_version,
                "reference_benchmark_data_version": snapshot.reference_benchmark_data_version,
                "snapshot_reference_value": snapshot.reference_value,
                "snapshot_initial_design_incumbent": snapshot.initial_design_incumbent,
                "action": branch.action,
                "horizon": branch.horizon,
                "reference_value": branch.reference_value,
                "initial_incumbent": branch.initial_incumbent,
                "final_incumbent": branch.final_incumbent,
                "initial_regret": branch.initial_regret,
                "final_regret": branch.final_regret,
                "raw_regret_improvement": branch.regret_improvement,
                "initial_potential": branch.initial_potential,
                "final_potential": branch.final_potential,
                "normalized_potential_improvement": branch.normalized_potential_improvement,
                "policy_steps": branch.policy_steps,
                "terminated": branch.terminated,
                "truncated": branch.truncated,
                "reference_breach": branch.reference_breach,
                "configuration_trace": json.dumps(branch.configuration_trace, separators=(",", ":")),
            }
        )
    return rows


_BRANCH_PROVENANCE_COLUMNS = frozenset(
    {
        "schema_version",
        "snapshot_index",
        "snapshot_record_hash",
        "task_id",
        "inner_seed",
        "action_space",
        "interaction_frequency",
        "action_history",
        "completed_evaluation_count",
        "budget_fraction",
        "history_policy",
        "outer_policy_seed",
        "source_manifest",
        "source_manifest_hash",
        "code_commit",
        "observation_hash",
        "reference_kind",
        "reference_source",
        "reference_source_hash",
        "reference_runtime_objective_transform",
        "reference_reporting_objective_transform",
        "reference_fidelity_json",
        "reference_tolerance",
        "reference_benchmark_code_version",
        "reference_benchmark_data_version",
        "snapshot_reference_value",
        "snapshot_initial_design_incumbent",
        "configuration_trace",
    }
)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _optional_int(value: str) -> int | None:
    return None if value == "" else int(value)


def _same_optional_float(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return bool(np.isclose(left, right, rtol=0.0, atol=0.0))


def validate_branch_row_provenance(  # noqa: C901
    row: Mapping[str, str],
    snapshot: Any,
    *,
    snapshot_index: int,
) -> None:
    """Require a branch row to match every field of its portable source record."""
    missing = sorted(_BRANCH_PROVENANCE_COLUMNS - set(row))
    if missing:
        raise ValueError(f"Branch CSV is missing required provenance columns: {missing!r}.")
    if int(row["snapshot_index"]) != snapshot_index:
        raise ValueError("Branch CSV snapshot_index does not match the selected portable snapshot.")
    if int(row["schema_version"]) != BRANCH_SCHEMA_VERSION:
        raise ValueError(f"Branch CSV schema_version must be {BRANCH_SCHEMA_VERSION}, got {row['schema_version']!r}.")

    expected_strings = {
        "task_id": snapshot.task_id,
        "action_space": snapshot.action_space,
        "history_policy": snapshot.history_policy,
        "source_manifest": snapshot.source_manifest,
        "source_manifest_hash": snapshot.source_manifest_hash,
        "code_commit": snapshot.code_commit,
        "observation_hash": snapshot.observation_hash,
        "reference_kind": snapshot.reference_kind,
        "reference_source": snapshot.reference_source,
        "reference_source_hash": snapshot.reference_source_hash,
        "reference_runtime_objective_transform": snapshot.reference_runtime_objective_transform,
        "reference_reporting_objective_transform": snapshot.reference_reporting_objective_transform,
        "reference_fidelity_json": snapshot.reference_fidelity_json,
        "reference_benchmark_code_version": snapshot.reference_benchmark_code_version,
        "reference_benchmark_data_version": snapshot.reference_benchmark_data_version,
    }
    mismatched = sorted(name for name, expected in expected_strings.items() if row[name] != expected)
    if mismatched:
        raise ValueError(f"Branch CSV provenance differs from its portable snapshot: {mismatched!r}.")
    if row["snapshot_record_hash"] != snapshot_record_digest(snapshot):
        raise ValueError("Branch CSV portable snapshot hash does not match its source record.")
    if int(row["inner_seed"]) != snapshot.inner_seed:
        raise ValueError("Branch CSV inner seed does not match its portable snapshot.")
    if int(row["interaction_frequency"]) != snapshot.interaction_frequency:
        raise ValueError("Branch CSV interaction frequency does not match its portable snapshot.")
    if tuple(json.loads(row["action_history"])) != snapshot.action_history:
        raise ValueError("Branch CSV action history does not match its portable snapshot.")
    if int(row["completed_evaluation_count"]) != len(snapshot.completed_evaluations):
        raise ValueError("Branch CSV completed-evaluation count does not match its portable snapshot.")
    if _optional_int(row["outer_policy_seed"]) != snapshot.outer_policy_seed:
        raise ValueError("Branch CSV outer policy seed does not match its portable snapshot.")
    optional_floats = {
        "budget_fraction": snapshot.budget_fraction,
        "reference_tolerance": snapshot.reference_tolerance,
        "snapshot_reference_value": snapshot.reference_value,
        "snapshot_initial_design_incumbent": snapshot.initial_design_incumbent,
    }
    for field, expected in optional_floats.items():
        if not _same_optional_float(_optional_float(row[field]), expected):
            raise ValueError(f"Branch CSV {field} does not match its portable snapshot.")


def _value_summary(matrix: np.ndarray, actions: Sequence[int]) -> dict[str, Any]:
    dynamic_best = np.max(matrix, axis=1)
    dynamic_oracle = float(np.mean(dynamic_best))
    static_values = np.mean(matrix, axis=0)
    best_static_index = int(np.flatnonzero(np.isclose(static_values, np.max(static_values), atol=1e-12, rtol=0))[0])
    best_static = float(static_values[best_static_index])
    headroom = max(dynamic_oracle - best_static, 0.0)
    tied = np.isclose(matrix, dynamic_best[:, None], atol=1e-12, rtol=0)
    frequencies = np.mean(tied / np.sum(tied, axis=1, keepdims=True), axis=0)
    if len(actions) == 1:
        gaps = np.zeros(matrix.shape[0], dtype=float)
    else:
        ordered = np.sort(matrix, axis=1)
        gaps = ordered[:, -1] - ordered[:, -2]
    return {
        "dynamic_oracle": dynamic_oracle,
        "best_static_action": int(actions[best_static_index]),
        "best_static": best_static,
        "dynamic_headroom": headroom,
        "relative_dynamic_headroom": headroom / max(abs(dynamic_oracle), np.finfo(float).eps),
        "best_action_frequency": {str(action): float(frequencies[index]) for index, action in enumerate(actions)},
        "mean_value_by_action": {str(action): float(static_values[index]) for index, action in enumerate(actions)},
        "mean_top1_top2_gap": float(np.mean(gaps)),
    }


def _summarize_branch_group(branches: Sequence[SnapshotBranchResult]) -> list[dict[str, Any]]:
    snapshot_indices = sorted({branch.snapshot_index for branch in branches})
    actions = sorted({branch.action for branch in branches})
    horizons = sorted({branch.horizon for branch in branches})
    outcomes = {(branch.snapshot_index, branch.action, branch.horizon): branch for branch in branches}
    summaries: list[dict[str, Any]] = []
    for horizon in horizons:
        raw = np.empty((len(snapshot_indices), len(actions)), dtype=float)
        normalized = np.empty_like(raw)
        for row_index, snapshot_index in enumerate(snapshot_indices):
            for action_index, action in enumerate(actions):
                try:
                    branch = outcomes[(snapshot_index, action, horizon)]
                except KeyError as error:
                    raise RuntimeError(
                        f"Incomplete branch matrix at snapshot={snapshot_index}, action={action}, horizon={horizon}."
                    ) from error
                raw[row_index, action_index] = branch.regret_improvement
                normalized[row_index, action_index] = branch.normalized_potential_improvement
        summaries.append(
            {
                "horizon": horizon,
                "n_snapshots": len(snapshot_indices),
                "raw_regret_improvement": _value_summary(raw, actions),
                "normalized_potential_improvement": _value_summary(normalized, actions),
            }
        )
    return summaries


def summarize_branch_report(report: SnapshotBranchReport) -> dict[str, Any]:
    """Summarize overall headroom and requested scientific strata."""
    grouping_fields = {
        "by_task": ("task_id",),
        "by_dimension": ("dimension",),
        "by_budget_phase": ("budget_phase",),
        "by_action_space": ("action_space",),
        "by_history_policy": ("history_policy",),
    }
    grouped_output: dict[str, list[dict[str, Any]]] = {}
    for grouping_name, fields in grouping_fields.items():
        groups: dict[tuple[Any, ...], list[SnapshotBranchResult]] = defaultdict(list)
        for branch in report.branches:
            snapshot = branch.snapshot
            values = {
                "task_id": snapshot.task_id,
                "dimension": _dimension(snapshot.task_id),
                "budget_phase": _budget_phase(snapshot.budget_fraction),
                "action_space": snapshot.action_space,
                "history_policy": snapshot.history_policy,
            }
            groups[tuple(values[field] for field in fields)].append(branch)
        grouped_output[grouping_name] = [
            {
                "group": dict(zip(fields, key, strict=True)),
                "summaries": _summarize_branch_group(group_branches),
            }
            for key, group_branches in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0]))
        ]
    return {
        "schema_version": BRANCH_SCHEMA_VERSION,
        "replay_process_environment": replay_process_environment(),
        "n_snapshots": len(report.snapshots),
        "n_branches": len(report.branches),
        "actions": list(report.actions),
        "horizons": list(report.horizons),
        "overall": [asdict(summary) for summary in report.summaries],
        "groupings": grouped_output,
    }


def write_branch_csv(report: SnapshotBranchReport, path: Path) -> None:
    """Write tidy per-snapshot/action/horizon branch outcomes."""
    rows = branch_rows(report)
    if not rows:
        raise ValueError("Cannot write an empty branch report.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_branch_summary(report: SnapshotBranchReport, path: Path) -> None:
    """Write overall and stratified dynamic-headroom summaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summarize_branch_report(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve_snapshot_references(
    snapshots: Sequence[Any],
    explicit_references: Mapping[str, float] | None,
) -> dict[str, float]:
    """Resolve references from portable records and validate optional overrides."""
    portable: dict[str, float] = {}
    for snapshot in snapshots:
        if snapshot.reference_value is None:
            continue
        value = float(snapshot.reference_value)
        previous = portable.setdefault(snapshot.task_id, value)
        if not np.isclose(previous, value, rtol=0.0, atol=0.0):
            raise ValueError(f"Portable snapshots disagree on the reference for task {snapshot.task_id!r}.")

    resolved = dict(portable if explicit_references is None else explicit_references)
    for task_id, portable_value in portable.items():
        if task_id in resolved and not np.isclose(float(resolved[task_id]), portable_value, rtol=0.0, atol=0.0):
            raise ValueError(f"Explicit and portable references disagree for task {task_id!r}.")
    return resolved


def run_saved_snapshot_branches(
    snapshot_path: Path,
    *,
    env_factory: Callable[[str, int, str], Any],
    reference_values: Mapping[str, float] | None = None,
    forbidden_task_ids: Collection[str],
    output_csv: Path,
    output_summary: Path,
    horizons: Sequence[int] = DEFAULT_BRANCH_HORIZONS,
    verify_replay: bool = True,
) -> SnapshotBranchReport:
    """Load, guard, replay, branch, and persist one portable snapshot panel."""
    snapshots = read_snapshots(snapshot_path)
    forbidden = set(forbidden_task_ids) | set(sealed_final_test_task_ids())
    prohibited = sorted({snapshot.task_id for snapshot in snapshots} & forbidden)
    if prohibited:
        raise ValueError(f"Snapshot diagnostic refuses forbidden/test task IDs: {prohibited!r}.")
    invalid_frequency = sorted(
        {snapshot.interaction_frequency for snapshot in snapshots if snapshot.interaction_frequency != 1}
    )
    if invalid_frequency:
        raise ValueError(
            "Exact H=[1,5,10] action branching requires interaction_frequency=1 snapshots; "
            f"found {invalid_frequency!r}."
        )
    resolved_references = _resolve_snapshot_references(snapshots, reference_values)
    missing = sorted({snapshot.task_id for snapshot in snapshots} - set(resolved_references))
    if missing:
        raise ValueError(f"No portable or explicit objective reference was supplied for snapshot tasks: {missing!r}.")
    replay_factory = bind_snapshot_action_space_factory(snapshots, env_factory)
    if verify_replay:
        for snapshot in snapshots:
            verify_portable_snapshot_replay(snapshot, replay_factory)
    report = run_snapshot_branch_diagnostic(
        snapshots,
        replay_factory,
        lambda task_id: float(resolved_references[task_id]),
        forbidden_task_ids=forbidden,
        horizons=horizons,
    )
    write_branch_csv(report, output_csv)
    write_branch_summary(report, output_summary)
    return report


def _load_callable(specification: str) -> Callable[..., Any]:
    try:
        module_name, attribute_name = specification.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("Factory must use the form 'python.module:callable'.") from error
    target = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(target):
        raise TypeError(f"Factory target {specification!r} is not callable.")
    return target


def _load_payload(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return json.loads(path.read_text(encoding="utf-8"))


def _load_task_ids(path: Path) -> set[str]:
    payload = _load_payload(path)
    if isinstance(payload, Mapping):
        payload = payload.get("task_ids")
    if not isinstance(payload, list) or any(not isinstance(task_id, str) for task_id in payload):
        raise ValueError(f"{path} must contain a list or a manifest object with task_ids.")
    return set(payload)


def _load_references(path: Path) -> dict[str, float]:
    payload = _load_payload(path)
    if isinstance(payload, Mapping) and "references" in payload:
        payload = payload["references"]
    if isinstance(payload, Mapping):
        rows = []
        for task_id, value in payload.items():
            row = dict(value) if isinstance(value, Mapping) else {"value": value}
            row.setdefault("task_id", task_id)
            rows.append(row)
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"{path} must contain task-keyed references or reference rows.")
    references: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, Mapping) or "task_id" not in row:
            raise ValueError("Every reference row must contain task_id.")
        task_id = str(row["task_id"])
        value = row.get("value", row.get("reference_runtime_value", row.get("reference_in_runtime_units")))
        if task_id in references:
            raise ValueError(f"Duplicate explicit objective reference for {task_id!r}.")
        references[task_id] = float(value)
    return references


def main(argv: Sequence[str] | None = None) -> int:
    """Run saved-snapshot branching from explicit scientific inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument(
        "--factory",
        required=True,
        help="Callable module:attribute accepting (task_id, inner_seed, action_space_name).",
    )
    parser.add_argument(
        "--references",
        type=Path,
        help="Optional external reference table; otherwise use validated values embedded in the snapshots.",
    )
    parser.add_argument("--forbidden-task-ids", type=Path, required=True)
    parser.add_argument("--horizon", type=int, action="append", default=[])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    arguments = parser.parse_args(argv)
    require_deterministic_replay_process_environment()
    run_saved_snapshot_branches(
        arguments.snapshots,
        env_factory=_load_callable(arguments.factory),
        reference_values=None if arguments.references is None else _load_references(arguments.references),
        forbidden_task_ids=_load_task_ids(arguments.forbidden_task_ids),
        output_csv=arguments.output_csv,
        output_summary=arguments.output_summary,
        horizons=arguments.horizon or DEFAULT_BRANCH_HORIZONS,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
