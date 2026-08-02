"""Merge paired evaluator outputs without collapsing outer PPO seeds."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from dacboenv.experiment.paired_evaluator import (
    EvaluationRecord,
    MethodCell,
    available_method_cells,
    hierarchical_paired_bootstrap,
    paired_method_comparison,
)

_OPTIONAL_INT_FIELDS = ("dimension", "outer_ppo_seed")
_INT_FIELDS = ("inner_seed", "evaluation_budget", "interaction_frequency")
_FLOAT_FIELDS = (
    "reference_value",
    "final_incumbent",
    "final_reference_regret",
    "normalized_final_regret",
    "anytime_auc",
    "episode_return",
    "deterministic_switch_rate",
    "runtime_seconds",
)
_BOOL_FIELDS = ("constant_policy",)


def _parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid boolean in evaluator CSV: {value!r}.")


def load_evaluation_records(path: Path) -> list[EvaluationRecord]:
    """Load the stable tidy schema with explicit type restoration."""
    with path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    records: list[EvaluationRecord] = []
    for row in rows:
        values: dict[str, Any] = dict(row)
        for field_name in _OPTIONAL_INT_FIELDS:
            values[field_name] = None if values[field_name] == "" else int(values[field_name])
        for field_name in _INT_FIELDS:
            values[field_name] = int(values[field_name])
        for field_name in _FLOAT_FIELDS:
            values[field_name] = float(values[field_name])
        for field_name in _BOOL_FIELDS:
            values[field_name] = _parse_bool(values[field_name])
        values["action_histogram"] = tuple(int(value) for value in json.loads(values["action_histogram"]))
        records.append(EvaluationRecord(**values))
    return records


def _scientific_payload(record: EvaluationRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload.pop("runtime_seconds")
    payload.pop("code_commit")
    return payload


def merge_evaluation_records(record_sets: list[list[EvaluationRecord]]) -> list[EvaluationRecord]:
    """Deduplicate shared baselines and reject conflicting repeated cells."""
    merged: dict[tuple[MethodCell, Any], EvaluationRecord] = {}
    for records in record_sets:
        for record in records:
            key = (record.method_cell, record.context_key)
            previous = merged.get(key)
            if previous is not None and _scientific_payload(previous) != _scientific_payload(record):
                raise ValueError(f"Conflicting duplicate evaluator result for {key!r}.")
            merged.setdefault(key, record)
    return list(merged.values())


def aggregate_outer_seed_results(
    records: list[EvaluationRecord],
    *,
    learned_method: str,
    action_family: str,
    checkpoint_type: str,
    baseline_methods: list[str],
    n_resamples: int,
) -> dict[str, Any]:
    """Compute one cross-seed probability and paired summary per baseline."""
    cells = available_method_cells(records)
    learned_cells = [
        cell
        for cell in cells
        if cell.method == learned_method
        and cell.action_family == action_family
        and cell.checkpoint_type == checkpoint_type
        and cell.outer_ppo_seed is not None
    ]
    if not learned_cells:
        raise ValueError("No learned outer-seed cells match the requested selector.")
    learned_by_seed = {int(cell.outer_ppo_seed): cell for cell in learned_cells}
    if len(learned_by_seed) != len(learned_cells):
        raise ValueError("Duplicate learned method cell for one outer PPO seed.")

    baseline_cells: dict[str, tuple[dict[int, MethodCell], MethodCell | None]] = {}
    for method in baseline_methods:
        matches = [cell for cell in cells if cell.method == method]
        shared = matches[0] if len(matches) == 1 and matches[0].outer_ppo_seed is None else None
        if shared is not None:
            baseline_cells[method] = (dict.fromkeys(learned_by_seed, shared), shared)
            continue
        matched = {
            int(cell.outer_ppo_seed): cell
            for cell in matches
            if cell.outer_ppo_seed is not None
            and cell.action_family == action_family
            and cell.checkpoint_type == checkpoint_type
        }
        if len(matched) != len(matches) or set(matched) != set(learned_by_seed):
            raise ValueError(
                f"Baseline {method!r} must be one shared outer-seed-free cell or have exactly one "
                f"action-family/checkpoint-matched cell for learned seeds {sorted(learned_by_seed)}; "
                f"found {matches!r}."
            )
        baseline_cells[method] = (matched, None)

    summaries: dict[str, Any] = {}
    for method, (baseline_by_seed, shared_baseline) in baseline_cells.items():
        per_seed = []
        mean_by_seed: dict[int, float] = {}
        for seed, cell in sorted(learned_by_seed.items()):
            baseline_cell = baseline_by_seed[seed]
            comparison = paired_method_comparison(records, cell, baseline_cell)
            bootstrap = hierarchical_paired_bootstrap(
                records,
                cell,
                baseline_cell,
                n_resamples=n_resamples,
                seed=seed,
            )
            mean_by_seed[seed] = comparison.mean_difference
            per_seed.append({"comparison": asdict(comparison), "hierarchical_bootstrap": asdict(bootstrap)})
        tolerance = 1e-12
        values = tuple(mean_by_seed.values())
        probability = {
            "method": learned_method,
            "action_family": action_family,
            "checkpoint_type": checkpoint_type,
            "baseline_cell": None if shared_baseline is None else asdict(shared_baseline),
            "baseline_cells_by_outer_seed": (
                {}
                if shared_baseline is not None
                else {str(seed): asdict(cell) for seed, cell in baseline_by_seed.items()}
            ),
            "metric": "normalized_final_regret",
            "n_outer_seeds": len(values),
            "beating_seeds": sum(value > tolerance for value in values),
            "tying_seeds": sum(abs(value) <= tolerance for value in values),
            "losing_seeds": sum(value < -tolerance for value in values),
            "probability_beating": sum(value > tolerance for value in values) / len(values),
            "mean_difference_by_seed": dict(sorted(mean_by_seed.items())),
        }
        summaries[method] = {"outer_seed_probability": probability, "per_outer_seed": per_seed}
    return {
        "schema_version": 1,
        "learned_method": learned_method,
        "action_family": action_family,
        "checkpoint_type": checkpoint_type,
        "outer_seeds": sorted(int(cell.outer_ppo_seed) for cell in learned_cells),
        "baselines": summaries,
    }


def main() -> None:
    """Merge explicit run outputs and write cross-outer-seed statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--learned-method", required=True)
    parser.add_argument("--action-family", required=True)
    parser.add_argument("--checkpoint-type", choices=("best", "final"), required=True)
    parser.add_argument("--baseline-methods", nargs="+", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged = merge_evaluation_records([load_evaluation_records(path) for path in args.records])
    payload = aggregate_outer_seed_results(
        merged,
        learned_method=args.learned_method,
        action_family=args.action_family,
        checkpoint_type=args.checkpoint_type,
        baseline_methods=args.baseline_methods,
        n_resamples=args.bootstrap_resamples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
