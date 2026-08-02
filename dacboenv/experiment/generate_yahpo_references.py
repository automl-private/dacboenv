"""Generate provenance-complete empirical YAHPO best-known references.

This is an explicit, bounded reference-generation command.  It never reads or
updates a running experiment's reference table, and it refuses the sealed
official final-test inventory by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from carps.objective_functions.yahpo import YahpoObjectiveFunction
from carps.utils.trials import TrialInfo

from dacboenv.experiment.collect_snapshots import current_git_commit
from dacboenv.experiment.yahpo_protocol import official_yahpo_task_ids, yahpo_data_identity

_TARGET_BY_SCENARIO = {
    "lcbench": "val_accuracy",
    "nb301": "val_accuracy",
    "rbv2_glmnet": "acc",
    "rbv2_ranger": "acc",
    "rbv2_rpart": "acc",
    "rbv2_super": "acc",
    "rbv2_xgboost": "acc",
}
_CANONICAL_TASK_PART_COUNT = 5


def _repository_root(path: Path) -> Path | None:
    for parent in (path, *path.parents):
        if (parent / ".git").exists():
            return parent
    return None


def source_provenance() -> dict[str, Any]:
    """Describe the exact generator source and whether a commit contains it."""
    source_path = Path(__file__).resolve()
    source_content_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    repository = _repository_root(source_path.parent)
    git_executable = shutil.which("git")
    if repository is None or git_executable is None:
        return {
            "source_code_commit": "unavailable",
            "source_code_commit_contains_method": False,
            "source_content_path": source_path.name,
            "source_content_sha256": source_content_sha256,
            "source_repository_status": "unavailable",
            "source_repository_status_sha256": hashlib.sha256(b"unavailable").hexdigest(),
            "provenance_status": "smoke_only_incomplete",
        }

    relative_source = source_path.relative_to(repository).as_posix()
    status = subprocess.run(  # noqa: S603
        [git_executable, "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    committed_source = subprocess.run(  # noqa: S603
        [git_executable, "show", f"HEAD:{relative_source}"],
        cwd=repository,
        check=False,
        capture_output=True,
    )
    commit_contains_method = (
        committed_source.returncode == 0
        and hashlib.sha256(committed_source.stdout).hexdigest() == source_content_sha256
    )
    repository_status = "clean" if not status else "dirty"
    provenance_status = (
        "complete" if repository_status == "clean" and commit_contains_method else "smoke_only_incomplete"
    )
    return {
        "source_code_commit": current_git_commit(repository),
        "source_code_commit_contains_method": commit_contains_method,
        "source_content_path": relative_source,
        "source_content_sha256": source_content_sha256,
        "source_repository_status": repository_status,
        "source_repository_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "provenance_status": provenance_status,
    }


def parse_yahpo_task_id(task_id: str) -> tuple[str, str]:
    """Parse ``yahpo/so/<scenario>/<instance>/None``."""
    parts = task_id.split("/")
    if (
        len(parts) != _CANONICAL_TASK_PART_COUNT
        or parts[0].lower() != "yahpo"
        or parts[1].lower() != "so"
        or parts[4] != "None"
    ):
        raise ValueError(f"Expected canonical YAHPO-SO task ID, got {task_id!r}.")
    scenario, instance = parts[2], parts[3]
    if scenario not in _TARGET_BY_SCENARIO:
        raise ValueError(f"Unsupported YAHPO-SO scenario {scenario!r}.")
    return scenario, instance


def _reporting_value(scenario: str, runtime_value: float) -> float:
    scale = 100.0 if scenario in {"lcbench", "nb301"} else 1.0
    return 1.0 + runtime_value / scale


def generate_reference(
    task_id: str,
    *,
    source_seeds: tuple[int, ...],
    evaluations_per_seed: int,
    generation_date: str,
    allow_official_test: bool = False,
) -> dict[str, Any]:
    """Run deterministic seeded random search and return one strict provider row."""
    if task_id in official_yahpo_task_ids() and not allow_official_test:
        raise ValueError(f"Reference generation refuses sealed official YAHPO test task {task_id!r}.")
    if not source_seeds or evaluations_per_seed <= 0:
        raise ValueError("At least one source seed and a positive evaluations_per_seed are required.")
    scenario, instance = parse_yahpo_task_id(task_id)
    target = _TARGET_BY_SCENARIO[scenario]
    observations: list[dict[str, Any]] = []
    best_value = np.inf
    for source_seed in source_seeds:
        objective = YahpoObjectiveFunction(
            bench=scenario,
            instance=instance,
            metric=[target],
            budget_type=None,
            seed=int(source_seed),
        )
        objective.configspace.seed(int(source_seed))
        for evaluation_index in range(evaluations_per_seed):
            configuration = objective.configspace.sample_configuration()
            trial_value = objective.evaluate(TrialInfo(config=configuration, seed=int(source_seed)))
            value = float(np.asarray(trial_value.cost, dtype=float).reshape(-1)[0])
            if not np.isfinite(value):
                raise RuntimeError(f"YAHPO returned non-finite cost for {task_id!r} at trial {evaluation_index}.")
            observations.append(
                {
                    "source_seed": int(source_seed),
                    "evaluation_index": evaluation_index,
                    "configuration": dict(configuration),
                    "runtime_value": value,
                }
            )
            best_value = min(best_value, value)

    source_payload = json.dumps(observations, allow_nan=False, separators=(",", ":"), sort_keys=True)
    source_hash = hashlib.sha256(source_payload.encode("utf-8")).hexdigest()
    data_identity = yahpo_data_identity()
    generation_source = source_provenance()
    return {
        "task_id": task_id,
        "value": float(best_value),
        "kind": "best_known",
        "runtime_objective_transform": "negative_accuracy",
        "reporting_objective_transform": "one_minus_accuracy",
        "fidelity": "fixed_maximum",
        "source": "dacboenv.experiment.generate_yahpo_references:seeded_random_search_v1",
        "source_hash": source_hash,
        "benchmark_code_version": f"carps={version('carps')};yahpo_gym={version('yahpo_gym')}",
        "benchmark_data_version": f"{data_identity['version']};git={data_identity['git_commit']}",
        "tolerance": 1e-8,
        "metadata": {
            "scenario": scenario,
            "instance": instance,
            "objective_target": target,
            "reporting_value": _reporting_value(scenario, float(best_value)),
            "source_method": "seeded_random_search_v1",
            "source_seeds": list(source_seeds),
            "source_evaluation_budget": len(source_seeds) * evaluations_per_seed,
            "generation_date": generation_date,
            "benchmark_data_hash": data_identity["config_space_tree_sha256"],
            "runtime_units": "negative_accuracy",
            "reporting_units": "one_minus_accuracy",
            "maximum_fidelity": True,
            "source_evaluation_trace_sha256": source_hash,
            **generation_source,
        },
    }


def main() -> None:
    """CLI entry point for an explicit, bounded generation campaign."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--seed", action="append", type=int, default=[])
    parser.add_argument("--evaluations-per-seed", type=int, default=128)
    parser.add_argument("--generation-date", default=datetime.now(UTC).date().isoformat())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-official-test", action="store_true", help="Reserved for an authorized final stage.")
    args = parser.parse_args()
    seeds = tuple(args.seed or (1_947_521, 2_651_337, 3_771_619))
    references = [
        generate_reference(
            task_id,
            source_seeds=seeds,
            evaluations_per_seed=args.evaluations_per_seed,
            generation_date=args.generation_date,
            allow_official_test=args.allow_official_test,
        )
        for task_id in args.task_id
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    statuses = {row["metadata"]["provenance_status"] for row in references}
    table_status = "complete" if statuses == {"complete"} else "smoke_only_incomplete"
    args.output.write_text(
        json.dumps(
            {"schema_version": 1, "status": table_status, "references": references},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(references)} reference row(s) to {args.output}")


if __name__ == "__main__":
    main()
