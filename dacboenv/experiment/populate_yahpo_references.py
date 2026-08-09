"""Populate YAHPO references from the explicit accuracy-bound convention.

This command performs no objective evaluations.  It records the protocol
owner's assumption that the best accuracy is 100 on percentage-valued
scenarios and 1 on fraction-valued scenarios.  CARP-S minimizes negated
accuracy, so the corresponding runtime references are -100 and -1.
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

from dacboenv.experiment.yahpo_protocol import installed_yahpo_inventory, yahpo_data_identity, yahpo_task_id
from dacboenv.reference import ASSUMED_BOUND_SOURCE_METHOD

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "dacboenv/experiment/analysis/yahpo_best_known_references.json"
SCENARIO_ACCURACY_SCALE = {
    "lcbench": 100.0,
    "nb301": 100.0,
    "rbv2_glmnet": 1.0,
    "rbv2_ranger": 1.0,
    "rbv2_rpart": 1.0,
    "rbv2_super": 1.0,
    "rbv2_xgboost": 1.0,
}
TARGET_BY_SCENARIO = {
    "lcbench": "val_accuracy",
    "nb301": "val_accuracy",
    "rbv2_glmnet": "acc",
    "rbv2_ranger": "acc",
    "rbv2_rpart": "acc",
    "rbv2_super": "acc",
    "rbv2_xgboost": "acc",
}


def _git_head() -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise RuntimeError("git is required to record reference-table source provenance.")
    result = subprocess.run(  # noqa: S603
        [git_executable, "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_identity() -> tuple[str, str]:
    policy = {
        "method": ASSUMED_BOUND_SOURCE_METHOD,
        "runtime_objective_transform": "negative_accuracy",
        "reporting_objective_transform": "one_minus_accuracy",
        "scenario_accuracy_scale": SCENARIO_ACCURACY_SCALE,
        "assumption_authority": "user_specified_protocol",
    }
    canonical = json.dumps(policy, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def assumed_reference(
    scenario: str,
    instance: str,
    *,
    generation_date: str,
    data_identity: dict[str, str],
    source_hash: str,
    source_content_hash: str,
    source_code_commit: str,
) -> dict[str, Any]:
    """Return one explicit, non-empirical assumed-bound reference row."""
    scale = SCENARIO_ACCURACY_SCALE[scenario]
    runtime_value = -scale
    return {
        "task_id": yahpo_task_id(scenario, instance),
        "value": runtime_value,
        "kind": "best_known",
        "runtime_objective_transform": "negative_accuracy",
        "reporting_objective_transform": "one_minus_accuracy",
        "fidelity": "fixed_maximum",
        "source": f"dacboenv.protocol:{ASSUMED_BOUND_SOURCE_METHOD}",
        "source_hash": source_hash,
        "benchmark_code_version": f"carps={version('carps')};yahpo_gym={version('yahpo_gym')}",
        "benchmark_data_version": f"{data_identity['version']};git={data_identity['git_commit']}",
        "tolerance": 1e-8,
        "metadata": {
            "scenario": scenario,
            "instance": str(instance),
            "objective_target": TARGET_BY_SCENARIO[scenario],
            "reporting_value": 0.0,
            "source_method": ASSUMED_BOUND_SOURCE_METHOD,
            "source_seeds": [],
            "source_evaluation_budget": 0,
            "generation_date": generation_date,
            "benchmark_data_hash": data_identity["config_space_tree_sha256"],
            "runtime_units": "negative_accuracy",
            "reporting_units": "one_minus_accuracy",
            "maximum_fidelity": True,
            "provenance_status": "complete",
            "reference_basis": "assumed_metric_upper_bound",
            "assumed_accuracy_upper_bound": scale,
            "assumption_authority": "user_specified_protocol",
            "empirical": False,
            "exactness_proved": False,
            "source_code_commit": source_code_commit,
            "source_content_path": Path(__file__).relative_to(REPOSITORY_ROOT).as_posix(),
            "source_content_sha256": source_content_hash,
        },
    }


def build_reference_table(generation_date: str) -> dict[str, Any]:
    """Enumerate every installed supported YAHPO task and assign its bound."""
    inventory = installed_yahpo_inventory(tuple(SCENARIO_ACCURACY_SCALE))
    data_identity = yahpo_data_identity()
    source_hash, source_content_hash = _source_identity()
    source_code_commit = _git_head()
    references = [
        assumed_reference(
            scenario,
            instance,
            generation_date=generation_date,
            data_identity=data_identity,
            source_hash=source_hash,
            source_content_hash=source_content_hash,
            source_code_commit=source_code_commit,
        )
        for scenario in SCENARIO_ACCURACY_SCALE
        for instance in inventory[scenario]
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "reference_convention": {
            "kind": "best_known",
            "basis": "assumed_metric_upper_bound",
            "empirical": False,
            "exactness_proved": False,
            "runtime_values": {"percentage_accuracy": -100.0, "fraction_accuracy": -1.0},
        },
        "references": references,
    }


def main() -> None:
    """Write the complete installed-task reference table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generation-date", default=datetime.now(UTC).date().isoformat())
    args = parser.parse_args()
    table = build_reference_table(args.generation_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(table['references'])} assumed YAHPO references to {args.output}")


if __name__ == "__main__":
    main()
