"""Build exact paired-evaluator context inventories from frozen manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from carps.utils.running import make_task

from dacboenv.experiment.evaluation_determinism import (
    EVALUATION_PROTOCOL_VERSION,
    context_inventory_hash,
)
from dacboenv.experiment.paired_evaluator import EvaluationContext, authorize_manifest_execution
from dacboenv.experiment.protocol import load_manifest
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider
from dacboenv.utils.carps_optimizer import get_task_config

_SUPPORTED_INTERACTION_FREQUENCIES = (1, 5, 10)


def _bbob_reference_and_budget(task_id: str) -> tuple[Any, int]:
    cfg = get_task_config(task_id)
    cfg.seed = 0
    task = make_task(cfg=cfg)
    reference = BBOBExactReferenceProvider().get_reference(
        task_id,
        task.objective_function,
        {
            "runtime_objective_transform": "identity",
            "reporting_objective_transform": "identity",
            "fidelity": "not_applicable",
        },
    )
    return reference, int(cfg.task.optimization_resources.n_trials)


def build_evaluation_contexts(
    manifest: dict[str, Any],
    *,
    interaction_frequency: int,
    reference_table: Path | None = None,
    allow_sealed_test: bool = False,
) -> list[EvaluationContext]:
    """Expand a frozen manifest without evaluating an objective."""
    authorize_manifest_execution(manifest, allow_sealed_test=allow_sealed_test)
    if interaction_frequency not in _SUPPORTED_INTERACTION_FREQUENCIES:
        raise ValueError(
            f"interaction_frequency must be one of {_SUPPORTED_INTERACTION_FREQUENCIES}, got {interaction_frequency!r}."
        )
    yahpo_provider = None
    if any(str(task_id).lower().startswith("yahpo/") for task_id in manifest["task_ids"]):
        if reference_table is None:
            raise ValueError("YAHPO/mixed context generation requires --reference-table.")
        yahpo_provider = ManifestReferenceProvider(
            reference_table,
            expected_runtime_objective_transform="negative_accuracy",
            expected_reporting_objective_transform="one_minus_accuracy",
            expected_fidelity="fixed_maximum",
        )

    task_metadata: dict[str, tuple[Any, int, str, int | None, str]] = {}
    for raw_task_id in manifest["task_ids"]:
        task_id = str(raw_task_id)
        parts = task_id.split("/")
        if task_id.startswith("bbob/") and len(parts) == 4:  # noqa: PLR2004
            reference, budget = _bbob_reference_and_budget(task_id)
            task_metadata[task_id] = (reference, budget, parts[2], int(parts[1]), parts[3])
        elif task_id.lower().startswith("yahpo/so/") and len(parts) == 5:  # noqa: PLR2004
            assert yahpo_provider is not None
            try:
                reference = yahpo_provider.references[task_id]
            except KeyError as error:
                raise ValueError(f"Reference table has no row for manifest task {task_id!r}.") from error
            cfg = get_task_config(task_id)
            budget = int(cfg.task.optimization_resources.n_trials)
            task_metadata[task_id] = (reference, budget, parts[2], None, parts[3])
        else:
            raise ValueError(f"Unsupported evaluation task ID {task_id!r}.")

    return [
        EvaluationContext(
            domain="bbob" if task_id.startswith("bbob/") else "yahpo",
            scenario_or_function=task_metadata[task_id][2],
            dimension=task_metadata[task_id][3],
            task_id=task_id,
            native_instance=task_metadata[task_id][4],
            inner_seed=int(seed),
            evaluation_budget=task_metadata[task_id][1],
            reference_kind=task_metadata[task_id][0].kind,
            reference_value=float(task_metadata[task_id][0].value),
            objective_transform=task_metadata[task_id][0].runtime_objective_transform,
            manifest_hash=str(manifest["manifest_hash"]),
            interaction_frequency=interaction_frequency,
            manifest_id=str(manifest["id"]),
            evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
        )
        for seed in manifest["inner_seeds"]
        for task_id in (str(value) for value in manifest["task_ids"])
    ]


def main() -> None:
    """Write a deterministic evaluator context JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--interaction-frequency", type=int, choices=_SUPPORTED_INTERACTION_FREQUENCIES, required=True)
    parser.add_argument("--reference-table", type=Path)
    parser.add_argument("--allow-sealed-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    contexts = build_evaluation_contexts(
        manifest,
        interaction_frequency=args.interaction_frequency,
        reference_table=args.reference_table,
        allow_sealed_test=args.allow_sealed_test,
    )
    serialized_contexts = [asdict(context) for context in contexts]
    payload = {
        "schema_version": 2,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "manifest_id": manifest["id"],
        "manifest_hash": manifest["manifest_hash"],
        "source_revision": current_source_revision(),
        "context_inventory_hash": context_inventory_hash(serialized_contexts),
        "contexts": serialized_contexts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
