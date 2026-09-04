"""Validate the installed OptBench inventory used for exact-reference training."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.running import make_task

from dacboenv.experiment.protocol import load_manifest
from dacboenv.utils.carps_optimizer import (
    get_installed_optbench_task_configs,
    get_optbench_task_dimension,
    get_task_config,
)

_MANIFEST_PATH = Path(__file__).resolve().parents[1] / "configs" / "instance_sets" / "optbench_train.yaml"


@dataclass(frozen=True, slots=True)
class OptBenchReferenceRecord:
    """One installed OptBench task's live optimum availability."""

    task_id: str
    dimension: int
    objective_class: str
    f_min: float | None
    has_finite_x_min: bool
    selected: bool


def audit_optbench_inventory(manifest_path: Path = _MANIFEST_PATH) -> dict[str, Any]:
    """Validate that the frozen manifest is exactly the finite-minimum subset.

    Parameters
    ----------
    manifest_path : Path
        OptBench instance-set manifest to validate.

    Returns
    -------
    dict[str, Any]
        JSON-compatible inventory and reference summary.

    Raises
    ------
    RuntimeError
        If an installed task and the frozen manifest disagree or a selected
        task does not expose a finite global minimizer/value.
    """
    manifest = load_manifest(manifest_path)
    selected = set(manifest["task_ids"])
    expected_values = manifest["selection_protocol"]["global_minima"]
    records: list[OptBenchReferenceRecord] = []
    eligible: set[str] = set()

    for task_id in sorted(get_installed_optbench_task_configs()):
        cfg = get_task_config(task_id)
        cfg.seed = 0
        objective = make_task(cfg).objective_function
        raw_minimum = getattr(objective, "f_min", None)
        f_min = float(raw_minimum) if raw_minimum is not None and math.isfinite(float(raw_minimum)) else None
        raw_x_min = getattr(objective, "x_min", None)
        dimension = get_optbench_task_dimension(task_id)
        has_finite_x_min = False
        if raw_x_min is not None:
            x_min = np.asarray(raw_x_min, dtype=np.float64)
            has_finite_x_min = x_min.shape == (dimension,) and bool(np.all(np.isfinite(x_min)))
        if f_min is not None and has_finite_x_min:
            eligible.add(task_id)
        records.append(
            OptBenchReferenceRecord(
                task_id=task_id,
                dimension=dimension,
                objective_class=type(objective).__name__,
                f_min=f_min,
                has_finite_x_min=has_finite_x_min,
                selected=task_id in selected,
            )
        )

    if selected != eligible:
        raise RuntimeError(
            "Frozen OptBench manifest differs from the installed finite-global-minimum inventory: "
            f"missing={sorted(eligible - selected)}, ineligible={sorted(selected - eligible)}."
        )
    for record in records:
        if not record.selected:
            continue
        expected = float(expected_values[record.task_id])
        if record.f_min is None or not math.isclose(record.f_min, expected, rel_tol=0.0, abs_tol=1e-10):
            raise RuntimeError(
                f"Frozen f_min for {record.task_id!r} is {expected}, but the installed objective exposes "
                f"{record.f_min}."
            )

    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_hash": manifest["manifest_hash"],
        "selected_task_count": len(selected),
        "excluded_task_count": len(records) - len(selected),
        "records": [asdict(record) for record in records],
    }


def main() -> None:
    """Print a successful installed-inventory audit or exit nonzero."""
    print(json.dumps(audit_optbench_inventory(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
