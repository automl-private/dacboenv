"""Validate Stage-A roots and print immutable checkpoint identities."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dacboenv.experiment.evaluation_determinism import file_sha256, require_process_determinism
from dacboenv.experiment.unified_evaluator import inspect_stage_a_run


def main() -> None:
    """Fail closed unless every requested Stage-A root is complete."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    args = parser.parse_args()
    contract = require_process_determinism()
    rows = []
    for run_root in args.run_roots:
        artifact = inspect_stage_a_run(run_root)
        row = asdict(artifact)
        for key, value in tuple(row.items()):
            if isinstance(value, Path):
                row[key] = str(value)
        row["best_model_sha256"] = file_sha256(artifact.best_model)
        row["final_model_sha256"] = file_sha256(artifact.final_model)
        rows.append(row)
    print(json.dumps({"process_contract": contract, "stage_a_runs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
