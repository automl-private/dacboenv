"""Atomic terminal status records for scientific evaluation episodes."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Replace one JSON record atomically on the destination filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluation_cell_key(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Return the complete identity used by matrix audit and repair tools."""
    return {
        "method_id": str(cell["method_id"]),
        "task_id": str(cell["task_id"]),
        "evaluation_seed": int(cell["evaluation_seed"]),
        "checkpoint_mode": str(cell.get("checkpoint_mode", "none")),
        "model_sha256": cell.get("model_sha256"),
    }


def evaluation_cell_hash(cell: Mapping[str, Any]) -> str:
    """Hash a canonical evaluation matrix cell identity."""
    return canonical_sha256(evaluation_cell_key(cell))


@contextmanager
def episode_status(
    status_path: Path,
    *,
    cell: Mapping[str, Any],
    context_hash: str,
    result_path: Path,
) -> Iterator[dict[str, Any]]:
    """Write exactly one atomic success/failed terminal record and re-raise."""
    runtime: dict[str, Any] = {"objective_evaluations_completed": 0}
    try:
        yield runtime
    except BaseException as error:
        traceback_path = status_path.with_suffix(".traceback.txt")
        traceback_path.parent.mkdir(parents=True, exist_ok=True)
        traceback_path.write_text(traceback.format_exc(), encoding="utf-8")
        atomic_json(
            status_path,
            {
                **evaluation_cell_key(cell),
                "cell_hash": evaluation_cell_hash(cell),
                "context_hash": context_hash,
                "status": "failed",
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback_path": str(traceback_path.resolve()),
                "objective_evaluations_completed": int(runtime["objective_evaluations_completed"]),
                **{key: value for key, value in runtime.items() if key != "objective_evaluations_completed"},
                "result_path": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path) if result_path.is_file() else None,
            },
        )
        raise
    else:
        atomic_json(
            status_path,
            {
                **evaluation_cell_key(cell),
                "cell_hash": evaluation_cell_hash(cell),
                "context_hash": context_hash,
                "status": "success",
                "exception_type": None,
                "exception_message": None,
                "traceback_path": None,
                "objective_evaluations_completed": int(runtime["objective_evaluations_completed"]),
                **{key: value for key, value in runtime.items() if key != "objective_evaluations_completed"},
                "result_path": str(result_path.resolve()),
                "result_sha256": file_sha256(result_path) if result_path.is_file() else None,
            },
        )


__all__ = ["atomic_json", "episode_status", "evaluation_cell_hash", "evaluation_cell_key"]
