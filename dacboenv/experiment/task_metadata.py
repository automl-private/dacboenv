"""Canonical, fail-closed benchmark task identifier metadata."""

from __future__ import annotations

from dataclasses import dataclass

EXPECTED_YAHPO_SCENARIOS = frozenset(
    {"lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost"}
)
BBOB_PARTS = 4
YAHPO_PARTS = 5


@dataclass(frozen=True)
class TaskMetadata:
    """Normalized task identity without objective or reference information."""

    domain: str
    scenario: str | None
    dataset_instance: str | None
    function_id: int | None
    dimension: int | None
    native_instance: str


def parse_task_metadata(task_id: str) -> TaskMetadata:
    """Parse one exact DACBO task ID; reject unknown and sealed nb301 forms."""
    parts = task_id.split("/")
    if len(parts) == BBOB_PARTS and parts[0].lower() == "bbob":
        dimension, function, instance = (int(value) for value in parts[1:])
        return TaskMetadata("bbob", None, None, function, dimension, str(instance))
    if len(parts) == YAHPO_PARTS and parts[:2] == ["yahpo", "so"]:
        scenario, dataset_instance, fidelity = parts[2], parts[3], parts[4:]
        if scenario == "nb301" or scenario not in EXPECTED_YAHPO_SCENARIOS:
            raise PermissionError(f"Unsupported or sealed YAHPO scenario in task ID: {task_id!r}")
        if not dataset_instance:
            raise ValueError(f"YAHPO dataset instance is empty in task ID: {task_id!r}")
        native_instance = "/".join((dataset_instance, *fidelity))
        return TaskMetadata("yahpo", scenario, dataset_instance, None, None, native_instance)
    raise ValueError(f"Unsupported canonical task ID: {task_id!r}")
