"""Fail-closed selective-data and split provenance checks."""

from __future__ import annotations

from typing import Any

from dacboenv.experiment.protocol import sealed_final_test_task_ids
from dacboenv.offline.provenance import reject_training_provenance
from dacboenv.selective_wei.context import canonical_hash


def validate_selective_branch_provenance(metadata: dict[str, Any], *, role: str) -> None:
    """Reject holdout, test, and learned-policy validation headroom labels."""
    serialized = str(metadata).lower()
    forbidden = ("learned_policy_validation_headroom", "d1_f5_learned_headroom", "context_split=test", "nb301")
    if any(token in serialized for token in forbidden):
        raise ValueError("Selective-WEI fitting rejects learned-policy/test branch provenance.")
    tasks = set(map(str, metadata.get("task_ids", [])))
    if tasks & sealed_final_test_task_ids():
        raise ValueError("Selective-WEI data contain sealed test tasks.")
    if role in {"model_fit", "gate_tune", "uncertainty_calibration"}:
        reject_training_provenance(metadata)
        if metadata.get("context_split", metadata.get("split")) != "train":
            raise ValueError("Selective fitting/tuning/calibration requires train-context branch data.")
    elif role == "dev":
        if metadata.get("context_split", metadata.get("split")) != "dev":
            raise ValueError("Selective development scoring requires the frozen dev branch split.")
    else:
        raise ValueError(f"Unknown selective data role {role!r}.")


def validate_task_partitions(partitions: dict[str, list[str]], train_tasks: list[str], dev_tasks: list[str]) -> str:
    """Prove fit/tune/calibration are disjoint train tasks and exclude dev."""
    keys = ("model_fit", "gate_tune", "uncertainty_calibration")
    sets = [set(partitions[key]) for key in keys]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise ValueError("Selective training partitions overlap.")
    if set().union(*sets) != set(train_tasks):
        raise ValueError("Selective training partitions do not exactly cover train tasks.")
    if set().union(*sets) & set(dev_tasks):
        raise ValueError("Development tasks entered selective fitting or calibration.")
    return canonical_hash({key: sorted(partitions[key]) for key in keys})
