"""Canonical deployment context and task-partition helpers."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict

from dacboenv.selective_wei.schemas import StaticContext

YAHPO_SCENARIOS = frozenset({"lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost"})
BBOB_TASK_PARTS = 4
YAHPO_TASK_PARTS = 5
MINIMUM_THREE_WAY_STRATUM = 3


def canonical_hash(value: object) -> str:
    """Hash one JSON-compatible scientific object canonically."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def context_from_task_id(task_id: str, *, phase_bin: int | None = None) -> StaticContext:
    """Parse only stable deployable context from a native task identifier."""
    parts = task_id.split("/")
    if len(parts) == BBOB_TASK_PARTS and parts[0] == "bbob":
        context = StaticContext(domain="bbob", dimension=int(parts[1]), phase_bin=phase_bin)
    elif len(parts) == YAHPO_TASK_PARTS and parts[:2] == ["yahpo", "so"] and parts[2] in YAHPO_SCENARIOS:
        context = StaticContext(domain="yahpo", scenario=parts[2], phase_bin=phase_bin)
    elif parts[0] == "optbench" or len(parts) == 1:
        from dacboenv.utils.carps_optimizer import get_optbench_task_dimension  # noqa: PLC0415 - optional runtime

        canonical = task_id if parts[0] == "optbench" else f"optbench/{task_id}"
        context = StaticContext(
            domain="optbench", dimension=get_optbench_task_dimension(canonical), phase_bin=phase_bin
        )
    else:
        raise ValueError(f"Unsupported or protected task identifier {task_id!r}.")
    context.validate()
    return context


def exact_context_key(context: StaticContext, *, include_phase: bool = False) -> str:
    """Return the primary scenario/dimension routing key."""
    context.validate()
    key = (
        f"{context.domain}:dimension:{context.dimension}"
        if context.domain in {"bbob", "optbench"}
        else f"yahpo:scenario:{context.scenario}"
    )
    if include_phase:
        if context.phase_bin is None:
            raise ValueError("Phase ablation requires an explicit phase bin.")
        key += f":phase:{context.phase_bin}"
    return key


def domain_context_key(context: StaticContext) -> str:
    """Return a domain-only fallback key."""
    return f"domain:{context.domain}"


def stratification_key(task_id: str) -> str:
    """Use dimension for BBOB and scenario for YAHPO partitioning."""
    context = context_from_task_id(task_id)
    return exact_context_key(context)


def selective_train_partition(
    task_ids: Iterable[str],
    *,
    seed: int = 20260830,
) -> dict[str, object]:
    """Create a frozen task-disjoint 60/20/20 fit/tune/calibration split."""
    supplied = list(task_ids)
    unique = sorted(set(supplied))
    if len(unique) != len(supplied):
        raise ValueError("Selective partition input contains duplicate task IDs.")
    strata: dict[str, list[str]] = defaultdict(list)
    for task in unique:
        strata[stratification_key(task)].append(task)
    partitions: dict[str, list[str]] = {
        "model_fit": [],
        "gate_tune": [],
        "uncertainty_calibration": [],
    }
    for stratum, tasks in sorted(strata.items()):
        ordered = sorted(tasks, key=lambda task: canonical_hash({"seed": seed, "stratum": stratum, "task": task}))
        n = len(ordered)
        tune = max(1, round(0.2 * n)) if n >= MINIMUM_THREE_WAY_STRATUM else 0
        calibration = max(1, round(0.2 * n)) if n >= MINIMUM_THREE_WAY_STRATUM else 0
        if tune + calibration >= n:
            calibration = max(0, n - tune - 1)
        fit = n - tune - calibration
        partitions["model_fit"].extend(ordered[:fit])
        partitions["gate_tune"].extend(ordered[fit : fit + tune])
        partitions["uncertainty_calibration"].extend(ordered[fit + tune :])
    normalized = {key: sorted(value) for key, value in partitions.items()}
    flat = [task for values in normalized.values() for task in values]
    if sorted(flat) != unique or len(flat) != len(set(flat)):
        raise RuntimeError("Selective task partition is not complete and disjoint.")
    payload: dict[str, object] = {
        "schema_version": "dacbo-selective-train-partition-v1",
        "seed": seed,
        "stratification": "bbob_dimension__yahpo_scenario",
        "partitions": normalized,
        "strata": {key: sorted(value) for key, value in strata.items()},
    }
    payload["partition_hash"] = canonical_hash(payload)
    return payload


def context_payload(context: StaticContext) -> Mapping[str, object]:
    """Return a stable JSON mapping for context provenance."""
    return asdict(context)
