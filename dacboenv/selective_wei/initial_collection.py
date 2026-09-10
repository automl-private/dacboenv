"""Five-arm static continuations through the existing CARP-S DACBO interface."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np

from dacboenv.experiment.collect_snapshots import completed_evaluations
from dacboenv.selective_wei.artifacts import atomic_json
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_base_data import PairedStaticLabels
from dacboenv.selective_wei.initial_context import InitialContext
from dacboenv.selective_wei.initial_features import InitialFeatureSettings

DEFAULT_FEATURE_SETTINGS = InitialFeatureSettings()
INTERACTION_FREQUENCY = 5


def read_verified(path: Path) -> dict[str, Any]:
    """Load a shard only if its complete canonical content hash matches."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    digest = payload.pop("content_hash")
    if canonical_hash(payload) != digest:
        raise ValueError(f"Corrupt initial-base shard: {path}.")
    return cast("dict[str, Any]", payload)


def write_verified(path: Path, payload: dict[str, Any]) -> None:
    """Atomically checkpoint complete structured state and its hash."""
    atomic_json(path, {**payload, "content_hash": canonical_hash(payload)})


def collect_static_context(  # noqa: C901, PLR0912, PLR0915 - explicit replay transaction
    factory: Callable[[], Any],
    output: Path,
    *,
    recipe: dict[str, Any],
    settings: InitialFeatureSettings = DEFAULT_FEATURE_SETTINGS,
    arm_order: Sequence[int] = (0, 1, 2, 3, 4),
) -> dict[str, Any]:
    """Replay a paired D0 and complete each static continuation to native T.

    All BO work uses the supplied CARP-S-backed DACBO factory. No independent
    BO loop is implemented. Initial objective calls are physically replayed
    per arm, explicitly accounted rather than silently claimed reused. An
    interrupted arm is replayed and its complete recorded prefix checked before
    continuation. Valid complete arms are reused only under the same recipe.
    """
    if sorted(arm_order) != list(range(5)):
        raise ValueError("Arm order must contain each WEI action exactly once.")
    if recipe.get("split") not in {"train", "dev"}:
        raise ValueError("Initial-base collection permits only explicit train/dev tasks.")
    protocol = {
        "version": "initial-static-execution-v1",
        "recipe": recipe,
        "features": asdict(settings),
        "target_semantics": "full_static_terminal_loss",
        "interaction_frequency": 5,
        "initialization_reuse": "deterministic_physical_replay_per_arm",
    }
    protocol_hash = canonical_hash(protocol)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "collection.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump({"protocol_hash": protocol_hash}, stream)
    try:
        anchor_path = output / "initialization.json"
        initialization = read_verified(anchor_path) if anchor_path.exists() else None
        if initialization is not None and initialization["protocol_hash"] != protocol_hash:
            raise ValueError("Initial-base recipe changed; use a new versioned output root.")
        for action in arm_order:
            path = output / f"arm_{action}.json"
            saved = read_verified(path) if path.exists() else None
            if saved is not None and (saved["protocol_hash"] != protocol_hash or saved["action"] != action):
                raise ValueError("Conflicting initial-base arm identity.")
            if saved is not None and saved["status"] == "complete":
                continue
            start = perf_counter()
            env = factory()
            try:
                if env.interaction_frequency != INTERACTION_FREQUENCY:
                    raise ValueError("Initial-base continuations require f=5.")
                env.reset()
                context = env.get_initial_context(asdict(settings))
                initial_records = [asdict(value) for value in completed_evaluations(env)]
                if initialization is None:
                    initialization = {
                        "protocol_hash": protocol_hash,
                        "protocol": protocol,
                        "context": asdict(context),
                        "context_digest": context.digest,
                        "initial_records": initial_records,
                        "feature_provenance": env._initial_feature_provenance,
                    }
                    write_verified(anchor_path, initialization)
                elif (
                    context.digest != initialization["context_digest"]
                    or initial_records != initialization["initial_records"]
                ):
                    raise ValueError("D0/z0/fitted model pairing mismatch across static arms.")
                old_records = [] if saved is None else saved["records"]
                rewards: list[float] = []
                while True:
                    records = [asdict(value) for value in completed_evaluations(env)]
                    prefix_length = min(len(records), len(old_records))
                    if records[:prefix_length] != old_records[:prefix_length]:
                        raise ValueError("Static-arm deterministic replay prefix mismatch.")
                    if len(records) > context.total_budget:
                        raise ValueError("Static continuation overran total logical BO budget.")
                    complete = len(records) == context.total_budget
                    if len(records) >= len(old_records):
                        history = env._smac_instance.runhistory
                        objective_times = [getattr(history[key], "time", None) for key in history]
                        write_verified(
                            path,
                            {
                                "protocol_hash": protocol_hash,
                                "action": action,
                                "context_digest": context.digest,
                                "design_digest": context.ordered_design_digest,
                                "status": "complete" if complete else "running",
                                "records": records,
                                "block_rewards": rewards,
                                "wall_seconds_this_attempt": perf_counter() - start,
                                "physical_evaluations_this_attempt": len(records),
                                "replayed_previous_prefix": len(old_records) if saved else 0,
                                "initial_timing": env._initial_feature_provenance.get("timing"),
                                "reported_objective_seconds": (
                                    float(np.sum([value for value in objective_times if value is not None]))
                                    if all(value is not None for value in objective_times)
                                    else None
                                ),
                            },
                        )
                    if complete:
                        break
                    _, reward, terminated, truncated, _ = env.step(action)
                    if not np.isfinite(reward):
                        raise ValueError("Nonfinite continuation reward.")
                    rewards.append(float(reward))
                    if (terminated or truncated) and env.get_n_finished_trials() != context.total_budget:
                        raise ValueError("Static continuation terminated before native budget T.")
            finally:
                env.close()
        assert initialization is not None
        anchor = InitialContext.from_dict(initialization["context"])
        arms = [read_verified(output / f"arm_{action}.json") for action in range(5)]
        if any(arm["status"] != "complete" for arm in arms):
            raise ValueError("Incomplete five-arm group cannot become a training row.")
        labels = PairedStaticLabels(
            anchor,
            tuple(record["cost"] for record in initialization["initial_records"]),
            tuple(tuple(np.minimum.accumulate([record["cost"] for record in arm["records"]])) for arm in arms),
            tuple(arm["context_digest"] for arm in arms),
            tuple(arm["design_digest"] for arm in arms),
            (anchor.total_budget,) * 5,
            0,
            int(recipe.get("continuation_replicate", 0)),
        )
        result = {
            "status": "complete",
            "protocol_hash": protocol_hash,
            "context": asdict(anchor),
            "context_digest": anchor.digest,
            "targets": labels.targets(),
            "recipe": recipe,
            "arm_hashes": [canonical_hash(arm) for arm in arms],
        }
        write_verified(output / "complete.json", result)
        return result
    finally:
        lock.unlink()
