"""Leakage-safe deployable feature reconstruction for headroom protocol v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dacboenv.experiment.headroom_predictability import parse_policy_observation

SENTINEL = -1.0
SEQUENCE_LENGTH = 10
SIGNIFICANT_IMPROVEMENT = 1e-12


def _configuration_descriptors(evaluations: list[dict[str, Any]]) -> dict[str, float]:
    configurations = [json.loads(row["configuration_json"]) for row in evaluations]
    keys = sorted({key for configuration in configurations for key in configuration})
    values_by_key = {key: [configuration.get(key) for configuration in configurations] for key in keys}
    categorical, integer, numerical = 0, 0, 0
    cardinality_log = 0.0
    active_counts = []
    for configuration in configurations:
        active_counts.append(sum(value is not None for value in configuration.values()))
    for values in values_by_key.values():
        active = [value for value in values if value is not None]
        if not active:
            continue
        if any(isinstance(value, (str, bool)) for value in active):
            categorical += 1
            cardinality_log += math.log(max(1, len(set(map(str, active)))))
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in active):
            integer += 1
        else:
            numerical += 1
    total = max(len(keys), 1)
    conditional = sum(
        any(value is None for value in values) and any(value is not None for value in values)
        for values in values_by_key.values()
    )
    return {
        "n_float": float(numerical),
        "n_integer": float(integer),
        "n_ordinal": 0.0,
        "n_categorical": float(categorical),
        "fraction_float": numerical / total,
        "fraction_integer": integer / total,
        "fraction_ordinal": 0.0,
        "fraction_categorical": categorical / total,
        "n_conditional": float(conditional),
        "n_conditions": float(conditional),
        "conditional_graph_depth": float(conditional > 0),
        "mean_active_variables": float(np.mean(active_counts)),
        "variance_active_variables": float(np.var(active_counts)),
        "log_categorical_cardinality": cardinality_log,
        "log_approx_cardinality": SENTINEL,
        "n_log_scaled": SENTINEL,
        "fraction_log_scaled": SENTINEL,
        "n_bounded_numeric": float(numerical + integer),
        "fraction_bounded_numeric": (numerical + integer) / total,
        "configspace_exact_schema_available": 0.0,
    }


def _history_features(snapshot: Mapping[str, Any]) -> dict[str, float]:
    evaluations = json.loads(snapshot["completed_evaluations_json"])
    costs = np.asarray([float(row["cost"]) for row in evaluations], dtype=float)
    incumbents = np.minimum.accumulate(costs)
    raw_actions = snapshot["action_history"]
    actions = json.loads(raw_actions) if isinstance(raw_actions, str) else np.asarray(raw_actions).tolist()
    output: dict[str, float] = {}
    for window in (1, 5, 10):
        available = len(incumbents) > window
        output[f"improvement_{window}"] = float(incumbents[-window - 1] - incumbents[-1]) if available else SENTINEL
        output[f"improvement_{window}_available"] = float(available)
    for window in (5, 10):
        available = len(incumbents) > window
        output[f"n_improvements_{window}"] = (
            float(np.sum(np.diff(incumbents[-window - 1 :]) < 0)) if available else SENTINEL
        )
        output[f"n_improvements_{window}_available"] = float(available)
    improvements = np.flatnonzero(np.diff(incumbents) < -SIGNIFICANT_IMPROVEMENT)
    output["time_since_significant_improvement"] = (
        float(len(incumbents) - 1 - improvements[-1]) if len(improvements) else float(len(incumbents))
    )
    output["previous_action"] = float(actions[-1]) if actions else SENTINEL
    output["previous_action_available"] = float(bool(actions))
    output["action_age"] = (
        float(next((index for index, value in enumerate(reversed(actions)) if value != actions[-1]), len(actions)))
        if actions
        else SENTINEL
    )
    recent_rewards = -np.diff(incumbents[-11:]) if len(incumbents) > 1 else np.asarray([])
    output["mean_recent_reward"] = float(np.mean(recent_rewards)) if len(recent_rewards) else SENTINEL
    output["recent_reward_variance"] = float(np.var(recent_rewards)) if len(recent_rewards) else SENTINEL
    output["recent_reward_available"] = float(bool(len(recent_rewards)))
    for unavailable in (
        "previous_action_block_potential_improvement",
        "prequential_calibration_error_mean",
        "prequential_calibration_error_trend",
        "surrogate_prediction_residual_trend",
    ):
        output[unavailable] = SENTINEL
        output[f"{unavailable}_available"] = 0.0
    return output


def _sequence(snapshot: Mapping[str, Any], global_state: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """Build a left-padded deployable sequence; earlier compact observations were not persisted."""
    raw_actions = snapshot["action_history"]
    actions = json.loads(raw_actions) if isinstance(raw_actions, str) else np.asarray(raw_actions).tolist()
    width = len(global_state) + 1
    sequence = np.full((SEQUENCE_LENGTH, width), SENTINEL, dtype=np.float32)
    mask = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
    sequence[-1, :-1] = global_state
    sequence[-1, -1] = float(actions[-1]) if actions else SENTINEL
    mask[-1] = 1.0
    return sequence.tolist(), mask.tolist()


def build_feature_table(snapshots: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exact compact plus reconstructable static/search/history features."""
    rows = []
    for snapshot in snapshots.to_dict("records"):
        global_state, action_features = parse_policy_observation(snapshot["observation_json"])
        evaluations = json.loads(snapshot["completed_evaluations_json"])
        descriptor = _configuration_descriptors(evaluations)
        history = _history_features(snapshot)
        total_budget = int(snapshot["total_budget"])
        dimension = (
            float(snapshot["dimension"]) if pd.notna(snapshot["dimension"]) else descriptor["mean_active_variables"]
        )
        initial_design_size = sum(
            index < len(evaluations) and index < round(total_budget * 0.2) for index in range(len(evaluations))
        )
        sequence, sequence_mask = _sequence(snapshot, global_state)
        feature = {
            "campaign_snapshot_id": snapshot["campaign_snapshot_id"],
            "split": snapshot["split"],
            "domain": "yahpo" if str(snapshot["task_id"]).startswith("yahpo/") else "bbob",
            "scenario": snapshot["scenario"],
            "task_id": snapshot["task_id"],
            "inner_seed": int(snapshot["inner_seed"]),
            "history_generator": snapshot["history_generator"],
            "action_space": snapshot["action_space"],
            "budget_fraction": float(snapshot["budget_fraction"]),
            "dimension": dimension,
            "effective_dimension": descriptor["mean_active_variables"],
            "native_budget": total_budget,
            "initial_design_size": initial_design_size,
            "initial_design_fraction": initial_design_size / total_budget,
            "surrogate_family": "gaussian_process",
            "remaining_budget": total_budget - len(evaluations),
            "budget_per_effective_dimension": total_budget / max(descriptor["mean_active_variables"], 1.0),
            "global_state_json": json.dumps(global_state.tolist(), separators=(",", ":")),
            "action_features_json": json.dumps(action_features.tolist(), separators=(",", ":")),
            "sequence_json": json.dumps(sequence, separators=(",", ":")),
            "sequence_mask_json": json.dumps(sequence_mask, separators=(",", ":")),
            **descriptor,
            **history,
        }
        if feature["domain"] == "bbob":
            function = int(str(snapshot["task_id"]).split("/")[2])
            feature["function_group_privileged"] = next(
                index for index, upper in enumerate((5, 9, 14, 19, 24)) if function <= upper
            )
        else:
            feature["function_group_privileged"] = SENTINEL
        rows.append(feature)
    frame = pd.DataFrame(rows)
    schema = {
        "version": "headroom-predictability-v2-features",
        "sentinel": SENTINEL,
        "sequence_length": SEQUENCE_LENGTH,
        "columns": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "warning": (
            "Only the current compact observation was persisted; earlier sequence observations "
            "are sentinel-padded with mask zero."
        ),
    }
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    schema["sha256"] = hashlib.sha256(payload).hexdigest()
    return frame, schema


def main(argv: Sequence[str] | None = None) -> int:
    """Create v2 feature tables without replay or future labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/headroom_predictability_v2"))
    args = parser.parse_args(argv)
    snapshots = pd.concat(
        [
            pd.read_parquet(args.artifact_directory / "headroom_train_snapshots.parquet"),
            pd.read_parquet(args.artifact_directory / "headroom_validation_snapshots.parquet"),
        ],
        ignore_index=True,
    )
    features, schema = build_feature_table(snapshots)
    args.output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output / "deployable_features.parquet", index=False)
    (args.output / "feature_schema.json").write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": len(features), "schema_hash": schema["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
