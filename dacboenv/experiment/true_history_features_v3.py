"""Merge fingerprint-verified true histories into corrected deployable features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dacboenv.experiment.headroom_predictability import parse_policy_observation
from dacboenv.experiment.task_metadata import parse_task_metadata

HISTORY_LENGTH = 10
SENTINEL = -1.0


def _vectorize(row: pd.Series) -> tuple[str, str]:
    observation_rows = json.loads(row.observation_sequence_json)
    actions = json.loads(row.action_sequence_json)
    rewards = json.loads(row.reward_sequence_json)
    fractions = json.loads(row.budget_fraction_sequence_json)
    mask = json.loads(row.mask_json)
    available_observations = [
        observation for observation, available in zip(observation_rows, mask, strict=True) if available
    ]
    if not available_observations:
        raise ValueError("A reconstructed history must contain its current deployable observation.")
    sample_global, sample_actions = parse_policy_observation(available_observations[0])
    sequence_width = int(sample_global.size + sample_actions.size + 5 + 2)
    sequence = []
    for observation, action, reward, fraction, available in zip(
        observation_rows, actions, rewards, fractions, mask, strict=True
    ):
        if not available:
            sequence.append([SENTINEL] * sequence_width)
            continue
        global_state, action_features = parse_policy_observation(observation)
        action_one_hot = np.zeros(5, dtype=float)
        action_one_hot[int(action)] = 1.0
        vector = np.concatenate((global_state, action_features.reshape(-1), action_one_hot, [reward, fraction]))
        if vector.size != sequence_width:
            raise ValueError("Deployable observation width changed within a reconstructed trajectory.")
        sequence.append(vector.tolist())
    if len(sequence) != HISTORY_LENGTH:
        raise ValueError("True-history sequence length changed from the frozen protocol.")
    return json.dumps(sequence, separators=(",", ":")), json.dumps(mask, separators=(",", ":"))


def main() -> int:
    """Create the v3 feature table only after both replay panels are complete."""
    root = Path("artifacts/headroom_predictability_v3")
    history = pd.concat(
        [
            pd.read_parquet(root / "history_sequences_train.parquet"),
            pd.read_parquet(root / "history_sequences_validation.parquet"),
        ]
    )
    features = pd.read_parquet("artifacts/headroom_predictability_v2/deployable_features.parquet")
    metadata = features.task_id.map(parse_task_metadata)
    features["domain"] = metadata.map(lambda value: value.domain)
    features["scenario"] = metadata.map(lambda value: value.scenario)
    features["dataset_instance"] = metadata.map(lambda value: value.dataset_instance)
    if features[features.domain == "yahpo"].scenario.isna().any():
        raise ValueError("YAHPO scenario propagation failed before model fitting.")
    history[["true_sequence_json", "true_sequence_mask_json"]] = history.apply(
        lambda row: pd.Series(_vectorize(row)), axis=1
    )
    merged = features.merge(
        history[["campaign_snapshot_id", "true_sequence_json", "true_sequence_mask_json", "sequence_length"]],
        on="campaign_snapshot_id",
        validate="one_to_one",
    )
    if len(merged) != len(features):
        raise RuntimeError("True histories do not cover every frozen snapshot.")
    merged["sequence_json"] = merged.true_sequence_json
    merged["sequence_mask_json"] = merged.true_sequence_mask_json
    merged.to_parquet(root / "deployable_features.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
