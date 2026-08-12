"""Fit frozen offline flat/shared action scorers on consolidated branch data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn

from dacboenv.experiment.headroom_predictability import (
    DEFAULT_TIE_TOLERANCES,
    FlatObservationMLP,
    SharedActionScorer,
    parse_policy_observation,
)

MODEL_SEEDS = (1103, 2207, 3301)
EARLY_STOPPING_PATIENCE = 20
NEGATIVE_CONTROL_SEED = 904214


@dataclass(frozen=True)
class ObservationMatrices:
    """Aligned deployable observations, counterfactual labels, and split groups."""

    ids: tuple[str, ...]
    global_state: np.ndarray
    action_features: np.ndarray
    q_values: np.ndarray
    groups: tuple[str, ...]


def _matrices(snapshots: pd.DataFrame, branches: pd.DataFrame, family: str, horizon: int) -> ObservationMatrices:
    selected = snapshots[snapshots["action_space"] == family].sort_values("campaign_snapshot_id")
    global_rows, action_rows, q_rows, groups = [], [], [], []
    for row in selected.itertuples(index=False):
        global_state, action_features = parse_policy_observation(row.observation_json)
        outcomes = branches[
            (branches["campaign_snapshot_id"] == row.campaign_snapshot_id) & (branches["horizon"] == horizon)
        ].sort_values("action")
        if tuple(outcomes["action"]) != tuple(range(action_features.shape[0])):
            raise ValueError(f"Incomplete action labels for {row.campaign_snapshot_id}.")
        global_rows.append(global_state)
        action_rows.append(action_features)
        q_rows.append(outcomes["q_value"].to_numpy(dtype=np.float32))
        groups.append(f"{row.task_id}|{row.inner_seed}|{row.history_generator}|{row.history_seed}")
    return ObservationMatrices(
        ids=tuple(selected["campaign_snapshot_id"]),
        global_state=np.asarray(global_rows, dtype=np.float32),
        action_features=np.asarray(action_rows, dtype=np.float32),
        q_values=np.asarray(q_rows, dtype=np.float32),
        groups=tuple(groups),
    )


def _normalization(train: ObservationMatrices) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_mean = train.global_state.mean(axis=0)
    global_std = np.maximum(train.global_state.std(axis=0), 1e-6)
    action_mean = train.action_features.reshape(-1, train.action_features.shape[-1]).mean(axis=0)
    action_std = np.maximum(
        train.action_features.reshape(-1, train.action_features.shape[-1]).std(axis=0),
        1e-6,
    )
    return global_mean, global_std, action_mean, action_std


def _tensors(
    data: ObservationMatrices, normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_mean, global_std, action_mean, action_std = normalization
    advantages = data.q_values - data.q_values.mean(axis=1, keepdims=True)
    return (
        torch.tensor((data.global_state - global_mean) / global_std),
        torch.tensor((data.action_features - action_mean) / action_std),
        torch.tensor(advantages),
    )


def _development_mask(groups: Sequence[str], seed: int = 904212) -> np.ndarray:
    unique = sorted(set(groups))
    rng = np.random.default_rng(seed)
    development = set(np.asarray(unique)[rng.permutation(len(unique))[: max(1, len(unique) // 5)]])
    mask = np.asarray([group in development for group in groups])
    if mask.all() or not mask.any():
        raise ValueError("Grouped development split is degenerate.")
    return mask


def _negative_control(
    train: ObservationMatrices,
    validation: ObservationMatrices,
    kind: str,
) -> tuple[ObservationMatrices, ObservationMatrices]:
    """Apply one fixed negative control without using validation outcomes for fitting."""
    rng = np.random.default_rng(NEGATIVE_CONTROL_SEED)
    if kind == "shared_shuffled_labels":
        permutation = rng.permutation(len(train.q_values))
        return replace(train, q_values=train.q_values[permutation]), validation
    if kind == "shared_shuffled_global":
        train_permutation = rng.permutation(len(train.global_state))
        validation_permutation = rng.permutation(len(validation.global_state))
        return (
            replace(train, global_state=train.global_state[train_permutation]),
            replace(validation, global_state=validation.global_state[validation_permutation]),
        )
    if kind == "shared_row_mean":
        train_mean = np.repeat(train.action_features.mean(axis=1, keepdims=True), 5, axis=1)
        validation_mean = np.repeat(validation.action_features.mean(axis=1, keepdims=True), 5, axis=1)
        return replace(train, action_features=train_mean), replace(validation, action_features=validation_mean)
    if kind == "shared_mismatched_rows":

        def permute_rows(values: np.ndarray) -> np.ndarray:
            output = values.copy()
            for row in output:
                row[:] = row[rng.permutation(len(row))]
            return output

        return (
            replace(train, action_features=permute_rows(train.action_features)),
            replace(validation, action_features=permute_rows(validation.action_features)),
        )
    raise ValueError(f"Unknown negative control {kind!r}.")


def _fit_model(
    kind: str,
    train: ObservationMatrices,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    seed: int,
) -> nn.Module:
    torch.manual_seed(seed)
    model: nn.Module
    if kind == "flat_mlp":
        model = FlatObservationMLP(
            train.global_state.shape[1], train.action_features.shape[1], train.action_features.shape[2]
        )
    else:
        model = SharedActionScorer(train.global_state.shape[1], train.action_features.shape[2])
    global_state, action_features, targets = _tensors(train, normalization)
    development = torch.tensor(_development_mask(train.groups))
    fitting = ~development
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss = np.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    for _epoch in range(200):
        model.train()
        optimizer.zero_grad()
        prediction = model(global_state[fitting], action_features[fitting])
        loss = nn.functional.smooth_l1_loss(prediction, targets[fitting])
        if kind == "shared_scorer":
            target_differences = targets[fitting, :, None] - targets[fitting, None, :]
            prediction_differences = prediction[:, :, None] - prediction[:, None, :]
            signs = torch.sign(target_differences)
            ranked_pairs = torch.relu(0.001 - signs * prediction_differences)[signs != 0]
            if ranked_pairs.numel():
                loss = loss + 0.1 * ranked_pairs.mean()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_loss = float(
                nn.functional.smooth_l1_loss(
                    model(global_state[development], action_features[development]), targets[development]
                )
            )
        if dev_loss < best_loss - 1e-8:
            best_loss = dev_loss
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= EARLY_STOPPING_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model


def _predict(
    model: nn.Module,
    data: ObservationMatrices,
    normalization: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    global_state, action_features, _targets = _tensors(data, normalization)
    model.eval()
    with torch.no_grad():
        return model(global_state, action_features).numpy()


def _metrics(q_values: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    predicted = np.argmax(scores, axis=1)
    selected = q_values[np.arange(len(q_values)), predicted]
    best = q_values.max(axis=1)
    pairwise_correct, pairwise_total, correlations = 0, 0, []
    for q_row, score_row in zip(q_values, scores, strict=True):
        correlation = spearmanr(q_row, score_row).statistic
        if np.isfinite(correlation):
            correlations.append(float(correlation))
        for left in range(len(q_row)):
            for right in range(left + 1, len(q_row)):
                if q_row[left] == q_row[right]:
                    continue
                pairwise_total += 1
                pairwise_correct += int(
                    np.sign(q_row[left] - q_row[right]) == np.sign(score_row[left] - score_row[right])
                )
    output: dict[str, Any] = {
        "mean_selected_value": float(np.mean(selected)),
        "mean_oracle_value": float(np.mean(best)),
        "mean_selected_action_regret": float(np.mean(best - selected)),
        "pairwise_ranking_accuracy": float(pairwise_correct / pairwise_total),
        "mean_spearman": float(np.mean(correlations)),
    }
    for tolerance in DEFAULT_TIE_TOLERANCES:
        output[f"top_action_accuracy_at_{tolerance:g}"] = float(np.mean(best - selected <= tolerance))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """Train only on frozen training snapshots and evaluate validation once."""
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    train_snapshots = pd.read_parquet(args.artifact_directory / "headroom_train_snapshots.parquet")
    validation_snapshots = pd.read_parquet(args.artifact_directory / "headroom_validation_snapshots.parquet")
    branches = pd.read_parquet(args.artifact_directory / "branch_results.parquet")
    metrics, predictions = [], []
    for family in ("wei", "af_selection"):
        for horizon in (1, 5, 10):
            train = _matrices(train_snapshots, branches, family, horizon)
            validation = _matrices(validation_snapshots, branches, family, horizon)
            kinds = (
                "flat_mlp",
                "shared_scorer",
                "shared_shuffled_labels",
                "shared_shuffled_global",
                "shared_row_mean",
                "shared_mismatched_rows",
            )
            for kind in kinds:
                fit_train, evaluate_validation = (
                    (train, validation)
                    if kind in {"flat_mlp", "shared_scorer"}
                    else _negative_control(train, validation, kind)
                )
                seed_scores = []
                for seed in MODEL_SEEDS:
                    architecture = "flat_mlp" if kind == "flat_mlp" else "shared_scorer"
                    control_normalization = _normalization(fit_train)
                    model = _fit_model(architecture, fit_train, control_normalization, seed)
                    scores = _predict(model, evaluate_validation, control_normalization)
                    seed_scores.append(scores)
                    metric = _metrics(validation.q_values, scores)
                    metric.update({"action_family": family, "horizon": horizon, "model": kind, "seed": seed})
                    metrics.append(metric)
                ensemble = np.mean(seed_scores, axis=0)
                for snapshot_id, score_row, q_row in zip(validation.ids, ensemble, validation.q_values, strict=True):
                    predictions.append(
                        {
                            "campaign_snapshot_id": snapshot_id,
                            "action_family": family,
                            "horizon": horizon,
                            "model": kind,
                            "predicted_action": int(np.argmax(score_row)),
                            "scores_json": json.dumps(score_row.tolist()),
                            "q_values_json": json.dumps(q_row.tolist()),
                        }
                    )
    metric_frame = pd.DataFrame(metrics)
    prediction_frame = pd.DataFrame(predictions)
    metric_frame.to_csv(args.artifact_directory / "predictability_metrics.csv", index=False)
    (args.artifact_directory / "predictability_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_frame.to_parquet(args.artifact_directory / "predictor_predictions.parquet", index=False)
    print(metric_frame.groupby(["action_family", "horizon", "model"])["mean_selected_action_regret"].mean())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
