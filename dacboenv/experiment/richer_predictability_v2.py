"""Frozen richer offline action-value diagnostics for protocol v2."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

MODEL_SEED = 1103
THRESHOLDS = (0.0, 1e-4, 1e-3, 5e-3, 1e-2)
MIN_EXPERT_SNAPSHOTS = 20
GAP_HIGH = 1e-3
GAP_SCALE = 1e-2
GRU_PATIENCE = 15


def _development_ids(features: pd.DataFrame) -> set[str]:
    groups = features.task_id.astype(str) + "|" + features.inner_seed.astype(str) + "|" + features.history_generator
    unique = np.asarray(sorted(groups.unique()))
    selected = set(unique[np.random.default_rng(904212).permutation(len(unique))[: max(1, len(unique) // 5)]])
    return set(features.loc[groups.isin(selected), "campaign_snapshot_id"])


def _long_form(features: pd.DataFrame, branches: pd.DataFrame, family: str, horizon: int) -> pd.DataFrame:
    state = features[features.action_space == family].copy()
    outcome = branches[(branches.action_space == family) & (branches.horizon == horizon)][
        ["campaign_snapshot_id", "action", "q_value"]
    ]
    rows = state.merge(outcome, on="campaign_snapshot_id", validate="one_to_many")
    action_matrices = {
        row.campaign_snapshot_id: np.asarray(json.loads(row.action_features_json)) for row in state.itertuples()
    }
    for index in range(4):
        rows[f"action_feature_{index}"] = [
            action_matrices[snapshot][int(action), index]
            for snapshot, action in zip(rows.campaign_snapshot_id, rows.action, strict=True)
        ]
    rows["advantage"] = rows.q_value - rows.groupby("campaign_snapshot_id").q_value.transform("mean")
    rows["gap"] = rows.groupby("campaign_snapshot_id").q_value.transform(
        lambda values: np.sort(values)[-1] - np.sort(values)[-2]
    )
    return rows


def _columns(rows: pd.DataFrame, feature_set: str) -> tuple[list[str], list[str]]:
    action = ["action", *[f"action_feature_{index}" for index in range(4)]]
    static_numeric = [
        "dimension",
        "effective_dimension",
        "native_budget",
        "initial_design_size",
        "initial_design_fraction",
        "remaining_budget",
        "budget_per_effective_dimension",
    ]
    search = [
        column
        for column in rows
        if column.startswith(("n_", "fraction_", "mean_active", "variance_active", "log_", "conditional_"))
    ]
    history = [
        column
        for column in rows
        if column.startswith(
            (
                "improvement_",
                "n_improvements_",
                "time_since_",
                "previous_action",
                "action_age",
                "mean_recent",
                "recent_reward",
            )
        )
    ]
    global_columns = [f"global_{index}" for index in range(13)]
    missing_global = [column for column in global_columns if column not in rows]
    if missing_global:
        global_values = np.asarray(rows.global_state_json.map(json.loads).tolist())
        for index, column in enumerate(global_columns):
            rows[column] = global_values[:, index]
    numeric = action + global_columns
    categorical: list[str] = []
    if feature_set in {"static", "rich", "privileged"}:
        numeric += static_numeric
        categorical += ["domain", "scenario", "surrogate_family"]
    if feature_set in {"rich", "privileged"}:
        numeric += search + history
    if feature_set == "privileged":
        numeric += ["function_group_privileged"]
    return sorted(set(numeric)), categorical


def _pipeline(model: Any, numeric: list[str], categorical: list[str]) -> Pipeline:
    transformer = ColumnTransformer(
        [
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ]
    )
    return Pipeline([("features", transformer), ("model", model)])


def _fit_regressor(
    kind: str,
    fit: pd.DataFrame,
    dev: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    weight_mode: str = "uniform",
) -> Pipeline:
    candidates: list[Any]
    if kind == "extra_trees":
        candidates = [
            ExtraTreesRegressor(
                n_estimators=200, min_samples_leaf=leaf, max_features=features, random_state=MODEL_SEED, n_jobs=1
            )
            for leaf in (2, 5)
            for features in (0.7, 1.0)
        ]
    else:
        candidates = [
            HistGradientBoostingRegressor(
                max_iter=150, learning_rate=rate, max_leaf_nodes=leaves, l2_regularization=1.0, random_state=MODEL_SEED
            )
            for rate in (0.03, 0.08)
            for leaves in (7, 15)
        ]
    weights = None
    if weight_mode == "gap_weighted":
        weights = np.minimum(fit.gap.to_numpy() / GAP_SCALE, 1.0)
    elif weight_mode == "high_gap_curriculum":
        weights = 1.0 + (fit.gap.to_numpy() > GAP_HIGH).astype(float)
    best: tuple[float, Pipeline] | None = None
    predictors = numeric + categorical
    for candidate in candidates:
        model = _pipeline(candidate, numeric, categorical)
        kwargs = {} if weights is None else {"model__sample_weight": weights}
        model.fit(fit[predictors], fit.advantage, **kwargs)
        score = mean_absolute_error(dev.advantage, model.predict(dev[predictors]))
        if best is None or score < best[0]:
            best = score, model
    assert best is not None
    chosen = best[1]
    full = pd.concat([fit, dev], ignore_index=True)
    full_weights = None
    if weight_mode == "gap_weighted":
        full_weights = np.minimum(full.gap.to_numpy() / GAP_SCALE, 1.0)
    elif weight_mode == "high_gap_curriculum":
        full_weights = 1.0 + (full.gap.to_numpy() > GAP_HIGH).astype(float)
    kwargs = {} if full_weights is None else {"model__sample_weight": full_weights}
    chosen.fit(full[predictors], full.advantage, **kwargs)
    return chosen


def _pairwise_model(fit: pd.DataFrame, dev: pd.DataFrame, numeric: list[str]) -> tuple[Pipeline, list[str]]:
    def pairs(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        records, labels = [], []
        for _snapshot, group in rows.groupby("campaign_snapshot_id"):
            ordered = group.sort_values("action")
            for left in range(5):
                for right in range(left + 1, 5):
                    if ordered.q_value.iloc[left] == ordered.q_value.iloc[right]:
                        continue
                    records.append(
                        {column: ordered[column].iloc[left] - ordered[column].iloc[right] for column in numeric}
                    )
                    labels.append(int(ordered.q_value.iloc[left] > ordered.q_value.iloc[right]))
        return pd.DataFrame(records), np.asarray(labels)

    fit_pairs, fit_labels = pairs(fit)
    dev_pairs, dev_labels = pairs(dev)
    best: tuple[float, Pipeline] | None = None
    for regularization in (0.1, 1.0, 10.0):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=regularization, max_iter=1000, random_state=MODEL_SEED)),
            ]
        )
        model.fit(fit_pairs[numeric], fit_labels)
        score = float(np.mean(model.predict(dev_pairs[numeric]) == dev_labels))
        if best is None or score > best[0]:
            best = score, model
    assert best is not None
    all_pairs, all_labels = pairs(pd.concat([fit, dev]))
    best[1].fit(all_pairs[numeric], all_labels)
    return best[1], numeric


def _pairwise_scores(model: Pipeline, rows: pd.DataFrame, numeric: list[str]) -> np.ndarray:
    scores = np.zeros((rows.campaign_snapshot_id.nunique(), 5))
    for state_index, (_snapshot, group) in enumerate(rows.groupby("campaign_snapshot_id", sort=True)):
        ordered = group.sort_values("action")
        for left in range(5):
            for right in range(left + 1, 5):
                difference = pd.DataFrame(
                    [{column: ordered[column].iloc[left] - ordered[column].iloc[right] for column in numeric}]
                )
                probability = model.predict_proba(difference[numeric])[0, 1]
                scores[state_index, left] += probability
                scores[state_index, right] += 1 - probability
    return scores


class ShortHistoryGRU(nn.Module):
    """Small masked sequence encoder with a shared action scorer."""

    def __init__(self, sequence_width: int, action_width: int) -> None:
        super().__init__()
        self.gru = nn.GRU(sequence_width, 32, batch_first=True)
        self.action = nn.Linear(action_width, 16)
        self.score = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        """Score action rows from the final available masked recurrent state."""
        output, _hidden = self.gru(sequence)
        positions = torch.arange(mask.shape[1], device=mask.device)[None, :]
        last_available = (positions * mask).max(dim=1)[0].long()
        state = output[torch.arange(len(output)), last_available]
        action = torch.relu(self.action(action_features))
        return self.score(torch.cat((state[:, None, :].expand(-1, 5, -1), action), dim=-1)).squeeze(-1)


def _gru_scores(
    train: pd.DataFrame, validation: pd.DataFrame, development_ids: set[str], *, model_seed: int = MODEL_SEED
) -> np.ndarray:
    def tensors(rows: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = rows.drop_duplicates("campaign_snapshot_id").sort_values("campaign_snapshot_id")
        sequence = torch.tensor(np.asarray(states.sequence_json.map(json.loads).tolist()), dtype=torch.float32)
        mask = torch.tensor(np.asarray(states.sequence_mask_json.map(json.loads).tolist()), dtype=torch.float32)
        matrices = np.asarray(states.action_features_json.map(json.loads).tolist(), dtype=np.float32)
        q = (
            rows.pivot_table(index="campaign_snapshot_id", columns="action", values="q_value", aggfunc="first")
            .loc[states.campaign_snapshot_id]
            .to_numpy(dtype=np.float32)
        )
        return sequence, mask, torch.tensor(matrices), torch.tensor(q - q.mean(axis=1, keepdims=True))

    torch.manual_seed(model_seed)
    torch.set_num_threads(1)
    sequence, mask, actions, targets = tensors(train)
    states = train.drop_duplicates("campaign_snapshot_id").sort_values("campaign_snapshot_id")
    development = torch.tensor(states.campaign_snapshot_id.isin(development_ids).to_numpy())
    model = ShortHistoryGRU(sequence.shape[-1], actions.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    best_loss, best_state, patience = np.inf, None, 0
    for _epoch in range(100):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.smooth_l1_loss(
            model(sequence[~development], mask[~development], actions[~development]), targets[~development]
        )
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            dev_loss = float(
                nn.functional.smooth_l1_loss(
                    model(sequence[development], mask[development], actions[development]), targets[development]
                )
            )
        if dev_loss < best_loss - 1e-8:
            best_loss, best_state, patience = (
                dev_loss,
                {key: value.clone() for key, value in model.state_dict().items()},
                0,
            )
        else:
            patience += 1
            if patience == GRU_PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    val_sequence, val_mask, val_actions, _targets = tensors(validation)
    with torch.no_grad():
        return model(val_sequence, val_mask, val_actions).numpy()


def _summaries(
    validation: pd.DataFrame, scores: np.ndarray, model: str, feature_set: str, family: str, horizon: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrices = validation.pivot_table(
        index="campaign_snapshot_id", columns="action", values="q_value", aggfunc="first"
    ).sort_index()
    predicted = np.argmax(scores, axis=1)
    records = []
    for index, snapshot in enumerate(matrices.index):
        q = matrices.loc[snapshot].to_numpy()
        records.append(
            {
                "campaign_snapshot_id": snapshot,
                "action_family": family,
                "horizon": horizon,
                "model": model,
                "feature_set": feature_set,
                "predicted_action": int(predicted[index]),
                "selected_value": float(q[predicted[index]]),
                "oracle_value": float(q.max()),
                "selected_action_regret": float(q.max() - q[predicted[index]]),
                "gap": float(np.sort(q)[-1] - np.sort(q)[-2]),
            }
        )
    frame = pd.DataFrame(records)
    metrics = []
    for subset, selected in (
        ("all", frame),
        ("gap_gt_1e-3", frame[frame.gap > GAP_HIGH]),
        ("gap_gt_1e-2", frame[frame.gap > GAP_SCALE]),
    ):
        metrics.append(
            {
                "action_family": family,
                "horizon": horizon,
                "model": model,
                "feature_set": feature_set,
                "subset": subset,
                "n": len(selected),
                "selected_value": selected.selected_value.mean(),
                "selected_action_regret": selected.selected_action_regret.mean(),
                "top_action_accuracy": np.mean(selected.selected_action_regret <= GAP_HIGH),
            }
        )
    return records, metrics


def _context_static_actions(training: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Fit deployable scenario/dimension static actions on training only."""
    train_states = training.copy()
    target_states = target.drop_duplicates("campaign_snapshot_id").sort_values("campaign_snapshot_id")
    train_states["context_key"] = np.where(
        train_states.domain == "yahpo",
        "yahpo:" + train_states.scenario.astype(str),
        "bbob:d" + train_states.dimension.astype(str),
    )
    target_states["context_key"] = np.where(
        target_states.domain == "yahpo",
        "yahpo:" + target_states.scenario.astype(str),
        "bbob:d" + target_states.dimension.astype(str),
    )
    global_action = int(training.groupby("action").q_value.mean().idxmax())
    mapping = {
        key: int(group.groupby("action").q_value.mean().idxmax())
        for key, group in train_states.groupby("context_key", sort=True)
    }
    return np.asarray([mapping.get(key, global_action) for key in target_states.context_key], dtype=int)


def main(argv: Sequence[str] | None = None) -> int:  # noqa: PLR0915
    """Fit all frozen richer diagnostics and evaluate validation once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/headroom_predictability_v2"))
    parser.add_argument("--branches", type=Path, default=Path("artifacts/branch_results.parquet"))
    parser.add_argument("--gru-seed", action="append", type=int, default=[])
    args = parser.parse_args(argv)
    features = pd.read_parquet(args.input / "deployable_features.parquet")
    branches = pd.read_parquet(args.branches)
    predictions, metrics, ablations, importance_rows, sparse_rows = [], [], [], [], []
    for family in ("wei", "af_selection"):
        for horizon in (1, 5, 10):
            long = _long_form(features, branches, family, horizon)
            # Materialize compact observation columns once before any split so
            # fit, development, and frozen validation use the identical schema.
            _columns(long, "compact")
            long = long.sort_values(["campaign_snapshot_id", "action"]).reset_index(drop=True)
            train, validation = long[long.split == "train"].copy(), long[long.split == "validation"].copy()
            development_ids = _development_ids(
                features[(features.split == "train") & (features.action_space == family)]
            )
            fit, dev = (
                train[~train.campaign_snapshot_id.isin(development_ids)],
                train[train.campaign_snapshot_id.isin(development_ids)],
            )
            for feature_set in ("compact", "static", "rich", "privileged"):
                numeric, categorical = _columns(train, feature_set)
                predictors = numeric + categorical
                for kind in ("extra_trees", "hist_gradient_boosting"):
                    for weight_mode in (
                        ("uniform", "gap_weighted", "high_gap_curriculum")
                        if feature_set == "rich" and kind == "extra_trees"
                        else ("uniform",)
                    ):
                        model = _fit_regressor(kind, fit, dev, numeric, categorical, weight_mode)
                        scores = model.predict(validation[predictors]).reshape(-1, 5)
                        name = kind if weight_mode == "uniform" else f"{kind}_{weight_mode}"
                        records, summary = _summaries(validation, scores, name, feature_set, family, horizon)
                        predictions.extend(records)
                        metrics.extend(summary)
                        ablations.extend(summary)
                        if kind == "extra_trees" and feature_set == "rich" and weight_mode == "uniform":
                            result = permutation_importance(
                                model,
                                dev[predictors],
                                dev.advantage,
                                n_repeats=5,
                                random_state=MODEL_SEED,
                                scoring="neg_mean_absolute_error",
                            )
                            for column, value in zip(predictors, result.importances_mean, strict=True):
                                importance_rows.append(
                                    {
                                        "action_family": family,
                                        "horizon": horizon,
                                        "feature": column,
                                        "importance": value,
                                    }
                                )
                if feature_set == "compact":
                    pair_model, pair_numeric = _pairwise_model(fit, dev, numeric)
                    scores = _pairwise_scores(pair_model, validation, pair_numeric)
                    records, summary = _summaries(validation, scores, "pairwise_logistic", feature_set, family, horizon)
                    predictions.extend(records)
                    metrics.extend(summary)
            for gru_seed in args.gru_seed or [MODEL_SEED]:
                gru_scores = _gru_scores(train, validation, development_ids, model_seed=gru_seed)
                records, summary = _summaries(
                    validation, gru_scores, f"short_history_gru_seed{gru_seed}", "sequence", family, horizon
                )
                predictions.extend(records)
                metrics.extend(summary)

            # Sparse residual threshold uses development only and rich Extra Trees scores.
            numeric, categorical = _columns(train, "rich")
            model = _fit_regressor("extra_trees", fit, dev, numeric, categorical)
            predictors = numeric + categorical
            dev_scores = model.predict(dev[predictors]).reshape(-1, 5)
            dev_matrix = dev.pivot_table(
                index="campaign_snapshot_id", columns="action", values="q_value", aggfunc="first"
            ).sort_index()
            dev_base_actions = _context_static_actions(fit, dev)
            best_threshold = max(
                THRESHOLDS,
                key=lambda threshold: np.mean(
                    [
                        dev_matrix.iloc[index, np.argmax(score)]
                        if score.max() - score[dev_base_actions[index]] > threshold
                        else dev_matrix.iloc[index, dev_base_actions[index]]
                        for index, score in enumerate(dev_scores)
                    ]
                ),
            )
            val_scores = model.predict(validation[predictors]).reshape(-1, 5)
            val_matrix = validation.pivot_table(
                index="campaign_snapshot_id", columns="action", values="q_value", aggfunc="first"
            ).sort_index()
            val_base_actions = _context_static_actions(train, validation)
            harmful, overrides, beneficial = 0, 0, 0
            true_positive_states, captured_true_positive = 0, 0
            selected_values = []
            for index, score in enumerate(val_scores):
                base_action = int(val_base_actions[index])
                action = int(np.argmax(score)) if score.max() - score[base_action] > best_threshold else base_action
                override = action != base_action
                advantage = float(val_matrix.iloc[index, action] - val_matrix.iloc[index, base_action])
                true_override_advantage = float(val_matrix.iloc[index].max() - val_matrix.iloc[index, base_action])
                true_positive_states += true_override_advantage > 0
                captured_true_positive += override and advantage > 0
                overrides += override
                harmful += override and advantage < 0
                beneficial += override and advantage > 0
                selected_values.append(float(val_matrix.iloc[index, action]))
            sparse_rows.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "threshold": best_threshold,
                    "base_selector": "context_static_deployable",
                    "selected_value": np.mean(selected_values),
                    "override_rate": overrides / len(val_scores),
                    "override_precision": beneficial / max(overrides, 1),
                    "override_recall": captured_true_positive / max(true_positive_states, 1),
                    "harmful_override_fraction": harmful / max(overrides, 1),
                }
            )
    pd.DataFrame(metrics).to_csv(args.input / "predictability_metrics.csv", index=False)
    pd.DataFrame(predictions).to_parquet(args.input / "predictability_predictions.parquet", index=False)
    pd.DataFrame(sparse_rows).to_csv(args.input / "sparse_residual_metrics.csv", index=False)
    pd.DataFrame(ablations).to_csv(args.input / "feature_ablation.csv", index=False)
    pd.DataFrame(importance_rows).to_csv(args.input / "feature_importance.csv", index=False)
    print(
        pd.DataFrame(metrics)
        .query("subset == 'all'")
        .sort_values(["action_family", "horizon", "selected_action_regret"])
        .groupby(["action_family", "horizon"])
        .head(3)
        .to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
