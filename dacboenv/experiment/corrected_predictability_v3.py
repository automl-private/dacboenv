"""Recompute predictor metrics with canonical metadata and local denominators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dacboenv.experiment.scenario_v3 import RESAMPLES, _subgroups, captured_ratio

TIE_TOLERANCE = 1e-3


def _predictions() -> pd.DataFrame:
    true_history_path = Path("artifacts/headroom_predictability_v3/predictability_predictions.parquet")
    v2 = pd.read_parquet(
        true_history_path
        if true_history_path.is_file()
        else "artifacts/headroom_predictability_v2/predictability_predictions.parquet"
    )
    legacy = pd.read_parquet("artifacts/predictor_predictions.parquet")
    legacy["feature_set"] = "compact"
    legacy["selected_value"] = [
        json.loads(values)[int(action)]
        for values, action in zip(legacy.q_values_json, legacy.predicted_action, strict=True)
    ]
    legacy["oracle_value_prediction_record"] = legacy.q_values_json.map(lambda value: max(json.loads(value)))
    columns = [
        "campaign_snapshot_id",
        "action_family",
        "horizon",
        "model",
        "feature_set",
        "predicted_action",
        "selected_value",
    ]
    return pd.concat([v2[columns], legacy[columns]], ignore_index=True).drop_duplicates(
        ["campaign_snapshot_id", "action_family", "horizon", "model", "feature_set"]
    )


def _paired_bootstrap(group: pd.DataFrame, rng: np.random.Generator) -> tuple[float, float, float]:
    domain_groups = {}
    for domain, domain_rows in group.groupby("domain"):
        stratum = "dimension" if domain == "bbob" else "scenario"
        domain_groups[domain] = {
            key: tuple(values.value_over_nonfeedback.to_numpy() for _trajectory, values in rows.groupby("trajectory"))
            for key, rows in domain_rows.groupby(stratum)
        }
    samples = []
    for _ in range(RESAMPLES):
        domains = []
        for strata in domain_groups.values():
            stratum_values = []
            for trajectories in strata.values():
                selected = rng.integers(len(trajectories), size=len(trajectories))
                values = []
                for index in selected:
                    trajectory = trajectories[int(index)]
                    values.extend(trajectory[rng.integers(len(trajectory), size=len(trajectory))])
                stratum_values.append(np.mean(values))
            domains.append(np.mean(stratum_values))
        samples.append(np.mean(domains))
    values = np.asarray(samples)
    return float(np.median(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> int:
    """Write corrected v3 prediction tables from all currently frozen models."""
    output = Path("artifacts/headroom_predictability_v3")
    states = pd.read_parquet(output / "corrected_state_values.parquet")
    predictions = _predictions().merge(
        states[
            [
                "campaign_snapshot_id",
                "action_space",
                "horizon",
                "domain",
                "scenario",
                "dataset_instance",
                "dimension",
                "function_id",
                "target_budget_fraction",
                "history_generator",
                "trajectory",
                "gap",
                "selected_value",
                "oracle_value",
            ]
        ].rename(columns={"selected_value": "nonfeedback_value"}),
        left_on=["campaign_snapshot_id", "action_family", "horizon"],
        right_on=["campaign_snapshot_id", "action_space", "horizon"],
        validate="many_to_one",
    )
    predictions["selected_action_regret"] = predictions.oracle_value - predictions.selected_value
    predictions["value_over_nonfeedback"] = predictions.selected_value - predictions.nonfeedback_value
    predictions.to_parquet(output / "predictability_predictions_v3.parquet", index=False)

    metrics = []
    for keys, model_rows in predictions.groupby(["action_family", "horizon", "model", "feature_set"]):
        family, horizon, model, feature_set = keys
        for subgroup, state_group in _subgroups(
            states[(states.action_space == family) & (states.horizon == horizon)]
        ).items():
            group = model_rows[model_rows.campaign_snapshot_id.isin(state_group.campaign_snapshot_id)]
            if not len(group):
                continue
            model_value = group.selected_value.mean()
            nonfeedback = group.nonfeedback_value.mean()
            oracle = group.oracle_value.mean()
            metrics.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "model": model,
                    "feature_set": feature_set,
                    "subgroup": subgroup,
                    "n": len(group),
                    "selected_value": model_value,
                    "nonfeedback_value": nonfeedback,
                    "oracle_value": oracle,
                    "residual_headroom": oracle - nonfeedback,
                    "selected_action_regret": group.selected_action_regret.mean(),
                    "value_over_nonfeedback": model_value - nonfeedback,
                    "captured_residual_headroom": captured_ratio(model_value, nonfeedback, oracle),
                    "tie_aware_top_action_accuracy_1e-3": np.mean(group.selected_action_regret <= TIE_TOLERANCE),
                }
            )
    pd.DataFrame(metrics).to_csv(output / "predictability_metrics_v3.csv", index=False)

    rng = np.random.default_rng(881204)
    bootstraps = []
    for keys, group in predictions.groupby(["action_family", "horizon", "model", "feature_set"]):
        median, low, high = _paired_bootstrap(group, rng)
        bootstraps.append(
            dict(
                zip(("action_family", "horizon", "model", "feature_set"), keys, strict=True),
                median=median,
                ci_low=low,
                ci_high=high,
                resamples=RESAMPLES,
            )
        )
    pd.DataFrame(bootstraps).to_csv(output / "model_improvement_bootstrap.csv", index=False)
    # Explicit interim controls: corrected scenario shuffle; temporal controls await real histories.
    pd.DataFrame(
        [
            {"control": "scenario_labels_shuffled_within_yahpo", "status": "prepared_training_only"},
            {"control": "history_shuffled_across_trajectories", "status": "awaiting_true_histories"},
            {"control": "history_time_reversed", "status": "awaiting_true_histories"},
            {"control": "history_values_zeroed_masks_retained", "status": "awaiting_true_histories"},
        ]
    ).to_csv(output / "negative_controls_v3.csv", index=False)
    pd.DataFrame(columns=["model", "feature_group", "selected_value"]).to_csv(
        output / "feature_ablation_v3.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
