"""Render protocol-v2 reports, hashes, and noninteractive figures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path("artifacts/headroom_predictability_v2")
FIGURES = Path("figures")
EPSILON = 1e-12
GAP_SMALL = 1e-3
GAP_MEDIUM = 1e-2
LONG_HORIZON = 10


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save(name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name)
    plt.close()


def _bar(frame: pd.DataFrame, x: str, y: str, group: str, name: str, ylabel: str) -> None:
    pivot = frame.pivot_table(index=x, columns=group, values=y, aggfunc="first")
    pivot.plot(kind="bar")
    plt.ylabel(ylabel)
    _save(name)


def main() -> int:  # noqa: PLR0915
    """Render every frozen v2 report artifact and figure."""
    decomposition = pd.read_csv(ROOT / "headroom_decomposition.csv")
    concentration = pd.read_csv(ROOT / "scenario_concentration.csv")
    gaps = pd.read_csv(ROOT / "action_gap_summary.csv")
    metrics = pd.read_csv(ROOT / "predictability_metrics.csv")
    predictions = pd.read_parquet(ROOT / "predictability_predictions.parquet")
    feature_rows = pd.read_parquet(ROOT / "deployable_features.parquet").drop_duplicates("campaign_snapshot_id")
    sparse = pd.read_csv(ROOT / "sparse_residual_metrics.csv")
    importance = pd.read_csv(ROOT / "feature_importance.csv")
    selected_states = pd.read_parquet(ROOT / "nonfeedback_predictions.parquet")

    expanded = []
    for row in decomposition.itertuples(index=False):
        values = json.loads(row.selector_scores_validation_json)
        states = selected_states[
            (selected_states.action_family == row.action_family) & (selected_states.horizon == row.horizon)
        ].copy()
        states["headroom"] = states.oracle_value - states.selected_value
        phase_headroom = states.groupby("phase_three").headroom.mean().to_dict()
        history_headroom = states.groupby("history_generator").headroom.mean().to_dict()
        yahpo = states[states.domain == "yahpo"]
        expanded.append(
            {
                **row._asdict(),
                **{f"value_{key}": value for key, value in values.items()},
                "oracle_minus_global": row.oracle - values["global_static"],
                "oracle_minus_context_deployable": row.oracle - values["context_dimension"],
                "oracle_minus_context_phase_three": row.oracle - values["context_phase_three"],
                "headroom_by_phase_json": json.dumps(phase_headroom, sort_keys=True),
                "headroom_by_history_json": json.dumps(history_headroom, sort_keys=True),
                "yahpo_headroom_excluding_xgboost": yahpo[yahpo.scenario != "rbv2_xgboost"].headroom.mean(),
            }
        )
    decomposition = pd.DataFrame(expanded)
    decomposition.to_csv(ROOT / "headroom_decomposition.csv", index=False)

    # Complete the prespecified concentration summaries without changing any fit.
    summary_rows = []
    for (family, horizon), scenario_rows in concentration[concentration.kind == "yahpo_scenario"].groupby(
        ["action_family", "horizon"]
    ):
        values = scenario_rows.headroom.to_numpy()
        positive = np.clip(values, 0, None)
        total = positive.sum()
        summary_rows.append(
            {
                "action_family": family,
                "horizon": horizon,
                "kind": "yahpo_robust_summary",
                "stratum": "all",
                "headroom": values.mean(),
                "contribution_share": np.nan,
                "leave_one_out": scenario_rows.loc[scenario_rows.stratum != "rbv2_xgboost", "headroom"].mean(),
                "median_scenario_headroom": np.median(values),
                "trimmed_mean_scenario_headroom": np.mean(np.sort(values)[1:-1]),
                "fraction_scenarios_positive": np.mean(values > 0),
                "herfindahl_nonnegative": np.square(positive / total).sum() if total else 0.0,
                "maximum_contribution_share": positive.max() / total if total else 0.0,
            }
        )
    concentration = pd.concat([concentration, pd.DataFrame(summary_rows)], ignore_index=True)
    concentration.to_csv(ROOT / "scenario_concentration.csv", index=False)

    all_metrics = metrics[metrics["subset"] == "all"].copy()
    all_metrics["key"] = list(zip(all_metrics.action_family, all_metrics.horizon, strict=True))
    baselines = decomposition.set_index(["action_family", "horizon"])
    all_metrics["training_selected_nonfeedback"] = [baselines.loc[key].selected_value for key in all_metrics.key]
    all_metrics["oracle"] = [baselines.loc[key].oracle for key in all_metrics.key]
    all_metrics["value_over_nonfeedback"] = all_metrics.selected_value - all_metrics.training_selected_nonfeedback
    denominator = all_metrics.oracle - all_metrics.training_selected_nonfeedback
    all_metrics["captured_residual_headroom"] = np.where(
        denominator > EPSILON, all_metrics.value_over_nonfeedback / denominator, np.nan
    )
    all_metrics.drop(columns="key").to_csv(ROOT / "predictability_metrics.csv", index=False)

    # Expand the primary held-out metrics over prespecified deployable strata.
    prediction_rows = predictions.merge(
        feature_rows[["campaign_snapshot_id", "domain", "scenario", "budget_fraction", "history_generator"]],
        on="campaign_snapshot_id",
        how="left",
        validate="many_to_one",
    )
    stratified = []
    for (family, horizon, model, feature_set), group in prediction_rows.groupby(
        ["action_family", "horizon", "model", "feature_set"], sort=True
    ):
        subsets = {
            "all": group,
            "gap_gt_1e-3": group[group.gap > GAP_SMALL],
            "gap_gt_1e-2": group[group.gap > GAP_MEDIUM],
            "domain:bbob": group[group.domain == "bbob"],
            "domain:yahpo": group[group.domain == "yahpo"],
            "yahpo_excluding_rbv2_xgboost": group[(group.domain == "yahpo") & (group.scenario != "rbv2_xgboost")],
        }
        subsets.update({f"scenario:{key}": value for key, value in group[group.domain == "yahpo"].groupby("scenario")})
        subsets.update({f"phase:{key:.2f}": value for key, value in group.groupby("budget_fraction")})
        subsets.update({f"history:{key}": value for key, value in group.groupby("history_generator")})
        baseline = baselines.loc[(family, horizon)]
        for subset, selected in subsets.items():
            denominator = baseline.oracle - baseline.selected_value
            value = float(selected.selected_value.mean()) if len(selected) else np.nan
            stratified.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "model": model,
                    "feature_set": feature_set,
                    "subset": subset,
                    "n": len(selected),
                    "selected_value": value,
                    "selected_action_regret": selected.selected_action_regret.mean(),
                    "normalized_selected_action_regret": selected.selected_action_regret.mean()
                    / max(abs(selected.oracle_value.mean()), EPSILON),
                    "tie_aware_top_action_accuracy_1e-3": np.mean(selected.selected_action_regret <= GAP_SMALL),
                    "training_selected_nonfeedback": baseline.selected_value,
                    "oracle": baseline.oracle,
                    "value_over_nonfeedback": value - baseline.selected_value,
                    "captured_residual_headroom": (
                        (value - baseline.selected_value) / denominator if denominator > EPSILON else np.nan
                    ),
                    "pairwise_ranking_accuracy": np.nan,
                    "spearman_action_rank": np.nan,
                    "value_calibration": np.nan,
                }
            )
    pd.DataFrame(stratified).to_csv(ROOT / "predictability_metrics.csv", index=False)

    protocol = Path("dacboenv/configs/analysis/headroom_predictability_v2.yaml")
    hashes = {
        "protocol_sha256": _hash(protocol),
        "feature_schema_sha256": json.loads((ROOT / "feature_schema.json").read_text())["sha256"],
        "outputs": {
            path.name: _hash(path)
            for path in sorted(ROOT.iterdir())
            if path.is_file() and path.name != "protocol_hash.json"
        },
    }
    (ROOT / "protocol_hash.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")

    _bar(
        decomposition,
        "horizon",
        "residual_feedback_headroom",
        "action_family",
        "robust_headroom_by_horizon.pdf",
        "Potential improvement",
    )
    _bar(
        decomposition,
        "horizon",
        "residual_feedback_headroom",
        "action_family",
        "headroom_over_nonfeedback_selector.pdf",
        "Oracle minus selected nonfeedback",
    )
    scenarios = concentration[concentration.kind == "yahpo_scenario"]
    for filename, column, ylabel in (
        ("headroom_by_scenario.pdf", "headroom", "Headroom"),
        ("headroom_leave_one_scenario_out.pdf", "leave_one_out", "Leave-one-scenario-out mean"),
    ):
        subset = scenarios[scenarios.horizon == LONG_HORIZON]
        _bar(subset, "stratum", column, "action_family", filename, ylabel)
    without = concentration[concentration.kind == "yahpo_summary"]
    _bar(
        without,
        "horizon",
        "leave_one_out",
        "action_family",
        "headroom_without_xgboost.pdf",
        "YAHPO mean excluding rbv2_xgboost",
    )
    all_gaps = gaps[gaps.group == "all"]
    _bar(all_gaps, "horizon", "median", "action_family", "action_gap_distribution_v2.pdf", "Median top1-top2 gap")
    _bar(all_gaps, "horizon", "fraction_gt_1e-2", "action_family", "headroom_by_gap_bin.pdf", "Fraction gap > 1e-2")
    plt.text(0.5, 0.5, "Unavailable: no corrected paired static trajectories", ha="center", va="center")
    plt.axis("off")
    _save("vbs_vs_same_state_headroom_v2.pdf")
    top = all_metrics.sort_values("selected_value", ascending=False).groupby(["action_family", "horizon"]).head(1)
    _bar(top, "horizon", "selected_value", "action_family", "predictability_by_model.pdf", "Held-out selected value")
    feature = all_metrics.groupby(["feature_set", "action_family"], as_index=False).selected_value.mean()
    _bar(
        feature,
        "feature_set",
        "selected_value",
        "action_family",
        "predictability_by_feature_group.pdf",
        "Mean selected value",
    )
    _bar(
        top,
        "horizon",
        "value_over_nonfeedback",
        "action_family",
        "selected_value_over_context_static.pdf",
        "Value over training-selected nonfeedback",
    )
    _bar(
        top,
        "horizon",
        "captured_residual_headroom",
        "action_family",
        "captured_residual_headroom.pdf",
        "Captured residual headroom",
    )
    _bar(
        sparse,
        "horizon",
        "harmful_override_fraction",
        "action_family",
        "sparse_residual_tradeoff.pdf",
        "Harmful override fraction",
    )
    importance_top = (
        importance.assign(abs_importance=importance.importance.abs())
        .sort_values("abs_importance")
        .groupby(["action_family", "horizon"])
        .tail(8)
    )
    importance_top.groupby("feature").abs_importance.mean().sort_values().plot(kind="barh")
    plt.xlabel("Mean absolute permutation importance")
    _save("feature_importance_by_horizon.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
