"""Render currently available corrected v3 scenario outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt

ROOT = Path("artifacts/headroom_predictability_v3")
FIGURES = Path("figures")
LONG_HORIZON = 10


def _save(name: str, data: pd.DataFrame) -> None:
    FIGURES.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURES / name)
    plt.close()
    data.to_csv(ROOT / name.replace(".pdf", "_plot_data.csv"), index=False)


def main() -> int:  # noqa: PLR0915
    """Create noninteractive figures without favorable fixed axis limits."""
    local = pd.read_csv(ROOT / "scenario_local_headroom.csv")
    scenarios = local[local.subgroup.str.startswith("scenario:")].copy()
    scenarios["scenario"] = scenarios.subgroup.str.removeprefix("scenario:")
    selected = scenarios[scenarios.horizon == LONG_HORIZON]
    selected.pivot_table(index="scenario", columns="action_family", values="residual_headroom").plot(kind="bar")
    plt.ylabel("Scenario-local residual headroom")
    _save("scenario_residual_headroom_v3.pdf", selected)

    bootstrap = pd.read_csv(ROOT / "scenario_exclusion_bootstrap.csv")
    selected = bootstrap[bootstrap.horizon == LONG_HORIZON]
    for family, group in selected.groupby("action_family"):
        plt.errorbar(
            group.excluded_scenario,
            group["median"],
            yerr=[group["median"] - group.ci_low, group.ci_high - group["median"]],
            marker="o",
            label=family,
        )
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Residual headroom after exclusion")
    plt.legend()
    _save("scenario_exclusion_intervals_v3.pdf", selected)

    concentration = pd.read_csv(ROOT / "scenario_concentration_v3.csv")
    concentration.pivot_table(index="horizon", columns="action_family", values="effective_contributing_scenarios").plot(
        marker="o"
    )
    plt.ylabel("Effective contributing scenarios")
    _save("headroom_concentration_v3.pdf", concentration)

    metrics_path = ROOT / "predictability_metrics_v3.csv"
    if metrics_path.is_file():
        metrics = pd.read_csv(metrics_path)
        scenario_metrics = metrics[metrics.subgroup.str.startswith("scenario:")]
        best = (
            scenario_metrics.sort_values("selected_value")
            .groupby(["action_family", "horizon", "subgroup"], as_index=False)
            .tail(1)
        )
        chosen = best[best.horizon == LONG_HORIZON]
        chosen.pivot_table(index="subgroup", columns="action_family", values="value_over_nonfeedback").plot(kind="bar")
        plt.ylabel("Best frozen model value over local nonfeedback")
        _save("predictability_by_corrected_scenario.pdf", chosen)
        excluded = metrics[metrics.subgroup == "yahpo_excluding_rbv2_xgboost"]
        best_excluded = (
            excluded.sort_values("selected_value").groupby(["action_family", "horizon"], as_index=False).tail(1)
        )
        best_excluded.pivot_table(index="horizon", columns="action_family", values="value_over_nonfeedback").plot(
            kind="bar"
        )
        plt.ylabel("Value over local nonfeedback, xgboost excluded")
        _save("predictability_without_xgboost.pdf", best_excluded)

        all_states = metrics[metrics.subgroup == "all"].copy()
        feature_best = (
            all_states.sort_values("selected_value")
            .groupby(["action_family", "horizon", "feature_set"], as_index=False)
            .tail(1)
        )
        feature_best["cell"] = feature_best.action_family + " H=" + feature_best.horizon.astype(str)
        feature_best.pivot_table(index="cell", columns="feature_set", values="value_over_nonfeedback").plot(kind="bar")
        plt.ylabel("Best frozen model value over nonfeedback")
        _save("feature_group_selected_value_v3.pdf", feature_best)

        improvement = pd.read_csv(ROOT / "model_improvement_bootstrap.csv")
        chosen_models = improvement[~improvement.model.str.contains("shuffled|permutation|row_mean", regex=True)]
        chosen_models = (
            chosen_models.sort_values("median").groupby(["action_family", "horizon"], as_index=False).tail(1)
        )
        chosen_models["cell"] = chosen_models.action_family + " H=" + chosen_models.horizon.astype(str)
        plt.errorbar(
            chosen_models.cell,
            chosen_models["median"],
            yerr=[
                chosen_models["median"] - chosen_models.ci_low,
                chosen_models.ci_high - chosen_models["median"],
            ],
            fmt="o",
        )
        plt.axhline(0, color="black", linewidth=0.8)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Value improvement over frozen nonfeedback")
        _save("model_improvement_over_nonfeedback_v3.pdf", chosen_models)

        true_gru = all_states[all_states.model.str.startswith("short_history_gru")].copy()
        padded_path = Path("artifacts/headroom_predictability_v2/predictability_metrics.csv")
        if padded_path.is_file() and len(true_gru):
            padded = pd.read_csv(padded_path)
            padded = padded[(padded.get("subset", "all") == "all") & padded.model.str.contains("short_history_gru")]
            padded = padded.assign(history="v2 padded")
            true = true_gru.assign(history="v3 true")
            common = ["action_family", "horizon", "model", "selected_value", "history"]
            history_comparison = pd.concat([padded[common], true[common]], ignore_index=True)
            history_comparison["cell"] = (
                history_comparison.action_family + " H=" + history_comparison.horizon.astype(str)
            )
            history_comparison.pivot_table(index="cell", columns="history", values="selected_value").plot(kind="bar")
            plt.ylabel("Selected action value")
            _save("true_history_vs_padded_gru.pdf", history_comparison)

    static_path = ROOT / "static_vbs_v3.parquet"
    comparison_path = ROOT / "vbs_vs_branch_headroom_v3.csv"
    if static_path.is_file():
        static = pd.read_parquet(static_path)
        sample = static.groupby("bo_evaluation", as_index=False).best_action.mean()
        sample.plot(x="bo_evaluation", y="best_action")
        plt.ylabel("Mean winning static-WEI action")
        _save("paired_static_vbs_winner_paths.pdf", sample)
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        comparison.plot.scatter(x="winner_change_count", y="branch_headroom")
        _save("vbs_fluctuation_vs_same_state_headroom_v3.pdf", comparison)
        comparison.plot.scatter(x="vbs_margin", y="branch_gap")
        _save("branch_gap_vs_vbs_margin.pdf", comparison)

    protocol = Path("dacboenv/configs/analysis/headroom_predictability_v3.yaml")
    files = sorted(path for path in ROOT.iterdir() if path.is_file() and path.name != "protocol_hash.json")
    payload = {
        "protocol_sha256": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
    }
    (ROOT / "protocol_hash.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
