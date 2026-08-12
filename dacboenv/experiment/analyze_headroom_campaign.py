"""Analyze executable selectors, uncertainty, predictors, and figures."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from dacboenv.experiment.headroom_predictability import grouped_bootstrap_mean

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BBOB_GROUP_UPPER_BOUNDS = (5, 9, 14, 19, 24)
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 671103
MINIMUM_SELECTOR_SUPPORT = 2


def _metadata(branches: pd.DataFrame) -> pd.DataFrame:
    frame = branches.copy()
    frame["function"] = frame["task_id"].map(
        lambda task: int(task.split("/")[2]) if str(task).startswith("bbob/") else np.nan
    )

    def function_group(function: float) -> float:
        if np.isnan(function):
            return np.nan
        return float(next(index for index, upper in enumerate(BBOB_GROUP_UPPER_BOUNDS) if function <= upper))

    frame["function_group"] = frame["function"].map(function_group)
    frame["phase"] = pd.cut(
        frame["target_budget_fraction"],
        bins=[-np.inf, 0.375, 0.625, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)
    frame["context"] = [
        f"yahpo:{row.scenario}" if row.domain == "yahpo" else f"bbob:d{int(row.dimension)}:g{int(row.function_group)}"
        for row in frame.itertuples(index=False)
    ]
    return frame


def _domain_balanced(frame: pd.DataFrame, value: str) -> float:
    domains = []
    for domain, rows in frame.groupby("domain", sort=True):
        stratum = "dimension" if domain == "bbob" else "scenario"
        domains.append(float(rows.groupby(stratum, sort=True, dropna=False)[value].mean().mean()))
    return float(np.mean(domains))


def _best_action(rows: pd.DataFrame) -> int:
    means = rows.groupby("action", sort=True)["q_value"].mean()
    return int(means[means == means.max()].index.min())


def _action_maps(train: pd.DataFrame) -> dict[str, Any]:
    global_values = {
        int(action): _domain_balanced(rows, "q_value") for action, rows in train.groupby("action", sort=True)
    }
    maximum = max(global_values.values())
    global_action = min(action for action, value in global_values.items() if value == maximum)
    domain = {str(key): _best_action(rows) for key, rows in train.groupby("domain", sort=True)}

    def mapping(columns: list[str]) -> dict[tuple[Any, ...], int]:
        result = {}
        for key, rows in train.groupby(columns, sort=True, dropna=False):
            normalized = key if isinstance(key, tuple) else (key,)
            if rows["campaign_snapshot_id"].nunique() >= MINIMUM_SELECTOR_SUPPORT:
                result[normalized] = _best_action(rows)
        return result

    return {
        "global": global_action,
        "domain": domain,
        "context": mapping(["context"]),
        "phase": mapping(["phase"]),
        "context_phase": mapping(["context", "phase"]),
    }


def _selector_table(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    maps = _action_maps(train)
    matrices = {
        snapshot_id: rows.set_index("action")["q_value"]
        for snapshot_id, rows in validation.groupby("campaign_snapshot_id", sort=True)
    }
    metadata = validation.drop_duplicates("campaign_snapshot_id").set_index("campaign_snapshot_id")
    records = []
    for snapshot_id, q_values in matrices.items():
        row = metadata.loc[snapshot_id]
        global_action = int(maps["global"])
        domain_action = int(maps["domain"].get(str(row.domain), global_action))
        context_action = int(maps["context"].get((row.context,), domain_action))
        phase_action = int(maps["phase"].get((row.phase,), global_action))
        context_phase_action = int(maps["context_phase"].get((row.context, row.phase), context_action))
        values = {
            "global_static": float(q_values.loc[global_action]),
            "context_static": float(q_values.loc[context_action]),
            "phase_only": float(q_values.loc[phase_action]),
            "context_phase": float(q_values.loc[context_phase_action]),
            "oracle": float(q_values.max()),
        }
        records.append({"campaign_snapshot_id": snapshot_id, **row.to_dict(), **values})
    return pd.DataFrame(records), maps


def _bootstrap_difference(rows: pd.DataFrame, left: str, right: str) -> dict[str, float]:
    frame = rows.copy()
    frame["difference"] = frame[left] - frame[right]
    bbob = frame[frame["domain"] == "bbob"]
    yahpo = frame[frame["domain"] == "yahpo"]
    bbob_samples = grouped_bootstrap_mean(
        bbob,
        "difference",
        ["function_group", "function", "task_id", "inner_seed", "campaign_snapshot_id"],
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    yahpo_samples = grouped_bootstrap_mean(
        yahpo,
        "difference",
        ["scenario", "task_id", "inner_seed", "campaign_snapshot_id"],
        n_resamples=BOOTSTRAP_RESAMPLES,
        seed=BOOTSTRAP_SEED + 1,
    )
    samples = 0.5 * bbob_samples + 0.5 * yahpo_samples
    return {
        "median": float(np.median(samples)),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def _save_plot(frame: pd.DataFrame, path: Path, draw: Callable[[Any, pd.DataFrame], None]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    draw(axis, frame)
    figure.savefig(path)
    plt.close(figure)
    frame.to_csv(path.with_name(f"{path.stem}_plot_data.csv"), index=False)


def _figures(  # noqa: C901, PLR0915
    output: Path,
    decomposition: pd.DataFrame,
    branches: pd.DataFrame,
    metrics: pd.DataFrame,
    captured: pd.DataFrame,
    scenario_metrics: pd.DataFrame,
) -> None:
    headroom = decomposition[["action_family", "horizon", "intrinsic_headroom", "ci_low", "ci_high"]]

    def draw_headroom(axis: Any, data: pd.DataFrame) -> None:
        for family, rows in data.groupby("action_family", sort=True):
            axis.plot(rows["horizon"], rows["intrinsic_headroom"], marker="o", label=family)
            axis.fill_between(rows["horizon"], rows["ci_low"], rows["ci_high"], alpha=0.2)
        axis.set(xlabel="Branch horizon", ylabel="Normalized potential headroom")
        axis.legend()

    _save_plot(headroom, output / "headroom_by_horizon.pdf", draw_headroom)

    gains = decomposition.melt(
        id_vars=["action_family", "horizon"],
        value_vars=["context_gain", "phase_gain", "feedback_oracle_gain"],
        var_name="component",
        value_name="gain",
    )

    def draw_gains(axis: Any, data: pd.DataFrame) -> None:
        labels = [f"{row.action_family}/H{row.horizon}" for row in decomposition.itertuples()]
        bottom = np.zeros(len(labels))
        for component in ("context_gain", "phase_gain", "feedback_oracle_gain"):
            values = data[data["component"] == component]["gain"].to_numpy()
            axis.bar(labels, values, bottom=bottom, label=component)
            bottom += values
        axis.tick_params(axis="x", rotation=45)
        axis.set(ylabel="Normalized potential gain")
        axis.legend(fontsize=8)

    _save_plot(gains, output / "headroom_decomposition.pdf", draw_gains)

    gap = (
        branches.groupby(["campaign_snapshot_id", "action_space", "horizon", "phase"], sort=True)["q_value"]
        .apply(lambda values: np.sort(values)[-1] - np.sort(values)[-2])
        .reset_index(name="top1_top2_gap")
    )

    def draw_gap(axis: Any, data: pd.DataFrame) -> None:
        for horizon, rows in data.groupby("horizon", sort=True):
            axis.hist(rows["top1_top2_gap"], bins=30, alpha=0.45, label=f"H={horizon}")
        axis.set(xlabel="Top1-top2 normalized-potential gap", ylabel="Snapshots")
        axis.legend()

    _save_plot(gap, output / "action_gap_distribution.pdf", draw_gap)

    winners = (
        branches.loc[branches.groupby(["campaign_snapshot_id", "horizon"])["q_value"].idxmax()]
        .groupby(["action_space", "phase", "horizon", "action"])
        .size()
        .reset_index(name="count")
    )

    def draw_winners(axis: Any, data: pd.DataFrame) -> None:
        pivot = data.pivot_table(index="phase", columns="action", values="count", aggfunc="sum", fill_value=0)
        pivot.div(pivot.sum(axis=1), axis=0).plot.bar(stacked=True, ax=axis)
        axis.set(ylabel="Winner fraction", xlabel="Budget phase")

    _save_plot(winners, output / "action_winner_by_phase.pdf", draw_winners)

    metric_plot = (
        metrics.groupby(["model", "action_family", "horizon"], sort=True)["mean_selected_action_regret"]
        .mean()
        .reset_index()
    )

    def draw_predictability(axis: Any, data: pd.DataFrame) -> None:
        for model, rows in data.groupby("model", sort=True):
            x = np.arange(len(rows))
            axis.plot(x, rows["mean_selected_action_regret"], marker=".", label=model)
        axis.set(ylabel="Mean selected-action regret", xlabel="Family/horizon cells")
        axis.legend(fontsize=6)

    _save_plot(metric_plot, output / "predictability_by_feature_set.pdf", draw_predictability)

    def draw_captured(axis: Any, data: pd.DataFrame) -> None:
        for model, rows in data.groupby("model", sort=True):
            labels = [
                f"{family}/H{horizon}" for family, horizon in zip(rows["action_family"], rows["horizon"], strict=True)
            ]
            axis.plot(labels, rows["captured_total_headroom"], marker="o", label=model)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.tick_params(axis="x", rotation=45)
        axis.set(ylabel="Captured total headroom")
        axis.legend(fontsize=6)

    _save_plot(captured, output / "captured_headroom_by_model.pdf", draw_captured)

    def draw_scenario(axis: Any, data: pd.DataFrame) -> None:
        selected = data[data["model"].isin(["flat_mlp", "shared_scorer"])]
        for model, rows in selected.groupby("model", sort=True):
            aggregated = rows.groupby("scenario", sort=True)["selected_action_regret"].mean()
            axis.plot(aggregated.index, aggregated.values, marker="o", label=model)
        axis.tick_params(axis="x", rotation=60)
        axis.set(ylabel="Mean selected-action regret", xlabel="YAHPO scenario")
        axis.legend()

    _save_plot(scenario_metrics, output / "predictability_by_scenario.pdf", draw_scenario)


def _predictor_summaries(
    predictions: pd.DataFrame,
    branches: pd.DataFrame,
    selectors: pd.DataFrame,
    decomposition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    q_lookup = branches.set_index(["campaign_snapshot_id", "horizon", "action"])["q_value"]
    metadata = branches.drop_duplicates(["campaign_snapshot_id", "horizon"]).set_index(
        ["campaign_snapshot_id", "horizon"]
    )
    records = []
    for row in predictions.itertuples(index=False):
        key = (row.campaign_snapshot_id, int(row.horizon), int(row.predicted_action))
        meta = metadata.loc[(row.campaign_snapshot_id, int(row.horizon))]
        records.append(
            {
                "campaign_snapshot_id": row.campaign_snapshot_id,
                "action_family": row.action_family,
                "horizon": int(row.horizon),
                "model": row.model,
                "selected_value": float(q_lookup.loc[key]),
                "domain": meta.domain,
                "scenario": meta.scenario,
                "dimension": meta.dimension,
            }
        )
    values = pd.DataFrame(records)
    oracle_lookup = selectors.set_index(["campaign_snapshot_id", "horizon"])["oracle"]
    values["oracle"] = [
        float(oracle_lookup.loc[(row.campaign_snapshot_id, row.horizon)]) for row in values.itertuples(index=False)
    ]
    values["selected_action_regret"] = values["oracle"] - values["selected_value"]
    captured_rows = []
    for (family, horizon, model), rows in values.groupby(["action_family", "horizon", "model"], sort=True):
        selected_value = _domain_balanced(rows, "selected_value")
        reference = decomposition[
            (decomposition["action_family"] == family) & (decomposition["horizon"] == horizon)
        ].iloc[0]
        total_denominator = float(reference["oracle"] - reference["global_static"])
        feedback_denominator = float(reference["oracle"] - reference["context_phase"])
        captured_rows.append(
            {
                "action_family": family,
                "horizon": int(horizon),
                "model": model,
                "selected_value": selected_value,
                "captured_total_headroom": (selected_value - float(reference["global_static"])) / total_denominator,
                "captured_feedback_headroom": (selected_value - float(reference["context_phase"]))
                / feedback_denominator,
            }
        )
    scenario = (
        values[values["domain"] == "yahpo"]
        .groupby(["action_family", "horizon", "model", "scenario"], sort=True)["selected_action_regret"]
        .mean()
        .reset_index()
    )
    return pd.DataFrame(captured_rows), scenario


def main(argv: Sequence[str] | None = None) -> int:
    """Run train-selected held-out decomposition and uncertainty analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, default=Path("artifacts"))
    args = parser.parse_args(argv)
    branches = _metadata(pd.read_parquet(args.artifact_directory / "branch_results.parquet"))
    metrics = pd.read_csv(args.artifact_directory / "predictability_metrics.csv")
    predictions = pd.read_parquet(args.artifact_directory / "predictor_predictions.parquet")
    decomposition_rows, selector_rows, bootstrap_rows = [], [], []
    for family in ("wei", "af_selection"):
        for horizon in (1, 5, 10):
            train = branches[
                (branches["split"] == "train") & (branches["action_space"] == family) & (branches["horizon"] == horizon)
            ]
            validation = branches[
                (branches["split"] == "validation")
                & (branches["action_space"] == family)
                & (branches["horizon"] == horizon)
            ]
            selected, maps = _selector_table(train, validation)
            selected["action_family"] = family
            selected["horizon"] = horizon
            selector_rows.append(selected)
            values = {
                name: _domain_balanced(selected, name)
                for name in ("global_static", "context_static", "phase_only", "context_phase", "oracle")
            }
            interval = _bootstrap_difference(selected, "oracle", "global_static")
            record = {
                "action_family": family,
                "horizon": horizon,
                **values,
                "context_gain": values["context_static"] - values["global_static"],
                "phase_gain": values["context_phase"] - values["context_static"],
                "feedback_oracle_gain": values["oracle"] - values["context_phase"],
                "intrinsic_headroom": values["oracle"] - values["global_static"],
                "global_action": int(maps["global"]),
                "ci_low": interval["ci_low"],
                "ci_high": interval["ci_high"],
            }
            decomposition_rows.append(record)
            for comparison, left, right in (
                ("intrinsic_headroom", "oracle", "global_static"),
                ("context_gain", "context_static", "global_static"),
                ("phase_gain", "context_phase", "context_static"),
                ("feedback_oracle_gain", "oracle", "context_phase"),
            ):
                bootstrap_rows.append(
                    {
                        "action_family": family,
                        "horizon": horizon,
                        "comparison": comparison,
                        **_bootstrap_difference(selected, left, right),
                    }
                )
    decomposition = pd.DataFrame(decomposition_rows)
    selectors = pd.concat(selector_rows, ignore_index=True)
    bootstrap = pd.DataFrame(bootstrap_rows)
    captured, scenario_metrics = _predictor_summaries(predictions, branches, selectors, decomposition)
    decomposition.to_csv(args.artifact_directory / "headroom_decomposition.csv", index=False)
    selectors.to_parquet(args.artifact_directory / "selector_predictions.parquet", index=False)
    bootstrap.to_csv(args.artifact_directory / "headroom_bootstrap.csv", index=False)
    captured.to_csv(args.artifact_directory / "captured_headroom.csv", index=False)
    scenario_metrics.to_csv(args.artifact_directory / "predictability_by_scenario.csv", index=False)
    (args.artifact_directory / "headroom_decomposition.json").write_text(
        json.dumps(decomposition_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _figures(args.artifact_directory, decomposition, branches, metrics, captured, scenario_metrics)
    print(decomposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
