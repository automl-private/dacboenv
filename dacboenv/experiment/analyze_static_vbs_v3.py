"""Analyze the completed, fingerprint-paired static-WEI v3 trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EPSILON = 1e-6
TIE_SMALL = 1e-4
TIE_MEDIUM = 1e-3


def _potential(incumbent: float, initial_incumbent: float, reference: float) -> float:
    denominator = max(initial_incumbent - reference, EPSILON)
    regret = max(incumbent - reference, 0.0) / denominator
    return float(-np.log(regret + EPSILON) / -np.log(EPSILON))


def main() -> int:
    """Produce tie-aware VBS paths and their same-state branch comparison."""
    root = Path("artifacts/headroom_predictability_v3")
    trajectories = pd.read_parquet(root / "paired_static_trajectories.parquet")
    snapshots = pd.concat(
        [
            pd.read_parquet("artifacts/headroom_train_snapshots.parquet"),
            pd.read_parquet("artifacts/headroom_validation_snapshots.parquet"),
        ]
    ).drop_duplicates(["split", "task_id", "inner_seed"])
    reference = snapshots.set_index(["split", "task_id", "inner_seed"])["reference_value"].to_dict()
    expanded = []
    for row in trajectories.itertuples(index=False):
        initial_costs = json.loads(row.initial_design_costs_json)
        initial_incumbent = float(np.min(initial_costs))
        reference_value = float(reference[(row.split, row.task_id, row.inner_seed)])
        for point in json.loads(row.trajectory_json):
            expanded.append(
                {
                    "split": row.split,
                    "task_id": row.task_id,
                    "inner_seed": row.inner_seed,
                    "domain": row.domain,
                    "scenario": row.scenario,
                    "dimension": row.dimension,
                    "action": row.action,
                    "alpha": row.alpha,
                    "bo_evaluation": point["bo_evaluation"],
                    "incumbent": point["incumbent"],
                    "potential": _potential(point["incumbent"], initial_incumbent, reference_value),
                }
            )
    panel = pd.DataFrame(expanded)
    vbs_rows = []
    keys = ["split", "task_id", "inner_seed", "bo_evaluation"]
    for key, group in panel.groupby(keys, sort=True):
        values = group.potential.to_numpy()
        order = np.argsort(values)
        maximum = values[order[-1]]
        margin = maximum - values[order[-2]]
        record = dict(zip(keys, key, strict=True))
        record.update(
            {
                "best_action": int(group.iloc[order[-1]].action),
                "best_alpha": float(group.iloc[order[-1]].alpha),
                "vbs_value": maximum,
                "top1_top2_margin": margin,
                "tied_exact_json": json.dumps(
                    group.loc[np.isclose(values, maximum, rtol=0, atol=0), "action"].tolist()
                ),
                "tied_1e-4_json": json.dumps(group.loc[maximum - values <= TIE_SMALL, "action"].tolist()),
                "tied_1e-3_json": json.dumps(group.loc[maximum - values <= TIE_MEDIUM, "action"].tolist()),
            }
        )
        vbs_rows.append(record)
    vbs = pd.DataFrame(vbs_rows)
    vbs.to_parquet(root / "static_vbs_v3.parquet", index=False)
    summary = []
    for key, group in vbs.groupby(["split", "task_id", "inner_seed"], sort=True):
        winners = group.sort_values("bo_evaluation").best_action.to_numpy()
        frequencies = pd.Series(winners).value_counts(normalize=True).to_numpy()
        summary.append(
            {
                "split": key[0],
                "task_id": key[1],
                "inner_seed": key[2],
                "winner_change_count": int(np.sum(winners[1:] != winners[:-1])),
                "winner_entropy": float(-np.sum(frequencies * np.log(frequencies))),
                "mean_margin": group.top1_top2_margin.mean(),
                "final_vbs_value": group.sort_values("bo_evaluation").vbs_value.iloc[-1],
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(root / "static_vbs_summary_v3.csv", index=False)

    states = pd.read_parquet(root / "corrected_state_values.parquet")
    states = states[(states.action_space == "wei")].copy()
    joined = states.merge(summary_frame, on=["split", "task_id", "inner_seed"], validate="many_to_one")
    nearest = []
    for row in joined.itertuples(index=False):
        path = vbs[(vbs.split == row.split) & (vbs.task_id == row.task_id) & (vbs.inner_seed == row.inner_seed)]
        target = float(row.target_budget_fraction) * path.bo_evaluation.max()
        selected = path.iloc[int(np.argmin(np.abs(path.bo_evaluation.to_numpy() - target)))]
        nearest.append(
            {
                "campaign_snapshot_id": row.campaign_snapshot_id,
                "horizon": row.horizon,
                "winner_change_count": row.winner_change_count,
                "vbs_margin": selected.top1_top2_margin,
                "branch_headroom": row.residual_headroom,
                "branch_gap": row.gap,
            }
        )
    comparison = pd.DataFrame(nearest)
    comparison.to_csv(root / "vbs_vs_branch_headroom_v3.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
