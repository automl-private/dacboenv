"""Correct scenario propagation and subgroup-local v3 headroom metrics."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from dacboenv.experiment.task_metadata import EXPECTED_YAHPO_SCENARIOS, parse_task_metadata

RESAMPLES = 2000
SEED = 881203
EPSILON = 1e-12
GAP_SMALL = 1e-3
GAP_MEDIUM = 1e-2


def attach_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical metadata and reject every malformed YAHPO row."""
    frame = rows.copy()
    metadata = frame.task_id.map(parse_task_metadata)
    frame["domain"] = metadata.map(lambda item: item.domain)
    frame["scenario"] = metadata.map(lambda item: item.scenario)
    frame["dataset_instance"] = metadata.map(lambda item: item.dataset_instance)
    frame["function_id"] = metadata.map(lambda item: item.function_id)
    frame["dimension"] = metadata.map(lambda item: item.dimension)
    frame["native_instance"] = metadata.map(lambda item: item.native_instance)
    invalid = frame[(frame.domain == "yahpo") & (~frame.scenario.isin(EXPECTED_YAHPO_SCENARIOS))]
    if len(invalid):
        raise ValueError("YAHPO rows reached analysis without one canonical scenario.")
    return frame


def captured_ratio(model: float, nonfeedback: float, oracle: float) -> float:
    """Return a subgroup-local ratio or NaN for a negligible denominator."""
    denominator = oracle - nonfeedback
    return np.nan if abs(denominator) < EPSILON else (model - nonfeedback) / denominator


def _state_table(branches: pd.DataFrame, nonfeedback: pd.DataFrame) -> pd.DataFrame:
    matrices = branches.pivot_table(
        index=["campaign_snapshot_id", "action_space", "horizon"], columns="action", values="q_value"
    )
    action_values = matrices.to_numpy()
    ordered_values = np.sort(action_values, axis=1)
    matrices["oracle_value"] = ordered_values[:, -1]
    matrices["gap"] = ordered_values[:, -1] - ordered_values[:, -2]
    meta = attach_metadata(branches.drop_duplicates(["campaign_snapshot_id", "action_space", "horizon"]))
    state = meta.merge(matrices.reset_index(), on=["campaign_snapshot_id", "action_space", "horizon"])
    keep = nonfeedback[["campaign_snapshot_id", "action_family", "horizon", "selected_action", "selected_value"]]
    state = state.merge(
        keep,
        left_on=["campaign_snapshot_id", "action_space", "horizon"],
        right_on=["campaign_snapshot_id", "action_family", "horizon"],
        validate="one_to_one",
    )
    state["residual_headroom"] = state.oracle_value - state.selected_value
    state["trajectory"] = (
        state.task_id.astype(str)
        + "|"
        + state.inner_seed.astype(str)
        + "|"
        + state.history_generator.astype(str)
        + "|"
        + state.action_space.astype(str)
    )
    return state


def _subgroups(states: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = {
        "all": states,
        "domain:bbob": states[states.domain == "bbob"],
        "domain:yahpo": states[states.domain == "yahpo"],
        "yahpo_excluding_rbv2_xgboost": states[(states.domain == "yahpo") & (states.scenario != "rbv2_xgboost")],
        "gap_gt_1e-3": states[states.gap > GAP_SMALL],
        "gap_gt_1e-2": states[states.gap > GAP_MEDIUM],
    }
    groups.update({f"scenario:{key}": value for key, value in states[states.domain == "yahpo"].groupby("scenario")})
    groups.update(
        {
            f"yahpo_leave_out:{scenario}": states[(states.domain == "yahpo") & (states.scenario != scenario)]
            for scenario in sorted(EXPECTED_YAHPO_SCENARIOS)
        }
    )
    groups.update(
        {f"bbob_dimension:{key}": value for key, value in states[states.domain == "bbob"].groupby("dimension")}
    )
    groups.update({f"budget_phase:{key}": value for key, value in states.groupby("target_budget_fraction")})
    groups.update({f"history:{key}": value for key, value in states.groupby("history_generator")})
    return groups


def _bootstrap_yahpo(states: pd.DataFrame, exclusion: str | None, rng: np.random.Generator) -> np.ndarray:
    source = states[states.domain == "yahpo"]
    if exclusion:
        source = source[source.scenario != exclusion]
    scenario_groups: dict[str, tuple[tuple[np.ndarray, ...], ...]] = {}
    for scenario, scenario_rows in source.groupby("scenario", sort=True):
        instances = []
        for _instance, instance_rows in scenario_rows.groupby("dataset_instance", sort=True):
            instances.append(
                tuple(
                    trajectory_rows.residual_headroom.to_numpy()
                    for _trajectory, trajectory_rows in instance_rows.groupby("trajectory", sort=True)
                )
            )
        scenario_groups[str(scenario)] = tuple(instances)
    results = []
    for _ in range(RESAMPLES):
        scenario_values = []
        for instances in scenario_groups.values():
            sampled_instances = rng.integers(len(instances), size=len(instances))
            values = []
            for instance_index in sampled_instances:
                trajectories = instances[int(instance_index)]
                sampled_trajectories = rng.integers(len(trajectories), size=len(trajectories))
                for trajectory_index in sampled_trajectories:
                    trajectory_values = trajectories[int(trajectory_index)]
                    indices = rng.integers(len(trajectory_values), size=len(trajectory_values))
                    values.extend(trajectory_values[indices])
            scenario_values.append(float(np.mean(values)))
        results.append(float(np.mean(scenario_values)))
    return np.asarray(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Write corrected scenario, subgroup, and exclusion artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/headroom_predictability_v3"))
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    branches = attach_metadata(pd.read_parquet("artifacts/branch_results.parquet"))
    nonfeedback = pd.read_parquet("artifacts/headroom_predictability_v2/nonfeedback_predictions.parquet")
    states = _state_table(branches, nonfeedback)

    counts = []
    snapshots = pd.concat(
        [
            pd.read_parquet("artifacts/headroom_train_snapshots.parquet"),
            pd.read_parquet("artifacts/headroom_validation_snapshots.parquet"),
        ]
    )
    snapshots = attach_metadata(snapshots)
    for (split, scenario), group in snapshots[snapshots.domain == "yahpo"].groupby(["split", "scenario"]):
        branch_group = branches[(branches.split == split) & (branches.scenario == scenario)]
        counts.append(
            {
                "split": split,
                "scenario": scenario,
                "trajectories": group.groupby(["task_id", "inner_seed", "history_generator", "action_space"]).ngroups,
                "snapshots": len(group),
                "action_rows": len(branch_group.drop_duplicates(["campaign_snapshot_id", "action_space", "action"])),
                "branch_outcomes": len(branch_group),
                "budget_phase_counts_json": json.dumps(group.budget_fraction.value_counts().sort_index().to_dict()),
                "history_generator_counts_json": json.dumps(
                    group.history_generator.value_counts().sort_index().to_dict()
                ),
            }
        )
    count_frame = pd.DataFrame(counts)
    assert set(count_frame.scenario) == EXPECTED_YAHPO_SCENARIOS
    count_frame.to_csv(args.output / "scenario_propagation_counts.csv", index=False)

    local_rows = []
    for (family, horizon), family_states in states.groupby(["action_space", "horizon"]):
        for subgroup, group in _subgroups(family_states).items():
            if not len(group):
                continue
            local_rows.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "subgroup": subgroup,
                    "n": len(group),
                    "nonfeedback_value": group.selected_value.mean(),
                    "oracle_value": group.oracle_value.mean(),
                    "residual_headroom": group.residual_headroom.mean(),
                    "mean_gap": group.gap.mean(),
                }
            )
    pd.DataFrame(local_rows).to_csv(args.output / "scenario_local_headroom.csv", index=False)

    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    for (family, horizon), family_states in states.groupby(["action_space", "horizon"]):
        for exclusion in (None, *sorted(EXPECTED_YAHPO_SCENARIOS)):
            samples = _bootstrap_yahpo(family_states, exclusion, rng)
            bootstrap_rows.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "excluded_scenario": exclusion or "none",
                    "median": np.median(samples),
                    "ci_low": np.quantile(samples, 0.025),
                    "ci_high": np.quantile(samples, 0.975),
                    "resamples": RESAMPLES,
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(args.output / "scenario_exclusion_bootstrap.csv", index=False)

    concentration = []
    scenarios = pd.DataFrame(local_rows)
    scenarios = scenarios[scenarios.subgroup.str.startswith("scenario:")]
    for (family, horizon), group in scenarios.groupby(["action_family", "horizon"]):
        values = group.residual_headroom.to_numpy()
        nonnegative = np.clip(values, 0, None)
        total = nonnegative.sum()
        concentration.append(
            {
                "action_family": family,
                "horizon": horizon,
                "median": np.median(values),
                "trimmed_mean": np.mean(np.sort(values)[1:-1]),
                "minimum": values.min(),
                "maximum": values.max(),
                "fraction_positive": np.mean(values > 0),
                "maximum_contribution_share": nonnegative.max() / total if total else 0.0,
                "herfindahl": np.square(nonnegative / total).sum() if total else 0.0,
                "effective_contributing_scenarios": 1 / np.square(nonnegative / total).sum() if total else 0.0,
            }
        )
    pd.DataFrame(concentration).to_csv(args.output / "scenario_concentration_v3.csv", index=False)
    states.to_parquet(args.output / "corrected_state_values.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
