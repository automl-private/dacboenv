"""Frozen v2 robust selector, concentration, gap, and bootstrap analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BBOB_UPPER = (5, 9, 14, 19, 24)
SELECTOR_CLASSES = (
    "global_static",
    "domain_static",
    "context_dimension",
    "context_dimension_function_group_privileged",
    "phase_three",
    "phase_five",
    "context_phase_three",
    "context_phase_five",
)
DEPLOYABLE_SELECTOR_CLASSES = tuple(
    selector for selector in SELECTOR_CLASSES if selector != "context_dimension_function_group_privileged"
)
BOOTSTRAP_RESAMPLES = 2000
MINIMUM_SUPPORT = 2
GAP_1E4 = 1e-4
GAP_1E3 = 1e-3
GAP_1E2 = 1e-2
GAP_5E2 = 5e-2


def enrich(rows: pd.DataFrame) -> pd.DataFrame:
    """Add only deterministic task/phase grouping columns."""
    frame = rows.copy()
    frame["function"] = frame.task_id.map(
        lambda value: int(value.split("/")[2]) if str(value).startswith("bbob/") else np.nan
    )
    frame["function_group"] = frame.function.map(
        lambda value: (
            np.nan if np.isnan(value) else next(index for index, upper in enumerate(BBOB_UPPER) if value <= upper)
        )
    )
    frame["phase_three"] = pd.cut(
        frame.target_budget_fraction,
        [-np.inf, 0.375, 0.625, np.inf],
        labels=["early", "middle", "late"],
    ).astype(str)
    frame["phase_five"] = np.minimum((frame.target_budget_fraction * 5).astype(int), 4)
    frame["trajectory"] = (
        frame.task_id.astype(str)
        + "|"
        + frame.inner_seed.astype(str)
        + "|"
        + frame.history_generator.astype(str)
        + "|"
        + frame.action_space.astype(str)
    )
    frame["context_dimension"] = [
        f"yahpo:{row.scenario}" if row.domain == "yahpo" else f"bbob:d{int(row.dimension)}"
        for row in frame.itertuples(index=False)
    ]
    frame["context_privileged"] = [
        row.context_dimension if row.domain == "yahpo" else f"{row.context_dimension}:g{int(row.function_group)}"
        for row in frame.itertuples(index=False)
    ]
    return frame


def domain_balanced(rows: pd.DataFrame, value: str) -> float:
    """Equal-weight domains, BBOB dimensions, and YAHPO scenarios."""
    values = []
    for domain, group in rows.groupby("domain", sort=True):
        stratum = "dimension" if domain == "bbob" else "scenario"
        values.append(float(group.groupby(stratum, dropna=False)[value].mean().mean()))
    return float(np.mean(values))


def _best_action(rows: pd.DataFrame) -> int:
    means = rows.groupby("action", sort=True).q_value.mean()
    return int(means[means == means.max()].index.min())


def fit_selector(rows: pd.DataFrame, selector: str) -> dict[str, Any]:
    """Fit one prespecified nonfeedback selector."""
    global_action = _best_action(rows)
    columns = {
        "global_static": [],
        "domain_static": ["domain"],
        "context_dimension": ["context_dimension"],
        "context_dimension_function_group_privileged": ["context_privileged"],
        "phase_three": ["phase_three"],
        "phase_five": ["phase_five"],
        "context_phase_three": ["context_dimension", "phase_three"],
        "context_phase_five": ["context_dimension", "phase_five"],
    }[selector]
    mapping: dict[str, int] = {}
    if columns:
        for key, group in rows.groupby(columns, sort=True, dropna=False):
            normalized = key if isinstance(key, tuple) else (key,)
            if group.campaign_snapshot_id.nunique() >= MINIMUM_SUPPORT:
                mapping[json.dumps(normalized, default=str)] = _best_action(group)
    return {"selector": selector, "global_action": global_action, "columns": columns, "mapping": mapping}


def selector_action(model: dict[str, Any], row: Any) -> int:
    """Apply a fitted selector with deterministic global fallback."""
    key = tuple(getattr(row, column) for column in model["columns"])
    return int(model["mapping"].get(json.dumps(key, default=str), model["global_action"]))


def evaluate_selector(rows: pd.DataFrame, model: dict[str, Any]) -> pd.DataFrame:
    """Return one selected outcome per snapshot."""
    metadata = rows.drop_duplicates("campaign_snapshot_id").set_index("campaign_snapshot_id")
    matrices = {
        snapshot: group.set_index("action").q_value
        for snapshot, group in rows.groupby("campaign_snapshot_id", sort=True)
    }
    records = []
    for snapshot, matrix in matrices.items():
        meta = metadata.loc[snapshot]
        action = selector_action(model, meta)
        records.append(
            {
                **meta.to_dict(),
                "campaign_snapshot_id": snapshot,
                "selected_action": action,
                "selected_value": float(matrix.loc[action]),
                "oracle_value": float(matrix.max()),
            }
        )
    return pd.DataFrame(records)


def _development_partition(rows: pd.DataFrame, seed: int = 904212) -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = np.asarray(sorted(rows.trajectory.unique()))
    rng = np.random.default_rng(seed)
    development = set(trajectories[rng.permutation(len(trajectories))[: max(1, len(trajectories) // 5)]])
    return rows[~rows.trajectory.isin(development)], rows[rows.trajectory.isin(development)]


def select_nonfeedback(train: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Select a class on grouped training development, never validation."""
    fitting, development = _development_partition(train)
    scores = {
        selector: domain_balanced(evaluate_selector(development, fit_selector(fitting, selector)), "selected_value")
        for selector in DEPLOYABLE_SELECTOR_CLASSES
    }
    selected = sorted(scores, key=lambda name: (-scores[name], DEPLOYABLE_SELECTOR_CLASSES.index(name)))[0]
    return selected, scores


def _hierarchy_trees(rows: pd.DataFrame) -> list[Any]:
    """Compile immutable hierarchy trees once, avoiding bootstrap groupby work."""
    metadata = rows.drop_duplicates("campaign_snapshot_id")

    def build(frame: pd.DataFrame, levels: list[str], depth: int = 0) -> Any:
        if depth == len(levels):
            return str(frame.campaign_snapshot_id.iloc[0])
        return tuple(build(group, levels, depth + 1) for _key, group in frame.groupby(levels[depth], sort=True))

    trees = []
    for domain, domain_rows in metadata.groupby("domain", sort=True):
        levels = (
            ["function_group", "function", "trajectory", "campaign_snapshot_id"]
            if domain == "bbob"
            else ["scenario", "task_id", "trajectory", "campaign_snapshot_id"]
        )
        trees.append(build(domain_rows, levels))
    return trees


def _resample_hierarchy(rows: pd.DataFrame, trees: list[Any], rng: np.random.Generator) -> pd.DataFrame:
    def draw(node: Any) -> list[str]:
        if isinstance(node, str):
            return [node]
        return [leaf for index in rng.integers(len(node), size=len(node)) for leaf in draw(node[int(index)])]

    snapshot_ids = [snapshot for tree in trees for snapshot in draw(tree)]
    indexed = rows.set_index("campaign_snapshot_id", drop=False)
    output = indexed.loc[snapshot_ids].copy().reset_index(drop=True)
    sizes = rows.groupby("campaign_snapshot_id", sort=False).size().to_dict()
    output["campaign_snapshot_id"] = [
        label for index, snapshot in enumerate(snapshot_ids) for label in [f"bootstrap-{index}"] * sizes[snapshot]
    ]
    return output


def _bootstrap(  # noqa: C901, PLR0915
    train: pd.DataFrame,
    validation: pd.DataFrame,
    selected_name: str,
    selected_model: dict[str, Any],
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap the hierarchy with compiled snapshot arrays.

    This is numerically equivalent to duplicating the five action rows for
    every sampled snapshot, but avoids millions of pandas groupby operations.
    """
    rng = np.random.default_rng(seed)
    conditional, full = [], []

    def arrays(rows: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        meta = rows.drop_duplicates("campaign_snapshot_id").sort_values("campaign_snapshot_id").reset_index(drop=True)
        q = (
            rows.pivot_table(index="campaign_snapshot_id", columns="action", values="q_value", aggfunc="first")
            .loc[meta.campaign_snapshot_id]
            .to_numpy()
        )
        return meta, q

    train_meta, train_q = arrays(train)
    validation_meta, validation_q = arrays(validation)

    def integer_trees(meta: pd.DataFrame) -> list[Any]:
        indexed = meta.assign(_leaf=np.arange(len(meta)))

        def build(frame: pd.DataFrame, levels: list[str], depth: int = 0) -> Any:
            if depth == len(levels):
                return int(frame._leaf.iloc[0])
            return tuple(build(group, levels, depth + 1) for _key, group in frame.groupby(levels[depth], sort=True))

        output = []
        for domain, domain_rows in indexed.groupby("domain", sort=True):
            levels = (
                ["function_group", "function", "trajectory", "campaign_snapshot_id"]
                if domain == "bbob"
                else ["scenario", "task_id", "trajectory", "campaign_snapshot_id"]
            )
            output.append(build(domain_rows, levels))
        return output

    def draw(trees: list[Any]) -> np.ndarray:
        def recurse(node: Any) -> list[int]:
            if isinstance(node, int):
                return [node]
            return [leaf for index in rng.integers(len(node), size=len(node)) for leaf in recurse(node[int(index)])]

        return np.asarray([leaf for tree in trees for leaf in recurse(tree)], dtype=int)

    train_trees, validation_trees = integer_trees(train_meta), integer_trees(validation_meta)

    def keys(meta: pd.DataFrame, selector: str) -> np.ndarray:
        columns = {
            "global_static": [],
            "domain_static": ["domain"],
            "context_dimension": ["context_dimension"],
            "context_dimension_function_group_privileged": ["context_privileged"],
            "phase_three": ["phase_three"],
            "phase_five": ["phase_five"],
            "context_phase_three": ["context_dimension", "phase_three"],
            "context_phase_five": ["context_dimension", "phase_five"],
        }[selector]
        return (
            np.asarray([json.dumps(tuple(row), default=str) for row in meta[columns].to_numpy()])
            if columns
            else np.full(len(meta), "")
        )

    def fit_numeric(
        meta: pd.DataFrame, q: np.ndarray, sampled: np.ndarray, selector: str
    ) -> tuple[int, dict[Any, int]]:
        global_action = int(np.argmax(q[sampled].mean(axis=0)))
        selector_keys = keys(meta, selector)
        mapping = {}
        for key in dict.fromkeys(selector_keys[sampled].tolist()):
            positions = sampled[selector_keys[sampled] == key]
            if len(positions) >= MINIMUM_SUPPORT:
                mapping[key] = int(np.argmax(q[positions].mean(axis=0)))
        return global_action, mapping

    def selected_values(
        train_indices: np.ndarray, validation_indices: np.ndarray, selector: str
    ) -> tuple[np.ndarray, np.ndarray]:
        global_action, mapping = fit_numeric(train_meta, train_q, train_indices, selector)
        validation_keys = keys(validation_meta, selector)
        actions = np.asarray([mapping.get(key, global_action) for key in validation_keys[validation_indices]])
        return validation_q[validation_indices, actions], validation_q[validation_indices].max(axis=1)

    def balanced(indices: np.ndarray, values: np.ndarray) -> float:
        sampled_meta = validation_meta.iloc[indices].reset_index(drop=True)
        domain_values = []
        for domain in ("bbob", "yahpo"):
            mask = sampled_meta.domain.to_numpy() == domain
            stratum = "dimension" if domain == "bbob" else "scenario"
            stratum_values = [
                values[mask][sampled_meta.loc[mask, stratum].to_numpy() == key].mean()
                for key in sampled_meta.loc[mask, stratum].unique()
            ]
            domain_values.append(float(np.mean(stratum_values)))
        return float(np.mean(domain_values))

    fixed_global = int(selected_model["global_action"])
    fixed_mapping = selected_model["mapping"]
    fixed_columns = selected_model["columns"]

    for _ in range(BOOTSTRAP_RESAMPLES):
        val_indices = draw(validation_trees)
        if fixed_columns:
            fixed_keys = np.asarray(
                [json.dumps(tuple(row), default=str) for row in validation_meta[fixed_columns].to_numpy()]
            )
            fixed_actions = np.asarray([fixed_mapping.get(key, fixed_global) for key in fixed_keys[val_indices]])
        else:
            fixed_actions = np.full(len(val_indices), fixed_global)
        conditional_delta = validation_q[val_indices].max(axis=1) - validation_q[val_indices, fixed_actions]
        conditional.append(balanced(val_indices, conditional_delta))

        train_indices = draw(train_trees)
        fitted_values, oracle_values = selected_values(train_indices, val_indices, selected_name)
        full.append(balanced(val_indices, oracle_values - fitted_values))
    return np.asarray(conditional), np.asarray(full)


def _concentration(selected: pd.DataFrame, family: str, horizon: int) -> pd.DataFrame:
    rows = selected.assign(headroom=selected.oracle_value - selected.selected_value)
    records = []
    yahpo = rows[rows.domain == "yahpo"]
    scenario_means = yahpo.groupby("scenario").headroom.mean()
    nonnegative = scenario_means.clip(lower=0)
    total = float(nonnegative.sum())
    for scenario, value in scenario_means.items():
        loo = float(scenario_means.drop(scenario).mean())
        records.append(
            {
                "action_family": family,
                "horizon": horizon,
                "kind": "yahpo_scenario",
                "stratum": scenario,
                "headroom": value,
                "contribution_share": 0.0 if total == 0 else max(value, 0) / total,
                "leave_one_out": loo,
            }
        )
    records.append(
        {
            "action_family": family,
            "horizon": horizon,
            "kind": "yahpo_summary",
            "stratum": "all",
            "headroom": scenario_means.mean(),
            "contribution_share": float(np.square(nonnegative / total).sum()) if total else 0.0,
            "leave_one_out": scenario_means.drop("rbv2_xgboost", errors="ignore").mean(),
        }
    )
    for column, kind in (("dimension", "bbob_dimension"), ("function_group", "bbob_function_group")):
        for stratum, value in rows[rows.domain == "bbob"].groupby(column).headroom.mean().items():
            records.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "kind": kind,
                    "stratum": str(stratum),
                    "headroom": value,
                    "contribution_share": np.nan,
                    "leave_one_out": np.nan,
                }
            )
    return pd.DataFrame(records)


def _gap_summary(validation: pd.DataFrame, family: str, horizon: int) -> pd.DataFrame:
    keys = ["campaign_snapshot_id", "domain", "scenario", "dimension", "phase_three", "history_generator"]
    gaps = (
        validation.groupby(keys, dropna=False)
        .q_value.apply(lambda values: np.sort(values)[-1] - np.sort(values)[-2])
        .reset_index(name="gap")
    )
    records = []
    for grouping, group in [
        ("all", gaps),
        *[(f"domain:{key}", value) for key, value in gaps.groupby("domain")],
        *[(f"scenario:{key}", value) for key, value in gaps[gaps.domain == "yahpo"].groupby("scenario")],
    ]:
        values = group.gap
        records.append(
            {
                "action_family": family,
                "horizon": horizon,
                "group": grouping,
                "n": len(values),
                "fraction_eq_0": np.mean(values == 0),
                "fraction_le_1e-4": np.mean(values <= GAP_1E4),
                "fraction_le_1e-3": np.mean(values <= GAP_1E3),
                "fraction_le_1e-2": np.mean(values <= GAP_1E2),
                "fraction_gt_1e-2": np.mean(values > GAP_1E2),
                "fraction_gt_5e-2": np.mean(values > GAP_5E2),
                "mean": values.mean(),
                "median": values.median(),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.9),
                "p95": values.quantile(0.95),
            }
        )
    return pd.DataFrame(records)


def main(argv: Sequence[str] | None = None) -> int:
    """Execute frozen v2 robust analysis into its versioned namespace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("artifacts/branch_results.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/headroom_predictability_v2"))
    args = parser.parse_args(argv)
    rows = enrich(pd.read_parquet(args.input))
    args.output.mkdir(parents=True, exist_ok=True)
    decomposition, bootstraps, concentrations, gaps, selected_rows = [], [], [], [], []
    for family in ("wei", "af_selection"):
        for horizon in (1, 5, 10):
            train = rows[(rows.split == "train") & (rows.action_space == family) & (rows.horizon == horizon)]
            validation = rows[(rows.split == "validation") & (rows.action_space == family) & (rows.horizon == horizon)]
            selected_name, development_scores = select_nonfeedback(train)
            selected_model = fit_selector(train, selected_name)
            selected = evaluate_selector(validation, selected_model)
            selected["action_family"], selected["horizon"] = family, horizon
            selected_rows.append(selected)
            all_selector_values = {
                name: domain_balanced(evaluate_selector(validation, fit_selector(train, name)), "selected_value")
                for name in SELECTOR_CLASSES
            }
            nonfeedback_envelope = max(all_selector_values.values())
            oracle = domain_balanced(selected, "oracle_value")
            selected_value = domain_balanced(selected, "selected_value")
            conditional, full = _bootstrap(train, validation, selected_name, selected_model, seed=771103 + horizon)
            decomposition.append(
                {
                    "action_family": family,
                    "horizon": horizon,
                    "selected_class": selected_name,
                    "selected_value": selected_value,
                    "oracle": oracle,
                    "residual_feedback_headroom": oracle - selected_value,
                    "nonfeedback_selector_oracle_analysis_only": nonfeedback_envelope,
                    "selector_scores_validation_json": json.dumps(all_selector_values, sort_keys=True),
                    "development_scores_json": json.dumps(development_scores, sort_keys=True),
                }
            )
            for kind, samples in (("conditional_validation", conditional), ("full_protocol", full)):
                bootstraps.append(
                    {
                        "action_family": family,
                        "horizon": horizon,
                        "kind": kind,
                        "median": np.median(samples),
                        "ci_low": np.quantile(samples, 0.025),
                        "ci_high": np.quantile(samples, 0.975),
                        "n_resamples": BOOTSTRAP_RESAMPLES,
                    }
                )
            concentrations.append(_concentration(selected, family, horizon))
            gaps.append(_gap_summary(validation, family, horizon))
    pd.DataFrame(decomposition).to_csv(args.output / "headroom_decomposition.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(args.output / "headroom_bootstrap.csv", index=False)
    pd.concat(concentrations, ignore_index=True).to_csv(args.output / "scenario_concentration.csv", index=False)
    pd.concat(gaps, ignore_index=True).to_csv(args.output / "action_gap_summary.csv", index=False)
    pd.concat(selected_rows, ignore_index=True).to_parquet(args.output / "nonfeedback_predictions.parquet", index=False)
    pd.DataFrame(columns=["status", "reason"]).assign(
        status=["unavailable"], reason=["No corrected paired static trajectories supplied."]
    ).to_csv(args.output / "vbs_vs_branch_headroom.csv", index=False)
    print(pd.DataFrame(decomposition).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
