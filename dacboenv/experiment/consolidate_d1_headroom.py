"""Consolidate D1 learned-state H=5/H=10 branches and calibrate DDQN Q rankings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dacboenv.experiment.evaluation_status import atomic_json

BOOTSTRAP_RESAMPLES = 2000
CAPTURE_EPSILON = 1e-8


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _balanced_value(frame: pd.DataFrame, column: str) -> float:
    domain_values = []
    for domain, rows in frame.groupby("domain", sort=True):
        stratum = "dimension" if domain == "bbob" else "scenario"
        domain_values.append(float(rows.groupby(stratum, dropna=False)[column].mean().mean()))
    return float(np.mean(domain_values))


def _paired_bootstrap(frame: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        domain_draws = []
        for domain, domain_rows in frame.groupby("domain", sort=True):
            stratum = "dimension" if domain == "bbob" else "scenario"
            groups = list(domain_rows.groupby(stratum, dropna=False, sort=True))
            sampled_strata = []
            for group_index in rng.integers(len(groups), size=len(groups)):
                _name, stratum_rows = groups[int(group_index)]
                trajectories = list(stratum_rows.groupby(["outer_seed", "task_id", "inner_seed"], sort=True))
                trajectory_draws = []
                for trajectory_index in rng.integers(len(trajectories), size=len(trajectories)):
                    _key, trajectory = trajectories[int(trajectory_index)]
                    phases = trajectory.iloc[rng.integers(len(trajectory), size=len(trajectory))]
                    trajectory_draws.append(float(phases.learned_minus_nonfeedback.mean()))
                sampled_strata.append(float(np.mean(trajectory_draws)))
            domain_draws.append(float(np.mean(sampled_strata)))
        values.append(float(np.mean(domain_draws)))
    return float(np.median(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average tied ranks.
    for value in np.unique(values):
        indices = np.flatnonzero(values == value)
        ranks[indices] = float(np.mean(ranks[indices]))
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank, right_rank = _rankdata(left), _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return float("nan")
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _classification(
    constant_fraction: float,
    within_episode_variation: bool,
    learned: float,
    modal: float,
    marginal: float,
    nonfeedback: float,
    capture: float,
    ci_low: float,
) -> str:
    if constant_fraction >= 0.99:
        return "global_static"
    if all(
        (
            within_episode_variation,
            learned > modal,
            learned > marginal,
            learned > nonfeedback,
            np.isfinite(capture) and capture > 0,
            ci_low > 0,
        )
    ):
        return "feedback_dynamic"
    return "unclassified"


def consolidate(root: Path) -> dict[str, Any]:  # noqa: C901, PLR0915
    root = root.resolve()
    manifest = json.loads((root / "d1_headroom_job_manifest.json").read_text(encoding="utf-8"))
    payloads = []
    for row in manifest["jobs"]:
        path = root / "jobs" / f"{row['job_index']:05d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing D1 headroom job {row['job_index']}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success" or payload.get("job_hash") != row["job_hash"]:
            raise RuntimeError(f"Invalid D1 headroom job {row['job_index']}.")
        if payload.get("all_actions_share_source") is not True:
            raise RuntimeError(f"Nonidentical branch source in job {row['job_index']}.")
        if payload.get("selector_registry_hash") != row["selector_registry_hash"]:
            raise RuntimeError(f"Selector-registry mismatch in job {row['job_index']}.")
        payloads.append(payload)

    snapshot_rows, branch_rows = [], []
    for payload in payloads:
        job, snapshot = payload["job"], payload["snapshot"]
        metadata = {
            **job,
            "snapshot_id": snapshot["snapshot_id"],
            "actual_budget_fraction": float(snapshot["budget_fraction"]),
            "domain": snapshot["domain"],
            "scenario": snapshot["scenario"] or None,
            "dimension": snapshot["dimension"],
            "learned_deterministic_action": int(payload["learned_deterministic_action"]),
            "policy_action_weights_json": json.dumps(payload["policy_action_weights"]),
            "policy_model_scores_json": json.dumps(payload["policy_model_scores"]),
            "policy_model_score_kind": payload["policy_model_score_kind"],
            "observation_json": snapshot["observation_json"],
            "action_history_json": json.dumps(snapshot["action_history"]),
            "initial_design_hash": snapshot["initial_design_hash"],
            "source_snapshot_digest": payload["source_snapshot_digest"],
            "reference_kind": payload["reference_kind"],
        }
        snapshot_rows.append(metadata)
        for branch in payload["branches"]:
            horizon = int(branch["horizon"])
            branch_rows.append(
                {
                    **metadata,
                    "action": int(branch["action"]),
                    "horizon": horizon,
                    "q_value": float(branch["normalized_potential_improvement"]),
                    "raw_regret_improvement": float(branch["regret_improvement"]),
                    "final_incumbent": float(branch["final_incumbent"]),
                    "final_normalized_regret": float(branch["final_regret"]),
                    "reward_sum": float(branch["reward_sum"]),
                    "reward_telescoping_error": float(branch["reward_telescoping_error"]),
                    "nonfeedback_action": int(payload["frozen_nonfeedback_actions"][str(horizon)]),
                }
            )
    snapshots = pd.DataFrame(snapshot_rows)
    branches = pd.DataFrame(branch_rows)
    if branches.duplicated(["job_hash", "action", "horizon"]).any():
        raise RuntimeError("Duplicate learned-state branch cells.")
    _write_parquet(snapshots, root / "learned_snapshots.parquet")
    _write_parquet(branches, root / "learned_branches.parquet")

    comparisons = []
    for run_id, run_rows in branches.groupby("run_id", sort=True):
        actions = snapshots[snapshots.run_id == run_id].learned_deterministic_action.astype(int)
        counts = actions.value_counts().sort_index()
        modal_action = int(counts[counts == counts.max()].index.min())
        marginal = counts.reindex(range(5), fill_value=0).to_numpy(dtype=float)
        marginal /= marginal.sum()
        for (horizon, job_hash), state in run_rows.groupby(["horizon", "job_hash"], sort=True):
            q = state.set_index("action").q_value.reindex(range(5)).to_numpy(dtype=float)
            learned_action = int(state.learned_deterministic_action.iloc[0])
            nonfeedback_action = int(state.nonfeedback_action.iloc[0])
            weights = np.asarray(json.loads(state.policy_action_weights_json.iloc[0]), dtype=float)
            model_scores = np.asarray(json.loads(state.policy_model_scores_json.iloc[0]), dtype=float)
            oracle = float(np.max(q))
            ordered = np.sort(q)
            gap = float(ordered[-1] - ordered[-2])
            true_best = int(np.argmax(q))
            predicted_best = int(np.argmax(model_scores))
            centered_scores = model_scores - np.mean(model_scores)
            centered_q = q - np.mean(q)
            row = state.iloc[0]
            comparisons.append(
                {
                    "run_id": run_id,
                    "outer_seed": int(row.outer_seed),
                    "training_domain": row.training_domain,
                    "algorithm_id": row.algorithm_id,
                    "action_family": row.action_family,
                    "task_id": row.task_id,
                    "inner_seed": int(row.inner_seed),
                    "snapshot_phase": float(row.snapshot_phase),
                    "actual_budget_fraction": float(row.actual_budget_fraction),
                    "domain": row.domain,
                    "scenario": row.scenario,
                    "dimension": row.dimension,
                    "model_sha256": row.model_sha256,
                    "horizon": int(horizon),
                    "job_hash": job_hash,
                    "learned_value": float(q[learned_action]),
                    "policy_weighted_value": float(np.dot(weights, q)),
                    "modal_value": float(q[modal_action]),
                    "marginal_value": float(np.dot(marginal, q)),
                    "nonfeedback_value": float(q[nonfeedback_action]),
                    "oracle_value": oracle,
                    "learned_action": learned_action,
                    "modal_action": modal_action,
                    "nonfeedback_action": nonfeedback_action,
                    "true_best_action": true_best,
                    "predicted_best_action": predicted_best,
                    "top1_correct": bool(q[predicted_best] >= oracle - 1e-12),
                    "top1_top2_gap": gap,
                    "predicted_margin": float(np.sort(model_scores)[-1] - np.sort(model_scores)[-2]),
                    "selected_action_regret": oracle - float(q[learned_action]),
                    "model_score_spearman": _spearman(model_scores, q),
                    "centered_score_mse": float(np.mean((centered_scores - centered_q) ** 2)),
                    "tied_1e3": gap <= 1e-3,
                    "gap_gt_1e3": gap > 1e-3,
                    "gap_gt_1e2": gap > 1e-2,
                }
            )
    values = pd.DataFrame(comparisons)
    values["learned_minus_nonfeedback"] = values.learned_value - values.nonfeedback_value
    denominator = values.oracle_value - values.nonfeedback_value
    values["captured_residual_headroom"] = np.where(
        np.abs(denominator) >= CAPTURE_EPSILON,
        values.learned_minus_nonfeedback / denominator,
        np.nan,
    )
    values.to_csv(root / "policy_comparisons.csv", index=False)

    run_summaries, bootstraps, classifications, q_rows = [], [], [], []
    for (run_id, horizon), group in values.groupby(["run_id", "horizon"], sort=True):
        learned = _balanced_value(group, "learned_value")
        modal = _balanced_value(group, "modal_value")
        marginal_value = _balanced_value(group, "marginal_value")
        nonfeedback = _balanced_value(group, "nonfeedback_value")
        oracle = _balanced_value(group, "oracle_value")
        median, low, high = _paired_bootstrap(group, seed=510000 + int(horizon))
        residual = oracle - nonfeedback
        capture = (learned - nonfeedback) / residual if abs(residual) >= CAPTURE_EPSILON else np.nan
        first = group.iloc[0]
        summary = {
            "run_id": run_id,
            "outer_seed": int(first.outer_seed),
            "training_domain": first.training_domain,
            "algorithm_id": first.algorithm_id,
            "action_family": first.action_family,
            "horizon": int(horizon),
            "learned_value": learned,
            "modal_value": modal,
            "marginal_value": marginal_value,
            "nonfeedback_value": nonfeedback,
            "oracle_value": oracle,
            "learned_minus_nonfeedback": learned - nonfeedback,
            "captured_residual_headroom": capture,
            "mean_selected_action_regret": _balanced_value(group, "selected_action_regret"),
            "mean_top1_top2_gap": _balanced_value(group, "top1_top2_gap"),
            "fraction_tied_1e3": float(group.tied_1e3.mean()),
            "fraction_gap_gt_1e3": float(group.gap_gt_1e3.mean()),
            "fraction_gap_gt_1e2": float(group.gap_gt_1e2.mean()),
        }
        run_summaries.append(summary)
        bootstraps.append(
            {
                "scope": run_id,
                "horizon": int(horizon),
                "metric": "learned_minus_nonfeedback",
                "median": median,
                "ci_low": low,
                "ci_high": high,
                "n_resamples": BOOTSTRAP_RESAMPLES,
            }
        )
        action_rows = snapshots[snapshots.run_id == run_id]
        constant_fraction = float(action_rows.learned_deterministic_action.value_counts(normalize=True).max())
        within_variation = bool(
            (action_rows.groupby(["task_id", "inner_seed"]).learned_deterministic_action.nunique() > 1).any()
        )
        classifications.append(
            {
                "run_id": run_id,
                "horizon": int(horizon),
                "classification": _classification(
                    constant_fraction,
                    within_variation,
                    learned,
                    modal,
                    marginal_value,
                    nonfeedback,
                    capture,
                    low,
                ),
                "constant_action_fraction": constant_fraction,
                "within_episode_variation": within_variation,
                "paired_ci_low": low,
                "paired_ci_high": high,
            }
        )
        q_rows.append(
            {
                "scope": run_id,
                "horizon": int(horizon),
                "top1_accuracy": float(group.top1_correct.mean()),
                "mean_spearman": float(group.model_score_spearman.mean(skipna=True)),
                "mean_selected_action_regret": _balanced_value(group, "selected_action_regret"),
                "mean_predicted_margin": float(group.predicted_margin.mean()),
                "mean_true_gap": float(group.top1_top2_gap.mean()),
                "mean_centered_score_mse": float(group.centered_score_mse.mean()),
            }
        )

    run_summary = pd.DataFrame(run_summaries)
    run_summary.to_csv(root / "learned_headroom_by_run.csv", index=False)
    family_summary = (
        run_summary.groupby(["training_domain", "algorithm_id", "action_family", "horizon"], as_index=False)
        .mean(numeric_only=True)
    )
    family_summary.to_csv(root / "learned_headroom_summary.csv", index=False)

    for (domain, algorithm, horizon), group in values.groupby(
        ["training_domain", "algorithm_id", "horizon"], sort=True
    ):
        median, low, high = _paired_bootstrap(group, seed=610000 + int(horizon))
        bootstraps.append(
            {
                "scope": f"family:{domain}:{algorithm}",
                "horizon": int(horizon),
                "metric": "learned_minus_nonfeedback",
                "median": median,
                "ci_low": low,
                "ci_high": high,
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "outer_seed_count": int(group.outer_seed.nunique()),
            }
        )
        q_rows.append(
            {
                "scope": f"family:{domain}:{algorithm}",
                "horizon": int(horizon),
                "top1_accuracy": float(group.top1_correct.mean()),
                "mean_spearman": float(group.model_score_spearman.mean(skipna=True)),
                "mean_selected_action_regret": _balanced_value(group, "selected_action_regret"),
                "mean_predicted_margin": float(group.predicted_margin.mean()),
                "mean_true_gap": float(group.top1_top2_gap.mean()),
                "mean_centered_score_mse": float(group.centered_score_mse.mean()),
                "outer_seed_count": int(group.outer_seed.nunique()),
            }
        )

    pd.DataFrame(bootstraps).to_csv(root / "bootstrap_summary.csv", index=False)
    pd.DataFrame(classifications).to_csv(root / "behavior_classification.csv", index=False)
    pd.DataFrame(q_rows).to_csv(root / "q_calibration.csv", index=False)
    result = {
        "status": "complete",
        "job_count": len(payloads),
        "snapshot_count": len(snapshots),
        "branch_row_count": len(branches),
        "run_count": int(snapshots.run_id.nunique()),
        "horizons": sorted(int(value) for value in branches.horizon.unique()),
    }
    atomic_json(root / "headroom_consolidation.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
