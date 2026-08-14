"""Validate and consolidate completed Stage-B learned-state branch jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dacboenv.experiment.augment_headroom_with_learned_policies import classify_behavior
from dacboenv.experiment.evaluation_status import atomic_json

BOOTSTRAP_RESAMPLES = 2000
CAPTURE_EPSILON = 1e-8
GAP_1E3 = 1e-3
GAP_1E2 = 1e-2


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _balanced_value(frame: pd.DataFrame, column: str) -> float:
    domain_values = []
    for domain, domain_rows in frame.groupby("domain", sort=True):
        stratum = "dimension" if domain == "bbob" else "scenario"
        domain_values.append(float(domain_rows.groupby(stratum, dropna=False)[column].mean().mean()))
    return float(np.mean(domain_values))


def _paired_bootstrap(frame: pd.DataFrame, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_domains = []
        for domain, domain_rows in frame.groupby("domain", sort=True):
            stratum = "dimension" if domain == "bbob" else "scenario"
            sampled_strata = []
            groups = list(domain_rows.groupby(stratum, dropna=False, sort=True))
            for index in rng.integers(len(groups), size=len(groups)):
                _name, stratum_rows = groups[int(index)]
                trajectory_keys = ["outer_ppo_seed", "task_id", "inner_seed"]
                trajectories = list(stratum_rows.groupby(trajectory_keys, sort=True))
                draws = []
                for trajectory_index in rng.integers(len(trajectories), size=len(trajectories)):
                    _key, trajectory = trajectories[int(trajectory_index)]
                    phases = trajectory.iloc[rng.integers(len(trajectory), size=len(trajectory))]
                    draws.append(float(phases.learned_minus_nonfeedback.mean()))
                sampled_strata.append(float(np.mean(draws)))
            sampled_domains.append(float(np.mean(sampled_strata)))
        values.append(float(np.mean(sampled_domains)))
    return float(np.median(values)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def consolidate(followup_root: Path) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Reject incomplete/conflicting jobs and compute frozen comparisons."""
    root = followup_root.resolve()
    manifest = json.loads((root / "headroom_job_manifest.json").read_text(encoding="utf-8"))
    jobs = []
    for row in manifest["jobs"]:
        path = root / "headroom_jobs" / f"{row['job_index']:05d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing learned-headroom job {row['job_index']}: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "success" or payload.get("job_hash") != row["job_hash"]:
            raise RuntimeError(f"Invalid/corrupt learned-headroom job {row['job_index']}.")
        if payload.get("selector_registry_hash") != row["selector_registry_hash"]:
            raise RuntimeError(f"Selector-registry hash mismatch in job {row['job_index']}.")
        if payload.get("all_actions_share_source") is not True:
            raise RuntimeError(f"Branch source mismatch in job {row['job_index']}.")
        jobs.append(payload)

    snapshot_rows, branch_rows = [], []
    for payload in jobs:
        job, snapshot = payload["job"], payload["snapshot"]
        metadata = {
            **job,
            "snapshot_id": snapshot["snapshot_id"],
            "domain": snapshot["domain"],
            "scenario": snapshot["scenario"] or None,
            "dimension": snapshot["dimension"],
            "learned_deterministic_action": payload["learned_deterministic_action"],
            "policy_action_probabilities_json": json.dumps(payload["policy_action_probabilities"]),
            "observation_json": snapshot["observation_json"],
            "action_history_json": json.dumps(snapshot["action_history"]),
            "initial_design_hash": snapshot["initial_design_hash"],
            "source_snapshot_digest": payload["source_snapshot_digest"],
            "reference_kind": payload["reference_kind"],
        }
        snapshot_rows.append(metadata)
        for branch in payload["branches"]:
            branch_rows.append(
                {
                    **metadata,
                    "action": int(branch["action"]),
                    "horizon": int(branch["horizon"]),
                    "q_value": float(branch["normalized_potential_improvement"]),
                    "raw_regret_improvement": float(branch["regret_improvement"]),
                    "final_incumbent": float(branch["final_incumbent"]),
                    "final_normalized_regret": float(branch["final_regret"]),
                    "reward_sum": float(branch["reward_sum"]),
                    "reward_telescoping_error": float(branch["reward_telescoping_error"]),
                    "nonfeedback_action": int(payload["frozen_nonfeedback_actions"][str(branch["horizon"])]),
                }
            )
    snapshots, branches = pd.DataFrame(snapshot_rows), pd.DataFrame(branch_rows)
    if branches.duplicated(["job_hash", "action", "horizon"]).any():
        raise RuntimeError("Duplicate learned-state branch cells detected.")
    root.mkdir(parents=True, exist_ok=True)
    _write_parquet(snapshots, root / "learned_snapshots.parquet")
    _write_parquet(branches, root / "learned_branches.parquet")

    comparisons = []
    for run_id, run in branches.groupby("run_id", sort=True):
        deterministic_actions = snapshots[snapshots.run_id == run_id].learned_deterministic_action.astype(int)
        counts = deterministic_actions.value_counts().sort_index()
        modal_action = int(counts[counts == counts.max()].index.min())
        marginal = counts.reindex(range(5), fill_value=0).to_numpy(dtype=float)
        marginal /= marginal.sum()
        for horizon, rows in run.groupby("horizon", sort=True):
            for job_hash, state in rows.groupby("job_hash", sort=True):
                q = state.set_index("action").q_value
                learned_action = int(state.learned_deterministic_action.iloc[0])
                nonfeedback_action = int(state.nonfeedback_action.iloc[0])
                ordered = np.sort(q.to_numpy())
                oracle = float(ordered[-1])
                gap = float(ordered[-1] - ordered[-2])
                comparisons.append(
                    {
                        **state.iloc[0][
                            [
                                "run_id",
                                "outer_ppo_seed",
                                "training_domain",
                                "action_family",
                                "task_id",
                                "inner_seed",
                                "snapshot_phase",
                                "domain",
                                "scenario",
                                "dimension",
                                "model_sha256",
                            ]
                        ].to_dict(),
                        "horizon": int(horizon),
                        "job_hash": job_hash,
                        "learned_value": float(q.loc[learned_action]),
                        "stochastic_value": float(
                            np.dot(
                                np.asarray(json.loads(state.policy_action_probabilities_json.iloc[0])),
                                q.reindex(range(5)),
                            )
                        ),
                        "modal_value": float(q.loc[modal_action]),
                        "marginal_value": float(np.dot(marginal, q.reindex(range(5)))),
                        "nonfeedback_value": float(q.loc[nonfeedback_action]),
                        "oracle_value": oracle,
                        "learned_action": learned_action,
                        "modal_action": modal_action,
                        "nonfeedback_action": nonfeedback_action,
                        "top1_top2_gap": gap,
                        "tied_1e3": gap <= GAP_1E3,
                        "gap_gt_1e3": gap > GAP_1E3,
                        "gap_gt_1e2": gap > GAP_1E2,
                    }
                )
    values = pd.DataFrame(comparisons)
    values["learned_minus_nonfeedback"] = values.learned_value - values.nonfeedback_value
    values["learned_selected_action_regret"] = values.oracle_value - values.learned_value
    denominator = values.oracle_value - values.nonfeedback_value
    values["captured_residual_headroom"] = np.where(
        np.abs(denominator) >= CAPTURE_EPSILON, values.learned_minus_nonfeedback / denominator, np.nan
    )
    values.to_csv(root / "policy_comparisons.csv", index=False)

    summaries, bootstraps, classifications = [], [], []
    for (run_id, horizon), group in values.groupby(["run_id", "horizon"], sort=True):
        learned = _balanced_value(group, "learned_value")
        nonfeedback = _balanced_value(group, "nonfeedback_value")
        oracle = _balanced_value(group, "oracle_value")
        modal = _balanced_value(group, "modal_value")
        marginal = _balanced_value(group, "marginal_value")
        median, low, high = _paired_bootstrap(group, seed=904211 + int(horizon))
        denominator_value = oracle - nonfeedback
        capture = (learned - nonfeedback) / denominator_value if abs(denominator_value) >= CAPTURE_EPSILON else np.nan
        summaries.append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "learned_value": learned,
                "stochastic_value": _balanced_value(group, "stochastic_value"),
                "modal_value": modal,
                "marginal_value": marginal,
                "nonfeedback_value": nonfeedback,
                "oracle_value": oracle,
                "learned_minus_nonfeedback": learned - nonfeedback,
                "captured_residual_headroom": capture,
                "mean_selected_action_regret": _balanced_value(group, "learned_selected_action_regret"),
                "mean_top1_top2_gap": _balanced_value(group, "top1_top2_gap"),
                "fraction_tied_1e3": float(group.tied_1e3.mean()),
                "fraction_gap_gt_1e3": float(group.gap_gt_1e3.mean()),
                "fraction_gap_gt_1e2": float(group.gap_gt_1e2.mean()),
            }
        )
        bootstraps.append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "metric": "learned_minus_nonfeedback",
                "median": median,
                "ci_low": low,
                "ci_high": high,
                "n_resamples": BOOTSTRAP_RESAMPLES,
            }
        )
        action_rows = snapshots[snapshots.run_id == run_id]
        constant_fraction = float(action_rows.learned_deterministic_action.value_counts(normalize=True).max())
        episode_unique = action_rows.groupby(["task_id", "inner_seed"]).learned_deterministic_action.nunique()
        within_variation = bool((episode_unique > 1).any())
        label = classify_behavior(
            constant_fraction=constant_fraction,
            within_episode_variation=within_variation,
            contextual_dependence=False,
            phase_explains_actions=False,
            evolving_state_increment=0.0,
            beats_modal=learned > modal,
            beats_marginal=learned > marginal,
            beats_nonfeedback=learned > nonfeedback,
            captures_positive_residual=bool(np.isfinite(capture) and capture > 0),
            paired_ci_excludes_zero=low > 0,
        )
        classifications.append(
            {
                "run_id": run_id,
                "horizon": horizon,
                "classification": label,
                "constant_action_fraction": constant_fraction,
                "within_episode_variation": within_variation,
                "paired_ci_low": low,
                "paired_ci_high": high,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(root / "learned_headroom_by_run.csv", index=False)
    family = summary.groupby(["training_domain", "action_family", "horizon"], as_index=False).mean(numeric_only=True)
    family.to_csv(root / "learned_headroom_summary.csv", index=False)
    for (domain, family_name, horizon), group in values.groupby(
        ["training_domain", "action_family", "horizon"], sort=True
    ):
        median, low, high = _paired_bootstrap(group, seed=194211 + int(horizon))
        bootstraps.append(
            {
                "run_id": f"family_mean:{domain}:{family_name}",
                "horizon": horizon,
                "metric": "learned_minus_nonfeedback",
                "median": median,
                "ci_low": low,
                "ci_high": high,
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "outer_seed_count": int(group.outer_ppo_seed.nunique()),
            }
        )
    pd.DataFrame(classifications).to_csv(root / "behavior_classification.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(root / "bootstrap_summary.csv", index=False)
    result = {
        "status": "complete",
        "job_count": len(jobs),
        "snapshot_count": len(snapshots),
        "branch_rows": len(branches),
        "run_count": snapshots.run_id.nunique(),
    }
    atomic_json(root / "headroom_consolidation.json", result)
    return result


__all__ = ["consolidate"]
