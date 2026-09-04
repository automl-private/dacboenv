"""Run one CARP-S/SMAC random-history same-state branch job."""

from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.collect_snapshots import (
    StaticSnapshotPolicy,
    UniformRandomSnapshotPolicy,
    bind_snapshot_action_space_factory,
    collect_context_snapshots,
)
from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.snapshot_branch import (
    require_deterministic_replay_process_environment,
    run_snapshot_branch_diagnostic,
    snapshot_record_digest,
)
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider

SCENARIOS = ("bbob", "lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost")
TIE_TOLERANCE = 1e-3
TELESCOPING_TOLERANCE = 1e-10


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _random_seed(task_id: str, seed: int) -> int:
    digest = canonical_sha256(
        {"stream": "offline_training_policy", "task_id": task_id, "seed": seed, "policy_id": "uniform_random_f5"}
    )
    return int(digest[:16], 16)


def _policy(row: dict[str, Any]) -> Any:
    source = str(row["source_policy"])
    if source == "uniform_random":
        return UniformRandomSnapshotPolicy(_random_seed(str(row["task_id"]), int(row["seed"])))
    action = {"static_0": 0, "static_2": 2, "static_4": 4}.get(source)
    if action is None:
        raise ValueError(f"Unsupported source policy {source!r}.")
    return StaticSnapshotPolicy(action)


def _observation(snapshot: Any) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(snapshot.observation_json)
    values = {}
    for key in ("global_state", "action_features"):
        item = payload[key]
        values[key] = np.asarray(item["values"], dtype=np.dtype(item["dtype"])).reshape(item["shape"])
    return values["global_state"].astype(np.float32), values["action_features"].astype(np.float32)


def _duplicate_groups(features: np.ndarray) -> list[int]:
    groups: dict[bytes, int] = {}
    result = []
    for row in np.asarray(features[:, 1:], dtype=np.float32):
        key = row.tobytes(order="C")
        groups.setdefault(key, len(groups))
        result.append(groups[key])
    return result


def run(config: DictConfig) -> dict[str, Any]:
    """Replay one random history and branch all actions once through H=10."""
    require_deterministic_replay_process_environment()
    manifest_path = Path(str(config.manifest)).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = int(config.job_index)
    row = manifest["jobs"][index]
    if int(row["job_index"]) != index:
        raise ValueError("Branch manifest job indices are not positional.")
    output = Path(row["output_path"])
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job_hash") == row["job_hash"]:
            return cast("dict[str, Any]", existing)
        raise RuntimeError(f"Refusing partial/conflicting branch shard {output}.")
    reference_table = Path(str(config.reference_table)).resolve()
    task_id = str(row["task_id"])
    data_context_split = str(row["context_split"])
    if data_context_split not in {"train", "dev"}:
        raise ValueError(
            "Offline branch rows must use data context split 'train' or 'dev', "
            f"got {data_context_split!r}."
        )
    environment_context_split = (
        "train" if data_context_split == "train" else "validation"
    )

    def factory(task: str, seed: int, family: str) -> Any:
        return real_structured_mixed_env(
            task,
            seed,
            family,
            context_split=environment_context_split,
            reference_table=reference_table if task.startswith("yahpo/") else None,
            interaction_frequency=5,
        )

    provider: Any
    if task_id.startswith("yahpo/"):
        provider = ManifestReferenceProvider(
            reference_table,
            expected_runtime_objective_transform="negative_accuracy",
            expected_reporting_objective_transform="one_minus_accuracy",
            expected_fidelity="fixed_maximum",
        )
    else:
        provider = BBOBExactReferenceProvider()
    snapshots = collect_context_snapshots(
        task_id=task_id,
        inner_seed=int(row["seed"]),
        env_factory=factory,
        policy=_policy(row),
        budget_fractions=[float(row["phase"])],
        action_space_name="wei",
        source_manifest="offline_final_v3",
        source_manifest_hash=str(row["final_manifest_hash"]),
        reference_provider=provider,
    )
    snapshot = snapshots[0]
    if snapshot.reference_value is None:
        raise ValueError("Offline branch snapshot lacks required exact/best-known reference metadata.")
    reference_value = float(snapshot.reference_value)
    bound = bind_snapshot_action_space_factory(snapshots, factory)
    report = run_snapshot_branch_diagnostic(
        snapshots,
        bound,
        lambda _task: reference_value,
        forbidden_task_ids=set(),
        horizons=(5, 10),
        tie_tolerance=1e-3,
    )
    source_digest = snapshot_record_digest(snapshot)
    if {snapshot_record_digest(branch.snapshot) for branch in report.branches} != {source_digest}:
        raise RuntimeError("All action branches must share one source state.")
    cells = {(branch.horizon, branch.action): branch for branch in report.branches}
    q5 = np.asarray([cells[5, action].normalized_potential_improvement for action in range(5)], dtype=float)
    q10 = np.asarray([cells[10, action].normalized_potential_improvement for action in range(5)], dtype=float)
    if len(cells) != 10:  # noqa: PLR2004
        raise RuntimeError("Expected exactly five actions at each H=5/H=10 prefix.")
    errors = np.asarray([abs(branch.reward_telescoping_error) for branch in report.branches])
    global_state, action_features = _observation(snapshot)
    scenario = snapshot.scenario or "bbob"
    scenario_id = SCENARIOS.index(scenario)
    phase_bin = min(int(float(snapshot.budget_fraction or row["phase"]) * 4), 3)
    record = {
        "global_state": global_state.tolist(),
        "action_features": action_features.tolist(),
        "q5": q5.tolist(),
        "q10": q10.tolist(),
        "valid_action_mask": [True] * 5,
        "tie_mask_q5": ((q5.max() - q5) <= TIE_TOLERANCE).tolist(),
        "tie_mask_q10": ((q10.max() - q10) <= TIE_TOLERANCE).tolist(),
        "top1_top2_gap_q5": float(np.sort(q5)[-1] - np.sort(q5)[-2]),
        "top1_top2_gap_q10": float(np.sort(q10)[-1] - np.sort(q10)[-2]),
        "task_id": task_id,
        "domain_id": 1 if task_id.startswith("yahpo/") else 0,
        "scenario_id": scenario_id,
        "phase_bin": phase_bin,
        "seed": int(row["seed"]),
        "data_context_split": data_context_split,
        "environment_context_split": environment_context_split,
        "source_policy_id": str(row["source_policy"]),
        "source_state_digest": source_digest,
        "source_replay_digest": source_digest,
        "candidate_duplicate_groups": json.dumps(_duplicate_groups(action_features), separators=(",", ":")),
        "reference_metadata": {
            "kind": snapshot.reference_kind,
            "source": snapshot.reference_source,
            "source_hash": snapshot.reference_source_hash,
        },
        "branch_protocol_hash": canonical_sha256(
            {"horizons": [5, 10], "actions": [0, 1, 2, 3, 4], "interaction_frequency": 5}
        ),
    }
    payload: dict[str, Any] = {
        "schema_version": "dacbo-offline-branch-job-v1",
        "status": "success",
        "job_hash": row["job_hash"],
        "job": row,
        "snapshot": asdict(snapshot),
        "branch_record": record,
        "branches": [asdict(branch) for branch in report.branches],
        "all_actions_share_source": True,
        "reward_telescoping_valid": bool(np.all(errors <= TELESCOPING_TOLERANCE)),
        "maximum_telescoping_error": float(errors.max(initial=0.0)),
        "single_h10_branch_with_h5_prefix": True,
    }
    _atomic_json(output, payload)
    return payload


def _config_from_cli() -> DictConfig:
    """Parse the leaf-worker dotlist without initializing Hydra globally.

    This worker calls real environment factories that compose a separate
    DACBO training template. Running the worker itself under ``@hydra.main``
    initializes ``GlobalHydra`` first and makes that nested composition fail.
    The Slurm worker already supplies the complete immutable row interface as
    ``key=value`` arguments, so OmegaConf dotlist parsing is sufficient here.
    """
    config = OmegaConf.from_cli()
    required = ("manifest", "job_index", "reference_table")
    missing = [key for key in required if OmegaConf.select(config, key, default=None) is None]
    if missing:
        raise ValueError(f"Offline branch worker is missing required CLI keys: {missing}.")
    return config


def main() -> None:
    """Run one manifest row and persist compact failure metadata on errors."""
    config = _config_from_cli()
    manifest = json.loads(Path(str(config.manifest)).read_text(encoding="utf-8"))
    row = manifest["jobs"][int(config.job_index)]
    try:
        result = run(config)
    except Exception as error:
        output = Path(row["output_path"]).with_suffix(".failed.json")
        _atomic_json(
            output,
            {
                "status": "failed",
                "job_hash": row["job_hash"],
                "job_index": row["job_index"],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps({"status": result["status"], "job_index": row["job_index"]}, sort_keys=True))


if __name__ == "__main__":
    main()
