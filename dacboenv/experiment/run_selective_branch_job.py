"""Run one base/selective-history same-state branch job through CARP-S."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np

from dacboenv.env.observation import GLOBAL_STATE_INDEX
from dacboenv.experiment.collect_snapshots import bind_snapshot_action_space_factory, collect_context_snapshots
from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.snapshot_branch import (
    require_deterministic_replay_process_environment,
    run_snapshot_branch_diagnostic,
    snapshot_record_digest,
)
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.policy.initial_base import InitialConditionedWEIPolicy
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider
from dacboenv.selective_wei.artifacts import atomic_json, load_policy_bundle
from dacboenv.selective_wei.base_selector import load_base_selector
from dacboenv.selective_wei.context import context_from_task_id
from dacboenv.selective_wei.initial_base_model import load_initial_base
from dacboenv.selective_wei.policy import SelectiveWEIPolicy
from dacboenv.selective_wei.terminal_intervention import terminal_interventions
from dacboenv.selective_wei.uncertainty import candidate_equivalence

if TYPE_CHECKING:
    from omegaconf import DictConfig

SCENARIOS = ("bbob", "lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost")
TIE_TOLERANCE = 1e-3
TELESCOPING_TOLERANCE = 1e-10


class _BaseHistoryPolicy:
    name = "contextual_base"
    outer_seed: int | None = None

    def __init__(self, registry: str, registry_hash: str) -> None:
        self.selector = load_base_selector(Path(registry), expected_hash=registry_hash)

    def __call__(self, observation: Any, env: Any) -> int:
        phase = None
        if self.selector.registry.kind == "context_phase":
            fraction = float(observation["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]])
            phase = min(max(int(fraction * 4), 0), 3)
        return self.selector.select(context_from_task_id(env.current_task_id, phase_bin=phase)).action


class _SelectiveHistoryPolicy:
    name = "selective_wei"
    outer_seed: int | None = None

    def __init__(self, bundle_path: str, bundle_hash: str, decision_log: Path) -> None:
        self.bundle_path = bundle_path
        self.bundle_hash = bundle_hash
        self.decision_log = decision_log
        self.policy: SelectiveWEIPolicy | None = None

    def __call__(self, observation: Any, env: Any) -> int:
        if self.policy is None:
            self.policy = SelectiveWEIPolicy(
                env,
                self.bundle_path,
                self.bundle_hash,
                decision_log=str(self.decision_log),
            )
        return self.policy(observation)


class _InitialBaseHistoryPolicy:
    """Collect from one once-selected full-static terminal base per episode."""

    name = "initial_conditioned_base"
    outer_seed: int | None = None

    def __init__(self, path: str) -> None:
        self.path = path
        self.policy: InitialConditionedWEIPolicy | None = None

    def __call__(self, observation: Any, env: Any) -> int:
        if self.policy is None:
            self.policy = InitialConditionedWEIPolicy(env, self.path)
        return self.policy(observation)

    def initialize(self, observation: Any, env: Any) -> None:
        """Freeze a0 even when the first requested snapshot lies inside D0."""
        self(observation, env)


def _observation(snapshot: Any) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(snapshot.observation_json)
    values = {}
    for key in ("global_state", "action_features"):
        item = payload[key]
        values[key] = np.asarray(item["values"], dtype=np.dtype(item["dtype"])).reshape(item["shape"])
    return values["global_state"].astype(np.float32), values["action_features"].astype(np.float32)


def _duplicate_groups(features: np.ndarray, tolerance: float = 1e-6) -> list[int]:
    return list(candidate_equivalence(features, tolerance=tolerance).groups)


def run(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915 - legacy and anchored campaign dispatch
    """Collect one source state and branch five actions at H5/H10."""
    require_deterministic_replay_process_environment()
    manifest = json.loads(Path(str(config.manifest)).read_text(encoding="utf-8"))
    row = manifest["jobs"][int(config.job_index)]
    if row.get("source_revision") and row["source_revision"] != current_source_revision():
        raise ValueError("Queued anchored branch source changed after preparation.")
    output = Path(row["output_path"])
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if row.get("initial_base_bundle"):
            digest = payload.pop("payload_hash")
            if canonical_sha256(payload) != digest:
                raise ValueError("Anchored branch output is corrupt.")
            payload["payload_hash"] = digest
        if payload.get("status") == "success" and payload.get("job_hash") == row["job_hash"]:
            return cast("dict[str, Any]", payload)
        raise RuntimeError(f"Refusing conflicting selective branch shard {output}.")
    task_id = str(row["task_id"])
    reference_table = Path(str(config.reference_table)).resolve()

    def factory(task: str, seed: int, family: str) -> Any:
        return real_structured_mixed_env(
            task,
            seed,
            family,
            context_split=str(row["environment_context_split"]),
            reference_table=reference_table if task.startswith("yahpo/") else None,
            interaction_frequency=5,
            initial_context_settings=row.get("initial_feature_settings"),
        )

    if row["campaign"] == "base":
        if row.get("initial_base_bundle"):
            initial_model = load_initial_base(Path(row["initial_base_bundle"]))
            if initial_model.semantic_hash != row["initial_base_semantic_hash"]:
                raise ValueError("Initial-base branch source artifact changed.")
            policy: Any = _InitialBaseHistoryPolicy(row["initial_base_bundle"])
        else:
            policy = _BaseHistoryPolicy(row["base_selector_registry"], row["base_selector_registry_hash"])
    else:
        bundle = load_policy_bundle(Path(row["policy_bundle"]), expected_hash=row["policy_artifact_hash"])
        policy = _SelectiveHistoryPolicy(
            row["policy_bundle"], bundle["policy_artifact_hash"], output.with_suffix(".decisions.jsonl")
        )
    provider: Any = (
        ManifestReferenceProvider(
            reference_table,
            expected_runtime_objective_transform="negative_accuracy",
            expected_reporting_objective_transform="one_minus_accuracy",
            expected_fidelity="fixed_maximum",
        )
        if task_id.startswith("yahpo/")
        else BBOBExactReferenceProvider()
    )
    snapshots = collect_context_snapshots(
        task_id=task_id,
        inner_seed=int(row["seed"]),
        env_factory=factory,
        policy=policy,
        budget_fractions=[float(row["phase"])],
        action_space_name="wei",
        source_manifest=f"selective_{row['campaign']}_branch_v1",
        source_manifest_hash=manifest["manifest_hash"],
        reference_provider=provider,
    )
    snapshot = snapshots[0]
    if row.get("initial_base_bundle") and not snapshot.initial_anchor_json:
        raise ValueError("Initial-base branch source is missing its immutable anchor.")
    if snapshot.reference_value is None:
        raise ValueError("Selective branch snapshot lacks reference metadata.")
    if row.get("target_semantics") == "override_then_base_terminal_advantage":
        result = terminal_interventions(
            snapshot, lambda task, seed: factory(task, seed, "wei"), horizon=int(row["intervention_horizon"])
        )
        payload = {
            "schema_version": "dacbo-terminal-intervention-job-v1",
            "status": "success",
            "job_hash": row["job_hash"],
            "job": row,
            "snapshot": asdict(snapshot),
            "terminal_intervention": result,
        }
        payload["payload_hash"] = canonical_sha256(payload)
        atomic_json(output, payload)
        return payload
    reference_value = float(snapshot.reference_value)
    report = run_snapshot_branch_diagnostic(
        snapshots,
        bind_snapshot_action_space_factory(snapshots, factory),
        lambda _task: reference_value,
        forbidden_task_ids=set(),
        horizons=(5, 10),
        tie_tolerance=TIE_TOLERANCE,
    )
    source_digest = snapshot_record_digest(snapshot)
    if {snapshot_record_digest(branch.snapshot) for branch in report.branches} != {source_digest}:
        raise RuntimeError("Selective action branches do not share one source fingerprint.")
    cells = {(branch.horizon, branch.action): branch for branch in report.branches}
    q5 = np.asarray([cells[5, action].normalized_potential_improvement for action in range(5)])
    q10 = np.asarray([cells[10, action].normalized_potential_improvement for action in range(5)])
    global_state, action_features = _observation(snapshot)
    scenario = snapshot.scenario or "bbob"
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
        "domain_id": int(task_id.startswith("yahpo/")),
        "scenario_id": SCENARIOS.index(scenario),
        "phase_bin": min(
            int(float(row["phase"] if snapshot.budget_fraction is None else snapshot.budget_fraction) * 4), 3
        ),
        "seed": int(row["seed"]),
        "source_policy_id": str(row["campaign"]),
        "data_context_split": row["data_context_split"],
        "environment_context_split": row["environment_context_split"],
        "source_state_digest": source_digest,
        "source_replay_digest": source_digest,
        "candidate_duplicate_groups": json.dumps(_duplicate_groups(action_features), separators=(",", ":")),
        "reference_metadata": {
            "kind": snapshot.reference_kind,
            "source": snapshot.reference_source,
            "source_hash": snapshot.reference_source_hash,
        },
        "branch_protocol_hash": canonical_sha256(
            {
                "protocol": row["protocol"],
                "campaign": row["campaign"],
                "horizons": [5, 10],
                "interaction_frequency": 5,
            }
        ),
    }
    if snapshot.initial_anchor_json:
        anchor = json.loads(snapshot.initial_anchor_json)
        record.update(
            initial_anchor_json=snapshot.initial_anchor_json,
            initial_base_action=anchor["state"]["decision"]["action"],
            initial_base_semantic_hash=anchor["state"]["decision"]["base_semantic_hash"],
            target_semantics="fixed_action_local_gain",
        )
    payload = {
        "schema_version": "dacbo-selective-branch-job-v1",
        "status": "success",
        "job_hash": row["job_hash"],
        "job": row,
        "snapshot": asdict(snapshot),
        "branch_record": record,
        "branches": [asdict(branch) for branch in report.branches],
        "all_actions_share_source": True,
        "reward_telescoping_valid": all(
            abs(branch.reward_telescoping_error) <= TELESCOPING_TOLERANCE for branch in report.branches
        ),
    }
    payload["payload_hash"] = canonical_sha256(payload)
    atomic_json(output, payload)
    return payload


@hydra.main(version_base=None, config_path="../configs", config_name="selective_branch_worker")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Run one manifest row and write compact failure metadata."""
    try:
        result = run(config)
    except Exception as error:
        manifest = json.loads(Path(str(config.manifest)).read_text(encoding="utf-8"))
        row = manifest["jobs"][int(config.job_index)]
        atomic_json(
            Path(row["output_path"]).with_suffix(".failed.json"),
            {
                "status": "failed",
                "job_hash": row["job_hash"],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps({"status": result["status"], "job_index": config.job_index}, sort_keys=True))


if __name__ == "__main__":
    main()
