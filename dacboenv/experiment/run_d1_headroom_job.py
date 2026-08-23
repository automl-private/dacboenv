"""Collect one learned D1 f5 state and branch all WEI actions through H=5/10."""

from __future__ import annotations

import argparse
import json
import traceback
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import DQN, PPO

from dacboenv.experiment.collect_snapshots import bind_snapshot_action_space_factory, collect_context_snapshots
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.evaluation_status import atomic_json
from dacboenv.experiment.headroom_v2 import BBOB_UPPER
from dacboenv.experiment.nonfeedback_registry import load_registry, registry_action
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.snapshot_branch import run_snapshot_branch_diagnostic, snapshot_record_digest
from dacboenv.policy.sb3_model import SB3DiscretePolicy
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider

EARLY_PHASE_LIMIT = 0.375
MIDDLE_PHASE_LIMIT = 0.625


class LearnedPolicy:
    """Lazily load an algorithm-neutral SB3 policy on its exact f5 environment."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.name = f"d1_{row['algorithm_id']}_f5_final"
        self.outer_seed = int(row["outer_seed"])
        self._bridge: SB3DiscretePolicy | None = None
        self.last_action_weights: list[float] | None = None
        self.last_model_scores: list[float] | None = None
        self.last_score_kind: str | None = None

    def _ensure(self, env: Any) -> SB3DiscretePolicy:
        if self._bridge is None:
            if env is None:
                raise RuntimeError("The learned policy has not yet been bound to an environment.")
            self._bridge = SB3DiscretePolicy(
                env,
                model=self.row["model_path"],
                model_class=self.row["algorithm_class"],
                normalization_wrapper=self.row["normalization_path"],
                algorithm_id=self.row["algorithm_id"],
            )
        return self._bridge

    def _model_observation(self, observation: Any) -> Any:
        bridge = self._ensure(None) if self._bridge is not None else None
        assert bridge is not None
        if hasattr(bridge._vec_env, "normalize_obs"):
            return bridge._vec_env.normalize_obs(observation)
        return observation

    def __call__(self, observation: Any, env: Any) -> int:
        bridge = self._ensure(env)
        model_observation = observation
        if hasattr(bridge._vec_env, "normalize_obs"):
            model_observation = bridge._vec_env.normalize_obs(observation)
        model = bridge._model
        with torch.no_grad():
            tensor, _vectorized = model.policy.obs_to_tensor(model_observation)
            if isinstance(model, DQN):
                scores = model.q_net(tensor).detach().cpu().numpy().reshape(-1)
                action = int(np.argmax(scores))
                weights = np.zeros(scores.shape[0], dtype=float)
                weights[action] = 1.0
                self.last_model_scores = [float(value) for value in scores]
                self.last_action_weights = weights.tolist()
                self.last_score_kind = "q_values"
                return action
            if isinstance(model, PPO):
                distribution = model.policy.get_distribution(tensor)
                probabilities = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
                action = int(np.argmax(probabilities))
                self.last_model_scores = [float(value) for value in probabilities]
                self.last_action_weights = [float(value) for value in probabilities]
                self.last_score_kind = "action_probabilities"
                return action
        raise TypeError(f"Unsupported learned model class: {type(model).__name__}")

    def action_at_snapshot(self, observation_json: str) -> tuple[int, list[float], list[float], str]:
        payload = json.loads(observation_json)
        observation = {
            key: np.asarray(value["values"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
            for key, value in payload.items()
        }
        action = self(observation, None)
        assert self.last_action_weights is not None
        assert self.last_model_scores is not None
        assert self.last_score_kind is not None
        return action, self.last_action_weights, self.last_model_scores, self.last_score_kind


def _selector_metadata(row: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    phase = float(row["snapshot_phase"])
    phase_three = "early" if phase <= EARLY_PHASE_LIMIT else "middle" if phase <= MIDDLE_PHASE_LIMIT else "late"
    phase_five = min(int(phase * 5), 4)
    if snapshot.domain == "yahpo":
        context = f"yahpo:{snapshot.scenario}"
        privileged = context
    else:
        context = f"bbob:d{int(snapshot.dimension)}"
        function = int(snapshot.task_id.split("/")[2])
        group = next(index for index, upper in enumerate(BBOB_UPPER) if function <= upper)
        privileged = f"{context}:g{group}"
    return {
        "domain": snapshot.domain,
        "context_dimension": context,
        "context_privileged": privileged,
        "phase_three": phase_three,
        "phase_five": phase_five,
    }


def run_job(row: dict[str, Any], output: Path, reference_table: Path, registry_path: Path) -> dict[str, Any]:
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job_hash") == row["job_hash"]:
            return existing
        raise RuntimeError(f"Partial/corrupt output exists: {output}")
    if file_sha256(Path(row["model_path"])) != row["model_sha256"]:
        raise RuntimeError("Model hash changed after D1 headroom preparation.")
    if row["normalization_path"] is not None and file_sha256(Path(row["normalization_path"])) != row["normalization_sha256"]:
        raise RuntimeError("Normalization hash changed after D1 headroom preparation.")
    registry = load_registry(registry_path)
    if registry["registry_hash"] != row["selector_registry_hash"]:
        raise RuntimeError("Frozen nonfeedback registry hash mismatch.")
    task_id = str(row["task_id"])
    if task_id.startswith("yahpo/"):
        provider: Any = ManifestReferenceProvider(
            reference_table,
            expected_runtime_objective_transform="negative_accuracy",
            expected_reporting_objective_transform="one_minus_accuracy",
            expected_fidelity="fixed_maximum",
        )
    else:
        provider = BBOBExactReferenceProvider()

    def factory(task: str, seed: int, family: str) -> Any:
        return real_structured_mixed_env(
            task,
            seed,
            family,
            context_split="validation",
            reference_table=reference_table if task.startswith("yahpo/") else None,
            interaction_frequency=5,
        )

    policy = LearnedPolicy(row)
    snapshots = collect_context_snapshots(
        task_id=task_id,
        inner_seed=int(row["inner_seed"]),
        env_factory=factory,
        policy=policy,
        budget_fractions=[float(row["snapshot_phase"])],
        action_space_name="wei",
        source_manifest=str(row["manifest_id"]),
        source_manifest_hash=str(row["manifest_hash"]),
        reference_provider=provider,
    )
    snapshot = snapshots[0]
    learned_action, action_weights, model_scores, score_kind = policy.action_at_snapshot(snapshot.observation_json)
    bound = bind_snapshot_action_space_factory(snapshots, factory)
    horizons = tuple(int(value) for value in row["branch_horizons"])
    report = run_snapshot_branch_diagnostic(
        snapshots,
        bound,
        lambda _task: float(snapshot.reference_value),
        forbidden_task_ids=set(),
        horizons=horizons,
    )
    branches = [asdict(branch) for branch in report.branches]
    source_digest = snapshot_record_digest(snapshot)
    source_digests = {snapshot_record_digest(branch.snapshot) for branch in report.branches}
    if source_digests != {source_digest}:
        raise RuntimeError("Action branches did not share one identical source state.")
    metadata = _selector_metadata(row, snapshot)
    nonfeedback = {
        str(horizon): registry_action(registry, "wei", horizon, metadata) for horizon in horizons
    }
    payload = {
        "schema_version": "d1-f5-learned-headroom-job-v1",
        "status": "success",
        "job_hash": row["job_hash"],
        "job": row,
        "snapshot": asdict(snapshot),
        "source_snapshot_digest": source_digest,
        "learned_deterministic_action": learned_action,
        "policy_action_weights": action_weights,
        "policy_model_scores": model_scores,
        "policy_model_score_kind": score_kind,
        "frozen_nonfeedback_actions": nonfeedback,
        "branches": branches,
        "summaries": [asdict(summary) for summary in report.summaries],
        "all_actions_share_source": True,
        "reference_kind": snapshot.reference_kind,
        "selector_registry_hash": registry["registry_hash"],
    }
    atomic_json(output, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = manifest["jobs"][args.job_index]
    output = args.output_root / f"{args.job_index:05d}.json"
    failure = args.output_root / f"{args.job_index:05d}.failed.json"
    args.output_root.mkdir(parents=True, exist_ok=True)
    registry_path = args.manifest.resolve().parent / "nonfeedback_selector_registry.json"
    try:
        result = run_job(row, output, args.reference_table.resolve(), registry_path)
    except Exception as error:
        atomic_json(
            failure,
            {
                "status": "failed",
                "job_index": args.job_index,
                "job_hash": row["job_hash"],
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    print(json.dumps({"job_index": args.job_index, "status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
