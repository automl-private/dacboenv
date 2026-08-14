"""Collect and branch one learned-policy Stage-B state atomically."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dacboenv.experiment.collect_snapshots import bind_snapshot_action_space_factory, collect_context_snapshots
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.evaluation_status import atomic_json
from dacboenv.experiment.headroom_v2 import BBOB_UPPER
from dacboenv.experiment.nonfeedback_registry import load_registry, registry_action
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.snapshot_branch import run_snapshot_branch_diagnostic, snapshot_record_digest
from dacboenv.policy.sb3_model import ModelPolicy
from dacboenv.reference import BBOBExactReferenceProvider, ManifestReferenceProvider

EARLY_PHASE_LIMIT = 0.375
MIDDLE_PHASE_LIMIT = 0.625


class _LearnedSnapshotPolicy:
    """Lazily bind the checkpoint to the exact evaluation environment."""

    def __init__(self, row: dict[str, Any]) -> None:
        self.row = row
        self.name = "stageb_final_deterministic"
        self.outer_seed = int(row["outer_ppo_seed"])
        self._policy: ModelPolicy | None = None
        self.last_probabilities: list[float] | None = None

    def __call__(self, observation: Any, env: Any) -> int:
        if self._policy is None:
            self._policy = ModelPolicy(
                env,
                model=self.row["model_path"],
                model_class="stable_baselines3.PPO",
                normalization_wrapper=self.row["normalization_path"],
            )
        model_observation = observation
        if hasattr(self._policy._vec_env, "normalize_obs"):
            model_observation = self._policy._vec_env.normalize_obs(observation)
        with torch.no_grad():
            tensor, _vectorized = self._policy._model.policy.obs_to_tensor(model_observation)
            distribution = self._policy._model.policy.get_distribution(tensor)
            probabilities = distribution.distribution.probs.detach().cpu().numpy().reshape(-1)
        self.last_probabilities = [float(value) for value in probabilities]
        action, _state = self._policy._model.predict(model_observation, deterministic=True)
        return int(np.asarray(action).item())

    def action_at_snapshot(self, observation_json: str, env: Any) -> tuple[int, list[float]]:
        payload = json.loads(observation_json)
        observation = {
            key: np.asarray(value["values"], dtype=np.dtype(value["dtype"])).reshape(value["shape"])
            for key, value in payload.items()
        }
        action = self(observation, env)
        assert self.last_probabilities is not None
        return action, self.last_probabilities


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
    """Execute one same-state five-action H=10 job."""
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("status") == "success" and existing.get("job_hash") == row["job_hash"]:
            return existing
        raise RuntimeError(f"Partial, corrupt, or another-model output already exists: {output}")
    if file_sha256(Path(row["model_path"])) != row["model_sha256"]:
        raise RuntimeError("Stage-B model hash changed after follow-up preparation.")
    if (
        row["normalization_path"] is not None
        and file_sha256(Path(row["normalization_path"])) != row["normalization_sha256"]
    ):
        raise RuntimeError("Stage-B normalization hash changed after follow-up preparation.")
    registry = load_registry(registry_path)
    if registry["registry_hash"] != row["selector_registry_hash"]:
        raise RuntimeError("Frozen nonfeedback selector registry differs from the job manifest.")
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
            interaction_frequency=int(row["interaction_frequency"]),
        )

    policy = _LearnedSnapshotPolicy(row)
    snapshots = collect_context_snapshots(
        task_id=task_id,
        inner_seed=int(row["inner_seed"]),
        env_factory=factory,
        policy=policy,
        budget_fractions=[float(row["snapshot_phase"])],
        action_space_name=str(row["action_family"]),
        source_manifest=str(row["manifest_id"]),
        source_manifest_hash=str(row["manifest_hash"]),
        reference_provider=provider,
    )
    snapshot = snapshots[0]
    learned_action, probabilities = policy.action_at_snapshot(snapshot.observation_json, None)
    bound = bind_snapshot_action_space_factory(snapshots, factory)
    report = run_snapshot_branch_diagnostic(
        snapshots,
        bound,
        lambda _task: float(snapshot.reference_value),
        forbidden_task_ids=set(),
        horizons=(1, 5, 10),
    )
    branches = [asdict(branch) for branch in report.branches]
    source_digests = {snapshot_record_digest(branch.snapshot) for branch in report.branches}
    if source_digests != {snapshot_record_digest(snapshot)}:
        raise RuntimeError("Action branches did not start from one identical portable source state.")
    metadata = _selector_metadata(row, snapshot)
    nonfeedback = {
        str(horizon): registry_action(registry, str(row["action_family"]), horizon, metadata) for horizon in (1, 5, 10)
    }
    payload = {
        "schema_version": "stageb-learned-headroom-job-v1",
        "status": "success",
        "job_hash": row["job_hash"],
        "job": row,
        "snapshot": asdict(snapshot),
        "source_snapshot_digest": snapshot_record_digest(snapshot),
        "learned_deterministic_action": learned_action,
        "policy_action_probabilities": probabilities,
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
    """Execute one indexed learned-state job."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    row = manifest["jobs"][args.job_index]
    output = args.output_root / f"{args.job_index:05d}.json"
    registry_path = args.manifest.resolve().parent / "nonfeedback_selector_registry.json"
    result = run_job(row, output, args.reference_table.resolve(), registry_path)
    print(json.dumps({"job_index": args.job_index, "status": result["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
