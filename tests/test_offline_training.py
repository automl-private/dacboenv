"""Scientific contracts for task-disjoint offline dynamic-WEI learning."""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from dacboenv.experiment import prepare_offline_carps_evaluation as carps_eval_module
from dacboenv.experiment.collect_offline_policies import export
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.finalize_offline_training_dataset import finalize
from dacboenv.experiment.offline_branch_campaign import environment_context_split, prepare
from dacboenv.experiment.offline_train import _branch_dev_metrics, train
from dacboenv.experiment.ppo import validate_offline_finetune_task_boundary
from dacboenv.experiment.protocol import load_manifest
from dacboenv.experiment.sb3_algorithms import build_sb3_algorithm
from dacboenv.offline.branch_dataset import BranchDataset
from dacboenv.offline.dataset import BehaviorDataset, HoldoutAccessError
from dacboenv.offline.deployment import (
    DeploymentSelectionState,
    deployment_head_for_mode,
    deployment_selection_eligible,
)
from dacboenv.offline.identity import offline_policy_id, stable_float_slug
from dacboenv.offline.losses import centered_huber_pairwise_loss, double_dqn_targets, offline_td_cql_loss
from dacboenv.offline.models.shared_dueling_q import OfflineQModelConfig, OfflineQNetwork
from dacboenv.offline.provenance import reject_training_provenance
from dacboenv.offline.replay_prefill import (
    OfflineOnlineMixSchedule,
    configure_offline_replay,
    prefill_dict_replay_buffer,
)
from dacboenv.offline.sampler import (
    BranchBatchSampler,
    BranchSamplerConfig,
    HierarchicalBatchSampler,
    SamplerConfig,
)
from gymnasium import Env, spaces
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv


def _source_fixture(path: Path) -> None:
    tasks = [
        "bbob/2/1/0",
        "bbob/2/2/0",
        "bbob/2/3/0",
        "yahpo/so/lcbench/1/None",
        "yahpo/so/lcbench/2/None",
        "yahpo/so/lcbench/3/None",
    ]
    observations, action_features, next_observations = [], [], []
    actions, rewards, terminated, truncated = [], [], [], []
    probabilities, log_probabilities, before, after, durations = [], [], [], [], []
    metadata, offsets = [], [0]
    alpha_rows = np.broadcast_to(np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)[:, None], (5, 4)).copy()
    for task_index, task in enumerate(tasks):
        domain = "yahpo" if task.startswith("yahpo/") else "bbob"
        scenario = "lcbench" if domain == "yahpo" else "bbob"
        for seed in range(5):
            base_state = np.linspace(0, 1, 13, dtype=np.float32) + task_index + seed / 10
            for policy in range(6):
                action = policy if policy < 5 else (task_index + seed) % 5
                observations.append(base_state)
                action_features.append(alpha_rows)
                next_observations.append(base_state + 0.01 * (action + 1))
                actions.append(action)
                rewards.append(0.01 * (action + 1))
                terminated.append(True)
                truncated.append(False)
                probability = 1.0 if policy < 5 else 0.2
                probabilities.append(probability)
                log_probabilities.append(np.log(probability))
                before.append(5)
                after.append(10)
                durations.append(5)
                metadata.append(
                    {
                        "task_id": task,
                        "domain": domain,
                        "scenario": scenario,
                        "seed": seed,
                        "policy_id": f"static_f5_alpha{policy}" if policy < 5 else "uniform_random_f5",
                        "policy_kind": "static" if policy < 5 else "random",
                        "configured_action": policy if policy < 5 else None,
                        "bo_budget": 30,
                    }
                )
                offsets.append(len(actions))
    strings = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in metadata]
    dataset_metadata = {
        "schema_version": "dacbo-offline-training-f5-v2",
        "data_role": "training",
        "context_split": "train",
        "source_revision": "synthetic",
    }
    np.savez_compressed(
        path,
        observations__global_state=np.asarray(observations, dtype=np.float32),
        observations__action_features=np.asarray(action_features, dtype=np.float32),
        next_observations__global_state=np.asarray(next_observations, dtype=np.float32),
        next_observations__action_features=np.asarray(action_features, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int8),
        behavior_probability=np.asarray(probabilities, dtype=np.float32),
        behavior_log_probability=np.asarray(log_probabilities, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float64),
        terminated=np.asarray(terminated, dtype=np.bool_),
        truncated=np.asarray(truncated, dtype=np.bool_),
        bo_evaluations_before=np.asarray(before, dtype=np.int32),
        bo_evaluations_after=np.asarray(after, dtype=np.int32),
        realized_duration=np.asarray(durations, dtype=np.int16),
        episode_offsets=np.asarray(offsets, dtype=np.int64),
        episode_metadata_json=np.asarray(strings, dtype=f"U{max(map(len, strings))}"),
        dataset_metadata_json=np.asarray(json.dumps(dataset_metadata, sort_keys=True, separators=(",", ":"))),
    )


@pytest.fixture
def finalized(tmp_path: Path) -> Path:
    source = tmp_path / "source.npz"
    output = tmp_path / "final"
    _source_fixture(source)
    config = OmegaConf.create(
        {
            "behavior_npz": str(source),
            "output_root": str(output),
            "learned_headroom_root": None,
            "split_seed": "offline-test-split",
            "tie_tolerance": 1e-3,
            "strict_expected_corpus": False,
        }
    )
    finalize(config)
    return output


def test_finalization_is_task_disjoint_pickle_free_and_all_action_paired(finalized: Path) -> None:
    manifest = json.loads((finalized / "final_offline_dataset_manifest.json").read_text())
    split_sets = [set(manifest["task_splits"][split]) for split in ("train", "dev", "holdout")]
    assert not split_sets[0] & split_sets[1]
    assert not split_sets[0] & split_sets[2]
    assert not split_sets[1] & split_sets[2]
    assert set.union(*split_sets) == {
        "bbob/2/1/0",
        "bbob/2/2/0",
        "bbob/2/3/0",
        "yahpo/so/lcbench/1/None",
        "yahpo/so/lcbench/2/None",
        "yahpo/so/lcbench/3/None",
    }
    with np.load(finalized / "behavior_train.npz", allow_pickle=False) as behavior:
        assert all(not behavior[key].dtype.hasobject for key in behavior.files)
        assert set(np.unique(behavior["policy_id"])) == {
            "static_f5_alpha0",
            "static_f5_alpha1",
            "static_f5_alpha2",
            "static_f5_alpha3",
            "static_f5_alpha4",
            "uniform_random_f5",
        }
        assert set(np.unique(behavior["seed"])) == set(range(5))
    branch = BranchDataset(finalized / "initial_counterfactual_train.npz")
    assert branch.arrays["q5"].shape[1] == 5
    assert np.all(branch.arrays["oracle_action"] == 4)
    train_manifest = load_manifest(finalized / "offline_train_v3.yaml")
    dev_manifest = load_manifest(finalized / "offline_dev_v3.yaml")
    holdout_manifest = load_manifest(finalized / "offline_holdout_v3.yaml")
    assert train_manifest["runnable"]
    assert dev_manifest["runnable"]
    assert not holdout_manifest["runnable"]
    config = OmegaConf.create(
        {
            "behavior_npz": manifest["source_dataset"],
            "output_root": str(finalized),
            "learned_headroom_root": None,
            "split_seed": "this-value-is-ignored-after-freeze",
            "tie_tolerance": 1e-3,
            "strict_expected_corpus": False,
        }
    )
    assert finalize(config)["manifest_hash"] == manifest["manifest_hash"]


def test_holdout_fails_closed_and_learned_headroom_is_rejected(finalized: Path) -> None:
    with pytest.raises(HoldoutAccessError, match="sealed"):
        BehaviorDataset(finalized / "behavior_holdout.npz")
    with pytest.raises(ValueError, match="Forbidden"):
        reject_training_provenance({"data_role": "learned_policy_validation_headroom", "context_split": "validation"})


def test_online_finetune_accepts_only_frozen_train_and_dev_tasks(tmp_path: Path) -> None:
    manifest = tmp_path / "final_offline_dataset_manifest.json"
    manifest.write_text(
        json.dumps({"task_splits": {"train": ["train-a"], "dev": ["dev-a"], "holdout": ["sealed-a"]}}),
        encoding="utf-8",
    )
    validate_offline_finetune_task_boundary(manifest, ["train-a"], ["dev-a"])
    with pytest.raises(ValueError, match="offline_train_v3"):
        validate_offline_finetune_task_boundary(manifest, ["sealed-a"], ["dev-a"])
    with pytest.raises(ValueError, match="offline_dev_v3"):
        validate_offline_finetune_task_boundary(manifest, ["train-a"], ["sealed-a"])


def test_train_only_normalization_and_hierarchical_sampling(finalized: Path) -> None:
    dataset = BehaviorDataset(finalized / "behavior_train.npz")
    sampler = HierarchicalBatchSampler(
        dataset,
        SamplerConfig(batch_size=40, ordinary_fraction=0.75, positive_fraction=0.25),
        seed=7,
    )
    first = sampler.sample()
    replay = HierarchicalBatchSampler(
        dataset,
        SamplerConfig(batch_size=40, ordinary_fraction=0.75, positive_fraction=0.25),
        seed=7,
    ).sample()
    assert np.array_equal(first, replay)
    assert sampler.last_composition["bbob_fraction"] == pytest.approx(0.5)
    assert sampler.last_composition["yahpo_fraction"] == pytest.approx(0.5)
    yahpo_indices = np.flatnonzero(dataset.arrays["domain_id"] == 1)
    yahpo_sampler = HierarchicalBatchSampler(
        dataset,
        SamplerConfig(batch_size=20, bbob_fraction=0.0, ordinary_fraction=0.75, positive_fraction=0.25),
        seed=8,
        eligible_indices=yahpo_indices,
    )
    yahpo_sampler.sample()
    assert yahpo_sampler.last_composition["yahpo_fraction"] == 1.0
    normalization = json.loads((finalized / "normalization_schema.json").read_text())
    assert (
        normalization["train_dataset_sha256"]
        == json.loads((finalized / "final_offline_dataset_manifest.json").read_text())["outputs"]["train"]["behavior"][
            "sha256"
        ]
    )
    assert normalization["action_features"]["preserve_mask"][0]


def test_shared_dueling_is_action_permutation_equivariant_and_heads_are_separate() -> None:
    torch.manual_seed(3)
    model = OfflineQNetwork(OfflineQModelConfig())
    state = torch.randn(4, 13)
    features = torch.randn(4, 5, 4)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    for head in ("long_q", "branch_q5", "branch_q10"):
        expected = model(state, features, head=head)
        actual = model(state, features[:, permutation], head=head)
        assert torch.equal(actual, expected[:, permutation])
    assert not torch.equal(model(state, features, head="long_q"), model(state, features, head="branch_q5"))
    assert model.parameter_count == 50278


def test_branch_ranking_double_dqn_and_cql_losses_are_finite() -> None:
    prediction = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]], requires_grad=True)
    target = torch.tensor([[0.0, 0.0, 0.2, 0.4, 0.4]])
    branch = centered_huber_pairwise_loss(prediction, target, torch.ones_like(target, dtype=torch.bool))
    assert branch.contributing_pairs > 0
    assert torch.isfinite(branch.total)
    online = torch.tensor([[9.0, 1.0]])
    target_next = torch.tensor([[2.0, 8.0]])
    bootstrap = double_dqn_targets(torch.tensor([1.0]), torch.tensor([0.0]), online, target_next)
    assert bootstrap.item() == pytest.approx(3.0)
    cql = offline_td_cql_loss(prediction, torch.tensor([4]), torch.tensor([0.6]), cql_coefficient=0.5)
    assert torch.isfinite(cql.total)
    duplicates = centered_huber_pairwise_loss(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        duplicate_groups=torch.zeros_like(target, dtype=torch.long),
    )
    assert duplicates.contributing_pairs == 0


def test_branch_sampler_is_deterministic_and_reports_high_gap_duplication(finalized: Path) -> None:
    branch = BranchDataset(finalized / "initial_counterfactual_train.npz")
    config = BranchSamplerConfig(batch_size=20, high_gap_fraction=0.25)
    first_sampler = BranchBatchSampler(branch, config, seed=19)
    second_sampler = BranchBatchSampler(branch, config, seed=19)
    assert np.array_equal(first_sampler.sample(), second_sampler.sample())
    assert first_sampler.last_composition["bbob_fraction"] == pytest.approx(0.5)
    assert first_sampler.last_composition["high_gap_fraction"] >= 0.25
    assert "effective_unique_states" in first_sampler.last_composition


def test_replay_prefill_preserves_finite_episode_terminal_semantics(finalized: Path) -> None:
    dataset = BehaviorDataset(finalized / "behavior_train.npz")
    observation_space = spaces.Dict(
        {
            "global_state": spaces.Box(-np.inf, np.inf, shape=(13,), dtype=np.float32),
            "action_features": spaces.Box(-np.inf, np.inf, shape=(5, 4), dtype=np.float32),
        }
    )
    replay = DictReplayBuffer(
        128,
        observation_space,
        spaces.Discrete(5),
        n_envs=1,
        handle_timeout_termination=False,
    )
    result = prefill_dict_replay_buffer(replay, dataset, seed=4, maximum_transitions=32)
    assert result == {"inserted": 32, "unique": 32, "dropped_for_vector_alignment": 0, "buffer_size": 32}
    assert replay.dones[: replay.pos].all()


def _training_config(finalized: Path, output: Path, *, resume: Path | None = None) -> object:
    repository = Path(__file__).resolve().parents[1]
    return OmegaConf.merge(
        OmegaConf.load(repository / "dacboenv/configs/offline_train.yaml"),
        {
            "offline_dataset": {
                "root": str(finalized),
                "branch_train": None,
                "branch_dev": None,
                "allow_holdout": False,
                "holdout_reason": None,
            },
            "offline_model": OmegaConf.load(repository / "dacboenv/configs/offline_model/shared_dueling.yaml"),
            "offline_algorithm": OmegaConf.load(repository / "dacboenv/configs/offline_algorithm/offline_cql.yaml"),
            "offline_training": {
                **OmegaConf.to_container(
                    OmegaConf.load(repository / "dacboenv/configs/offline_training/base.yaml"), resolve=False
                ),
                "experiment_id": "resume-smoke",
                "domain": "mixed",
                "algorithm_mode": "offline_cql",
                "maximum_updates": 4,
                "batch_size": 20,
                "branch_batch_size": 10,
                "dev_interval": 2,
                "checkpoint_interval": 2,
                "target_update_interval": 2,
                "resume_from": None if resume is None else str(resume),
            },
            "seed": 8,
            "device": "cpu",
            "output_root": str(output),
        },
    )


def test_offline_cql_checkpoint_resume_is_exact(finalized: Path, tmp_path: Path) -> None:
    output = tmp_path / "run"
    train(_training_config(finalized, output))
    original = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
    train(_training_config(finalized, output, resume=output / "step_00002.pt"))
    resumed = torch.load(output / "final.pt", map_location="cpu", weights_only=False)
    for key in original["model_state"]:
        assert torch.equal(original["model_state"][key], resumed["model_state"][key])
    assert original["history"] == resumed["history"]
    assert original["deployment_selection"] == resumed["deployment_selection"]
    assert resumed["deployment_selection"]["checkpoint_selection_metric"] == "dev/deployment_selected_value"


class _ControlledHeads:
    """Minimal model fixture with independently controlled deployed/branch rankings."""

    def __init__(self, long_action: int, branch_action: int) -> None:
        self.actions = {"long_q": long_action, "branch_q5": branch_action}

    def eval(self) -> _ControlledHeads:
        return self

    def __call__(self, state: torch.Tensor, features: torch.Tensor, *, head: str) -> torch.Tensor:
        values = torch.zeros((len(state), features.shape[1]), device=state.device)
        values[:, self.actions[head]] = 1.0
        return values


def test_deployment_head_metrics_and_selection_contract(finalized: Path) -> None:
    branch = BranchDataset(finalized / "initial_counterfactual_dev.npz")
    indices = np.arange(len(branch), dtype=np.int64)
    comparators = np.zeros(len(branch), dtype=np.int64)
    model = _ControlledHeads(long_action=4, branch_action=0)
    metrics = _branch_dev_metrics(  # type: ignore[arg-type]
        model,
        branch,
        indices,
        torch.device("cpu"),
        comparators,
        0,
        "long_q",
    )
    assert metrics["deployment_selected_value"] > metrics["branch_q5_selected_value"]
    before = metrics["deployment_selected_value"]
    model.actions["branch_q5"] = 3
    changed_branch = _branch_dev_metrics(  # type: ignore[arg-type]
        model,
        branch,
        indices,
        torch.device("cpu"),
        comparators,
        0,
        "long_q",
    )
    assert changed_branch["deployment_selected_value"] == before
    assert changed_branch["branch_q5_selected_value"] != metrics["branch_q5_selected_value"]
    model.actions["long_q"] = 2
    changed_long = _branch_dev_metrics(  # type: ignore[arg-type]
        model,
        branch,
        indices,
        torch.device("cpu"),
        comparators,
        0,
        "long_q",
    )
    assert changed_long["deployment_selected_value"] != before

    expected = {
        "branch_q5_only": "branch_q5",
        "offline_fqi": "long_q",
        "offline_cql": "long_q",
        "branch_pretrain_then_fqi": "long_q",
        "branch_pretrain_then_cql": "long_q",
        "joint_branch_cql": "long_q",
        "behavior_cloning": "long_q",
    }
    assert {mode: deployment_head_for_mode(mode) for mode in expected} == expected


def test_staged_o4_selection_eligibility_and_patience() -> None:
    state = DeploymentSelectionState(deployment_head="long_q")
    assert not deployment_selection_eligible("branch_pretrain_then_cql", 5, 5)
    assert not state.consider(value=100.0, update=5, eligible=False)
    assert state.selected_update is None
    assert state.patience_counter == 0
    assert deployment_selection_eligible("branch_pretrain_then_cql", 6, 5)
    assert state.consider(value=1.0, update=6, eligible=True)
    assert state.selected_update == 6
    assert state.patience_counter == 0
    assert not state.consider(value=0.5, update=7, eligible=True)
    assert state.patience_counter == 1
    assert deployment_selection_eligible("branch_q5_only", 1, 5000)


def test_o3_and_staged_o4_checkpoint_selection_smokes(finalized: Path, tmp_path: Path) -> None:
    o3_config = _training_config(finalized, tmp_path / "o3")
    o3_config.offline_training.algorithm_mode = "branch_q5_only"
    o3_config.offline_training.experiment_id = "O3-smoke"
    o3_config.offline_training.maximum_updates = 2
    o3_config.offline_training.dev_interval = 1
    o3_summary = train(o3_config)
    o3_checkpoint = torch.load(tmp_path / "o3" / "best_branch_dev.pt", map_location="cpu", weights_only=False)
    assert o3_summary["deployment_head"] == "branch_q5"
    assert o3_summary["selected_update"] in {1, 2}
    assert o3_checkpoint["deployment_selection"]["checkpoint_selection_head"] == "branch_q5"

    o4_config = _training_config(finalized, tmp_path / "o4")
    o4_config.offline_training.algorithm_mode = "branch_pretrain_then_cql"
    o4_config.offline_training.experiment_id = "O4-smoke"
    o4_config.offline_training.branch_pretrain_updates = 1
    o4_config.offline_training.maximum_updates = 2
    o4_config.offline_training.dev_interval = 1
    o4_summary = train(o4_config)
    o4_checkpoint = torch.load(tmp_path / "o4" / "best_branch_dev.pt", map_location="cpu", weights_only=False)
    assert o4_summary["deployment_head"] == "long_q"
    assert o4_summary["selected_update"] == 2
    assert o4_checkpoint["deployment_selection"]["checkpoint_selection_head"] == "long_q"
    assert o4_summary["checkpoint_selection_metric"] == "dev/deployment_selected_value"
    history = json.loads((tmp_path / "o4" / "training_history.json").read_text())
    assert history[0]["dev/deployment_selection_eligible"] == 0.0
    assert history[1]["dev/deployment_selection_eligible"] == 1.0


def test_branch_manifest_export_and_otus_shell_contracts(finalized: Path, tmp_path: Path) -> None:
    branch_root = tmp_path / "branch"
    manifest = prepare(
        OmegaConf.create(
            {
                "final_dataset_root": str(finalized),
                "output_root": str(branch_root),
                "source_policies": ["uniform_random"],
                "seeds": [0, 1, 2, 3, 4],
                "phases": [0.25, 0.5, 0.75],
            }
        )
    )
    assert {row["context_split"] for row in manifest["jobs"]} == {"train", "dev"}
    assert {(row["data_context_split"], row["environment_context_split"]) for row in manifest["jobs"]} == {
        ("train", "train"),
        ("dev", "validation"),
    }
    assert environment_context_split("train") == "train"
    assert environment_context_split("dev") == "validation"
    with pytest.raises(ValueError, match="forbid"):
        environment_context_split("holdout")
    assert not any(row["context_split"] == "holdout" for row in manifest["jobs"])
    assert len(manifest["jobs"]) == (2 + 2) * 5 * 3

    run_output = tmp_path / "offline-run"
    train(_training_config(finalized, run_output))
    exported = export(
        OmegaConf.create(
            {
                "run_root": str(tmp_path),
                "output_root": str(tmp_path / "bundle"),
                "checkpoint": "best_branch_dev",
                "explicit_checkpoint": None,
            }
        )
    )
    assert exported["policies"][0]["deployment_head"] == "long_q"
    assert exported["policies"][0]["checkpoint_selection_metric"] == "dev/deployment_selected_value"
    assert exported["policies"][0]["interaction_frequency"] == 5

    repository = Path(__file__).resolve().parents[1]
    scripts = sorted((repository / "scripts/otus").glob("*offline*.sh"))
    assert scripts
    for script in scripts:
        subprocess.run(["/usr/bin/bash", "-n", str(script)], check=True)  # noqa: S603
        if script.name in {"otus_collect_offline_branches.sh", "otus_train_offline_q.sh"}:
            array_lines = [line for line in script.read_text(encoding="utf-8").splitlines() if "sbatch --array" in line]
            assert all("%" not in line for line in array_lines)
    finetune_source = (repository / "scripts/otus/otus_finetune_offline_ddqn.sh").read_text(encoding="utf-8")
    assert "source_policy_id" in finetune_source
    assert "duplicate fine-tuning run IDs" in finetune_source
    assert "duplicate source policy IDs" in finetune_source


def _copy_export_run(source: Path, destination: Path, coefficient: float) -> None:
    destination.mkdir(parents=True)
    payload = torch.load(source / "best_branch_dev.pt", map_location="cpu", weights_only=False)
    payload["resolved_config"]["offline_training"]["experiment_id"] = "O2"
    payload["resolved_config"]["offline_training"]["algorithm_mode"] = "offline_cql"
    payload["resolved_config"]["offline_training"]["cql_coefficient"] = coefficient
    checkpoint = destination / "best_branch_dev.pt"
    torch.save(payload, checkpoint)
    shutil.copyfile(source / "normalization_schema.json", destination / "normalization_schema.json")
    shutil.copyfile(
        source / "training_fitted_nonfeedback_registry.json",
        destination / "training_fitted_nonfeedback_registry.json",
    )
    completion = {
        "status": "complete",
        "best_branch_dev_sha256": file_sha256(checkpoint),
    }
    (destination / "training_complete.json").write_text(json.dumps(completion), encoding="utf-8")


def test_collision_proof_policy_identity_and_atomic_export(finalized: Path, tmp_path: Path) -> None:
    source = tmp_path / "source-run"
    train(_training_config(finalized, source))
    run_root = tmp_path / "runs"
    for coefficient in (0.1, 0.5, 1.0):
        _copy_export_run(source, run_root / stable_float_slug(coefficient), coefficient)
    bundle_root = tmp_path / "bundle"
    inventory = export(
        OmegaConf.create(
            {
                "run_root": str(run_root),
                "output_root": str(bundle_root),
                "checkpoint": "best_branch_dev",
                "explicit_checkpoint": None,
            }
        )
    )
    identities = [row["policy_id"] for row in inventory["policies"]]
    assert len(identities) == len(set(identities)) == 3
    assert {row["cql_coefficient"] for row in inventory["policies"]} == {0.1, 0.5, 1.0}
    assert all((bundle_root / "policies" / f"{identity}.yaml").is_file() for identity in identities)
    assert {stable_float_slug(value) for value in (0.0, 0.1, 0.5, 1.0)} == {"0", "0p1", "0p5", "1"}
    assert (
        len(
            {
                offline_policy_id(
                    experiment_id="O2",
                    algorithm_mode="offline_cql",
                    cql_coefficient=value,
                    training_seed=0,
                    selected_update=4,
                    checkpoint_mode="best_branch_dev",
                    model_sha256="a" * 64,
                )
                for value in (0.1, 0.5, 1.0)
            }
        )
        == 3
    )

    duplicate_root = tmp_path / "duplicates"
    shutil.copytree(run_root / "0p1", duplicate_root / "a")
    shutil.copytree(run_root / "0p1", duplicate_root / "b")
    rejected_output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="duplicate"):
        export(
            OmegaConf.create(
                {
                    "run_root": str(duplicate_root),
                    "output_root": str(rejected_output),
                    "checkpoint": "best_branch_dev",
                    "explicit_checkpoint": None,
                }
            )
        )
    assert not (rejected_output / "policies").exists()


def test_carps_hydra_layout_and_parallel_failure_propagation(
    finalized: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    train(_training_config(finalized, run))
    export(
        OmegaConf.create(
            {
                "run_root": str(run),
                "output_root": str(tmp_path / "bundle"),
                "checkpoint": "best_branch_dev",
                "explicit_checkpoint": None,
            }
        )
    )
    monkeypatch.setattr(
        carps_eval_module,
        "_task_configs",
        lambda _tasks, _root: {"BBOB": ["dummy-bbob"], "YAHPO/SO": ["dummy-yahpo"]},
    )
    evaluation_root = tmp_path / "evaluation"
    plan = carps_eval_module.prepare(
        OmegaConf.create(
            {
                "policy_inventory": str(tmp_path / "bundle" / "offline_policy_inventory.json"),
                "output_root": str(evaluation_root),
            }
        )
    )
    assert Path(plan["scientific_result_root"]) == evaluation_root / "runs"
    assert Path(plan["hydra_sweep_root"]) == evaluation_root / "hydra_sweeps"
    script = Path(plan["launcher"])
    source = script.read_text(encoding="utf-8")
    assert f"baserundir={evaluation_root}/runs" in source
    assert f"hydra.sweep.dir={evaluation_root}/hydra_sweeps/" in source
    assert "hydra.job.chdir=false" in source
    launch_lines = [line for line in source.splitlines() if line.startswith("launch ")]
    assert len(launch_lines) == len(plan["launcher_labels"])
    assert len(set(plan["launcher_labels"])) == len(plan["launcher_labels"])
    marker = tmp_path / "later-launcher-ran"
    rewritten: list[str] = []
    launch_index = 0
    for line in source.splitlines():
        if not line.startswith("launch "):
            rewritten.append(line)
            continue
        label = shlex.split(line)[1]
        command = "exit 1" if launch_index == 0 else f"printf ok >> {shlex.quote(str(marker))}"
        rewritten.append(f"launch {shlex.quote(label)} /usr/bin/bash -c {shlex.quote(command)}")
        launch_index += 1
    script.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    result = subprocess.run(["/usr/bin/bash", str(script)], capture_output=True, text=True, check=False)  # noqa: S603
    assert result.returncode != 0
    assert plan["launcher_labels"][0] in result.stderr
    assert marker.is_file()


def test_staged_otus_manifests_have_exact_counts_and_unique_outputs(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text("#!/usr/bin/env bash\nprintf 'Submitted batch job 1\\n'\n", encoding="utf-8")
    fake_sbatch.chmod(0o755)
    final_root, branch_root = tmp_path / "final", tmp_path / "branch"
    final_root.mkdir()
    branch_root.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OTUS_PYTHON": str(repository / ".env/bin/python"),
    }

    def launch(mode: str, *extra: str) -> list[dict[str, str]]:
        output = tmp_path / mode
        subprocess.run(  # noqa: S603
            [
                "/usr/bin/bash",
                str(repository / "scripts/otus/otus_train_offline_q.sh"),
                str(final_root),
                str(branch_root),
                str(output),
                mode,
                *extra,
            ],
            check=True,
            cwd=repository,
            env=environment,
        )
        with (output / f"offline_training_{mode}.tsv").open(encoding="utf-8") as stream:
            return list(csv.DictReader(stream, delimiter="\t"))

    stage1 = launch("yahpo_stage1")
    assert len(stage1) == 12
    assert {row["cell"] for row in stage1} == {
        "branch_q5_yahpo",
        "offline_fqi_yahpo",
        "offline_cql_yahpo",
    }
    assert {row["cql_coefficient"] for row in stage1 if row["cell"] == "offline_cql_yahpo"} == {"0.1", "0.5"}
    assert len({row["output_path"] for row in stage1}) == 12
    o4 = launch("yahpo_o4", "--cql-coefficient", "0.1")
    assert len(o4) == 3
    assert {row["cell"] for row in o4} == {"branch_q5_cql_yahpo"}
    assert {row["cql_coefficient"] for row in o4} == {"0.1"}
    assert {row["updates"] for row in o4} == {"15000"}
    assert len({row["output_path"] for row in o4}) == 3


def test_offline_finetune_resolved_budget_and_utd(finalized: Path) -> None:
    manifest = json.loads((finalized / "final_offline_dataset_manifest.json").read_text())
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        yahpo = compose(
            config_name=None,
            overrides=[
                f"hydra.searchpath=[file://{finalized}/hydra_configs]",
                "+training=offline_ddqn_f5_finetune",
            ],
        )
    assert yahpo.experiment.n_workers == 12
    assert yahpo.rl_algorithm.hyperparameters.gradient_steps == 3
    assert yahpo.experiment.total_timesteps == 12288
    assert yahpo.experiment.checkpoint_freq == 3072
    assert yahpo.experiment.bo_evaluation_budget == 61440
    assert yahpo.experiment.checkpoint_bo_evaluations == 15360
    assert set(yahpo.dacboenv.task_ids) <= set(manifest["task_splits"]["train"])
    assert set(yahpo.experiment.validation.task_ids) <= set(manifest["task_splits"]["dev"])

    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        mixed = compose(
            config_name=None,
            overrides=[
                f"hydra.searchpath=[file://{finalized}/hydra_configs]",
                "+training=offline_ddqn_f5_finetune_mixed",
            ],
        )
    assert mixed.rl_algorithm.hyperparameters.gradient_steps / mixed.experiment.n_workers == pytest.approx(0.25)
    assert set(mixed.dacboenv.task_ids) == set(manifest["task_splits"]["train"])
    assert set(mixed.experiment.validation.task_ids) == set(manifest["task_splits"]["dev"])


class _TinyDictEnv(Env[dict[str, np.ndarray], int]):
    observation_space = spaces.Dict(
        {
            "global_state": spaces.Box(-100, 100, shape=(13,), dtype=np.float32),
            "action_features": spaces.Box(-100, 100, shape=(5, 4), dtype=np.float32),
        }
    )
    action_space = spaces.Discrete(5)

    def __init__(self) -> None:
        self.steps = 0

    def _observation(self) -> dict[str, np.ndarray]:
        features = np.zeros((5, 4), dtype=np.float32)
        features[:, 0] = np.linspace(0, 1, 5)
        return {"global_state": np.full(13, self.steps / 10, dtype=np.float32), "action_features": features}

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None) -> tuple[dict, dict]:
        del options
        super().reset(seed=seed)
        self.steps = 0
        return self._observation(), {}

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:
        self.steps += 1
        return self._observation(), float(action == 4), self.steps >= 3, False, {}


def test_offline_weights_prefill_mixture_and_double_dqn_save_load(finalized: Path, tmp_path: Path) -> None:
    run_output = tmp_path / "offline"
    train(_training_config(finalized, run_output))
    checkpoint = run_output / "final.pt"
    normalizer = finalized / "normalization_schema.json"
    cfg = OmegaConf.create(
        {
            "rl_algorithm_id": "double_dqn",
            "rl_algorithm": {
                "hyperparameters": {
                    "policy": "MultiInputPolicy",
                    "learning_rate": 1e-4,
                    "buffer_size": 128,
                    "learning_starts": 0,
                    "batch_size": 4,
                    "tau": 1.0,
                    "gamma": 1.0,
                    "train_freq": [1, "step"],
                    "gradient_steps": 1,
                    "target_update_interval": 2,
                    "exploration_fraction": 0.1,
                    "exploration_initial_eps": 0.1,
                    "exploration_final_eps": 0.0,
                    "max_grad_norm": 10.0,
                    "optimize_memory_usage": False,
                    "n_steps": 1,
                    "replay_buffer_kwargs": {},
                    "verbose": 0,
                },
                "policy_kwargs": {},
            },
            "offline_initialization": {
                "enabled": True,
                "checkpoint": str(checkpoint),
                "normalizer": str(normalizer),
                "checkpoint_sha256": file_sha256(checkpoint),
                "normalizer_sha256": file_sha256(normalizer),
            },
        }
    )
    env = DummyVecEnv([_TinyDictEnv])
    model = build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path / "tb"), model_seed=31)
    result = configure_offline_replay(
        model,
        dataset_path=finalized / "behavior_train.npz",
        seed=9,
        maximum_transitions=32,
        mixture=OfflineOnlineMixSchedule(initial_fraction=0.5, final_fraction=0.1, decay_steps=4),
    )
    assert result["inserted"] == 32
    model.learn(total_timesteps=4)
    observation = env.reset()
    q_before = model.q_net(model.policy.obs_to_tensor(observation)[0]).detach()
    model.save(tmp_path / "ddqn")
    loaded = type(model).load(tmp_path / "ddqn", env=env)
    q_after = loaded.q_net(loaded.policy.obs_to_tensor(observation)[0]).detach()
    assert torch.equal(q_before, q_after)
