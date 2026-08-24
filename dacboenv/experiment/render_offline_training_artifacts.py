"""Render stable machine-readable contracts for the offline-Q implementation."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.finalize_offline_training_dataset import deterministic_task_split
from dacboenv.experiment.protocol import load_manifest
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.offline.losses import double_dqn_targets, offline_td_cql_loss
from dacboenv.offline.models.shared_dueling_q import OfflineQModelConfig, OfflineQNetwork
from dacboenv.offline.schema import (
    ALPHA_GRID,
    BEHAVIOR_REQUIRED_ARRAYS,
    BRANCH_REQUIRED_ARRAYS,
    MIDRUN_BRANCH_SCHEMA_VERSION,
    OFFLINE_FINAL_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _model_schemas() -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for variant in ("flat", "shared", "shared_dueling"):
        config = OfflineQModelConfig(variant=variant)  # type: ignore[arg-type]
        model = OfflineQNetwork(config)
        variants[variant] = {"config": asdict(config), "parameter_count": model.parameter_count}
    return {
        "schema_version": "dacbo-offline-q-model-v1",
        "input": {"global_state": [13], "action_features": [5, 4], "alpha_grid": ALPHA_GRID.tolist()},
        "heads": {
            "branch_q5": "fixed-action normalized-potential improvement over five BO evaluations",
            "branch_q10": "fixed-action normalized-potential improvement over ten BO evaluations",
            "long_q": "finite-episode Bellman action value over external f5 transitions",
        },
        "variants": variants,
        "primary": "shared_dueling",
    }


def render(config: DictConfig) -> dict[str, Any]:
    """Write schemas, formula test vectors, and a hash-bound registry."""
    output = Path(str(config.output_root)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset_schema = {
        "schema_version": OFFLINE_FINAL_SCHEMA_VERSION,
        "component": "behavior_f5",
        "required_arrays": sorted(BEHAVIOR_REQUIRED_ARRAYS),
        "observation_shapes": {
            "global_state": [13],
            "action_features": [5, 4],
            "next_global_state": [13],
            "next_action_features": [5, 4],
        },
        "action_grid": ALPHA_GRID.tolist(),
        "interaction_frequency": 5,
        "gamma": 1.0,
        "object_dtype_allowed": False,
        "split_unit": "complete_task_id",
    }
    branch_schema = {
        "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
        "required_arrays": sorted(BRANCH_REQUIRED_ARRAYS),
        "q5_required": True,
        "q10_optional": True,
        "next_observations_optional": True,
        "action_count": 5,
        "context_splits_allowed_for_training": ["train", "dev"],
        "midrun_execution_provenance_arrays": [
            "data_context_split",
            "environment_context_split",
        ],
        "midrun_execution_semantics": "task-split-environment-v2",
        "data_to_environment_split": {"train": "train", "dev": "validation"},
        "learned_policy_validation_headroom_allowed": False,
    }
    model_schema = _model_schemas()
    registry = {
        "schema_version": "dacbo-offline-algorithm-registry-v1",
        "algorithms": {
            "branch_q5_only": {"branch_head": True, "bellman": False, "cql": False},
            "offline_fqi": {"branch_head": False, "bellman": "double_dqn", "cql": False},
            "offline_cql": {"branch_head": False, "bellman": "double_dqn", "cql": True},
            "branch_pretrain_then_fqi": {"branch_head": "pretrain", "bellman": "double_dqn", "cql": False},
            "branch_pretrain_then_cql": {"branch_head": "pretrain", "bellman": "double_dqn", "cql": True},
            "joint_branch_cql": {"branch_head": "joint", "bellman": "double_dqn", "cql": True},
            "behavior_cloning": {"diagnostic_only": True},
        },
        "cql_coefficients": [0.1, 0.5, 1.0],
        "primary_domain": "yahpo",
        "experiment_cells": {
            "O0": "offline_fqi_yahpo",
            "O1": "replay_prefill_yahpo",
            "O2": "offline_cql_yahpo",
            "O3": "branch_q5_yahpo",
            "O4": "branch_q5_cql_yahpo",
            "O5": "branch_q5_cql_online_finetune_yahpo",
        },
    }
    for name, payload in (
        ("dataset_schema.json", dataset_schema),
        ("branch_schema.json", branch_schema),
        ("model_schema.json", model_schema),
        ("algorithm_registry.json", registry),
    ):
        _atomic_json(output / name, payload)

    repository = Path(__file__).resolve().parents[2]
    episode_metadata = []
    for filename in ("bbob_train.yaml", "yahpo_train.yaml"):
        task_manifest = load_manifest(repository / "dacboenv/configs/instance_sets" / filename)
        for task_id in task_manifest["task_ids"]:
            task = str(task_id)
            episode_metadata.append(
                {
                    "task_id": task,
                    "domain": "yahpo" if task.startswith("yahpo/") else "bbob",
                    "scenario": task.split("/")[2] if task.startswith("yahpo/") else "bbob",
                }
            )
    assignment = deterministic_task_split(episode_metadata, "dacbo-offline-v3-split-2026-08-24")
    split_preview = {
        "schema_version": "dacbo-offline-task-split-preview-v1",
        "source": "frozen bbob_train/yahpo_train manifests",
        "split_seed": "dacbo-offline-v3-split-2026-08-24",
        "task_split_hash": canonical_sha256(assignment),
        "splits": {
            split: {
                "tasks": sorted(task for task, assigned in assignment.items() if assigned == split),
            }
            for split in ("train", "dev", "holdout")
        },
    }
    for split in split_preview["splits"].values():
        split["count"] = len(split["tasks"])
        split["task_list_hash"] = canonical_sha256(split["tasks"])
    _atomic_json(output / "task_split_preview.json", split_preview)

    reward = torch.tensor([1.0, 2.0])
    done = torch.tensor([0.0, 1.0])
    online_next = torch.tensor([[9.0, 1.0], [1.0, 3.0]])
    target_next = torch.tensor([[2.0, 8.0], [4.0, 5.0]])
    targets = double_dqn_targets(reward, done, online_next, target_next).numpy(force=True)
    q_values = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5]])
    cql = offline_td_cql_loss(q_values, torch.tensor([4]), torch.tensor([0.6]), cql_coefficient=0.5)
    vectors = output / "test_vectors.npz"
    np.savez_compressed(
        vectors,
        reward=reward.numpy(force=True),
        done=done.numpy(force=True),
        online_next=online_next.numpy(force=True),
        target_next=target_next.numpy(force=True),
        double_dqn_target=targets,
        cql_data_q=np.asarray([float(cql.data_q)]),
        cql_penalty=np.asarray([float(cql.cql)]),
    )
    inputs = {
        "source_revision": current_source_revision(repository),
        "files": {
            path.name: file_sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "protocol_hash.json"
        },
    }
    protocol = {**inputs, "protocol_hash": canonical_sha256(inputs)}
    _atomic_json(output / "protocol_hash.json", protocol)
    return protocol


@hydra.main(  # type: ignore[untyped-decorator]
    version_base=None,
    config_path="../configs/offline_artifacts",
    config_name="base",
)
def main(config: DictConfig) -> None:
    """Render the repository-owned offline implementation artifacts."""
    print(json.dumps(render(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
