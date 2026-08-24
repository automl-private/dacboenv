"""Bounded real CARP-S integration for exported offline action-value policies."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.real_env import real_structured_mixed_env
from dacboenv.experiment.sb3_algorithms import build_sb3_algorithm
from dacboenv.offline.models.shared_dueling_q import OfflineQModelConfig, OfflineQNetwork
from dacboenv.offline.normalization import fit_observation_normalizer
from dacboenv.policy.offline_q import OfflineQPolicy
from dacboenv.policy.sb3_model import SB3DiscretePolicy
from dacboenv.rl.double_dqn import DoubleDQN
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv


def test_offline_q_policy_completes_real_non_test_bbob_carps_episode(tmp_path: Path) -> None:
    """Load a checkpoint through the CARP-S policy bridge and finish one episode."""
    model_config = OfflineQModelConfig()
    torch.manual_seed(123)
    model = OfflineQNetwork(model_config)
    normalizer = fit_observation_normalizer(
        np.zeros((4, 13), dtype=np.float32),
        np.broadcast_to(np.linspace(0, 1, 5, dtype=np.float32)[:, None], (4, 5, 4)),
        train_dataset_sha256="synthetic-training-sha",
    )
    checkpoint = tmp_path / "smoke.pt"
    normalizer_path = tmp_path / "normalization_schema.json"
    torch.save(
        {
            "schema_version": "dacbo-offline-q-checkpoint-v1",
            "model_config": asdict(model_config),
            "model_state": model.state_dict(),
            "provenance": {"behavior_train_sha256": "synthetic-training-sha"},
        },
        checkpoint,
    )
    normalizer_path.write_text(json.dumps(normalizer.to_dict(), sort_keys=True), encoding="utf-8")
    env = real_structured_mixed_env(
        "bbob/2/3/0",
        18031,
        "wei",
        context_split="train",
        interaction_frequency=5,
    )
    try:
        observation, info = env.reset(seed=701)
        policy = OfflineQPolicy(
            env,
            checkpoint=str(checkpoint),
            normalizer=str(normalizer_path),
            checkpoint_sha256=file_sha256(checkpoint),
            normalizer_sha256=file_sha256(normalizer_path),
            deployment_head="long_q",
            interaction_frequency=5,
        )
        rewards = []
        while True:
            assert env.observation_space.contains(observation)
            action = policy(observation)
            assert 0 <= action < 5
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        assert rewards
        assert np.isfinite(rewards).all()
        assert info["task_id"] == "bbob/2/3/0"
        assert env.objective_reference is not None
        assert env.objective_reference.kind == "exact"
    finally:
        env.close()

    construction_env = real_structured_mixed_env(
        "bbob/2/3/0", 19, "wei", context_split="train", interaction_frequency=5
    )
    construction_env.reset()
    vector_env = DummyVecEnv([lambda: construction_env])
    config = OmegaConf.create(
        {
            "rl_algorithm_id": "double_dqn",
            "rl_algorithm": {
                "hyperparameters": {
                    "policy": "MultiInputPolicy",
                    "learning_rate": 1e-4,
                    "buffer_size": 32,
                    "learning_starts": 0,
                    "batch_size": 4,
                    "tau": 1.0,
                    "gamma": 1.0,
                    "train_freq": [1, "step"],
                    "gradient_steps": 1,
                    "target_update_interval": 2,
                    "exploration_fraction": 0.1,
                    "exploration_initial_eps": 0.0,
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
                "normalizer": str(normalizer_path),
                "checkpoint_sha256": file_sha256(checkpoint),
                "normalizer_sha256": file_sha256(normalizer_path),
            },
        }
    )
    model = build_sb3_algorithm(config, vector_env, tensorboard_log=str(tmp_path / "tensorboard"), model_seed=31)
    model_path = tmp_path / "offline_initialized_ddqn"
    model.save(model_path)
    vector_env.close()

    bridge_env = real_structured_mixed_env("bbob/2/3/0", 23, "wei", context_split="train", interaction_frequency=5)
    try:
        observation, _ = bridge_env.reset()
        bridge = SB3DiscretePolicy(
            bridge_env,
            str(model_path.with_suffix(".zip")),
            model_class=DoubleDQN,
            algorithm_id="double_dqn",
        )
        action = bridge(observation)
        _, reward, _, _, _ = bridge_env.step(action)
        assert 0 <= int(action) < 5
        assert np.isfinite(reward)
    finally:
        bridge_env.close()
