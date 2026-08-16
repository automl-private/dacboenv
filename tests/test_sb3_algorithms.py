"""Focused PPO/DQN/Double-DQN abstraction contracts."""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from dacboenv.experiment.collect_ppo import create_ppo_eval_configs
from dacboenv.experiment.ppo import ProtocolEvalCallback, resolve_training_schedule, run_step_zero_validation
from dacboenv.experiment.sb3_algorithms import (
    DQN_DISCRETE_ERROR,
    build_sb3_algorithm,
    validate_algorithm_action_space,
)
from dacboenv.policy.sb3_model import SB3DiscretePolicy
from dacboenv.rl.double_dqn import DoubleDQN, double_dqn_bootstrap, vanilla_dqn_bootstrap
from gymnasium import spaces
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv


class TinyDictEnv(gym.Env):
    """Deterministic five-action Dict environment for short SB3 tests."""

    def __init__(self, action_space: spaces.Space | None = None) -> None:
        self.observation_space = spaces.Dict(
            {
                "global_state": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
                "action_features": spaces.Box(-1, 1, shape=(5, 2), dtype=np.float32),
            }
        )
        self.action_space = action_space or spaces.Discrete(5)
        self.step_count = 0

    def observation(self) -> dict[str, np.ndarray]:
        """Return the current deterministic observation."""
        return {
            "global_state": np.asarray([self.step_count / 4, 0, 0], dtype=np.float32),
            "action_features": np.zeros((5, 2), dtype=np.float32),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):  # noqa: ARG002
        """Reset the deterministic counter."""
        super().reset(seed=seed)
        self.step_count = 0
        return self.observation(), {}

    def step(self, action: int):
        """Advance one transition."""
        self.step_count += 1
        return self.observation(), float(action == 2), self.step_count >= 4, False, {}

    def restart_fixed_instance_sequence(self) -> None:
        """Match the fixed-manifest DACBO validation interface."""
        self.step_count = 0


def config(training: str, algorithm: str | None = None):
    overrides = [f"+training={training}"]
    if algorithm is not None:
        overrides.append(f"rl_algorithm={algorithm}")
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        return compose(config_name=None, overrides=overrides)


@pytest.mark.parametrize(("algorithm", "expected"), [("dqn", DQN), ("double_dqn", DoubleDQN)])
def test_dict_observation_algorithms_train_save_and_load(algorithm: str, expected: type[DQN], tmp_path: Path) -> None:
    cfg = config("bbob_dqn_smoke", algorithm)
    env = DummyVecEnv([TinyDictEnv for _ in range(4)])
    model = build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path), model_seed=17)
    assert type(model) is expected
    model.learn(total_timesteps=16, progress_bar=False)
    assert model.num_timesteps == 16
    assert model.replay_buffer.size() == 4
    assert model._n_updates == 8
    observation = env.reset()
    tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        before = model.q_net(tensor).cpu().numpy()
    assert np.isfinite(before).all()
    destination = tmp_path / algorithm
    model.save(destination)
    loaded = expected.load(destination, env=env)
    with th.no_grad():
        after = loaded.q_net(tensor).cpu().numpy()
    np.testing.assert_array_equal(before, after)
    first = loaded.predict(observation, deterministic=True)[0]
    second = loaded.predict(observation, deterministic=True)[0]
    np.testing.assert_array_equal(first, second)
    env.close()


def test_double_dqn_target_differs_only_when_argmax_differs() -> None:
    online = th.tensor([[5.0, 1.0], [1.0, 4.0]])
    target = th.tensor([[2.0, 8.0], [3.0, 2.0]])
    np.testing.assert_array_equal(vanilla_dqn_bootstrap(target).numpy(), [[8.0], [3.0]])
    np.testing.assert_array_equal(double_dqn_bootstrap(online, target).numpy(), [[2.0], [2.0]])

    rewards = th.tensor([[1.0], [1.0]])
    dones = th.tensor([[0.0], [1.0]])
    double_target = rewards + (1 - dones) * double_dqn_bootstrap(online, target)
    np.testing.assert_array_equal(double_target.numpy(), [[3.0], [1.0]])
    np.testing.assert_array_equal(double_dqn_bootstrap(target, target), vanilla_dqn_bootstrap(target))


@pytest.mark.parametrize(
    "action_space",
    [spaces.Box(-1, 1, shape=(1,)), spaces.MultiDiscrete([5, 3]), spaces.MultiBinary(5)],
)
def test_off_policy_algorithms_reject_non_discrete_actions(action_space: spaces.Space) -> None:
    with pytest.raises(TypeError, match="currently require a Discrete"):
        validate_algorithm_action_space("dqn", action_space)
    with pytest.raises(TypeError, match="currently require a Discrete"):
        validate_algorithm_action_space("double_dqn", action_space)
    validate_algorithm_action_space("ppo", action_space)
    assert DQN_DISCRETE_ERROR.startswith("DQN and Double DQN")


def test_pinned_sb3_dict_n_step_is_rejected(tmp_path: Path) -> None:
    cfg = config("bbob_dqn_smoke")
    cfg.rl_algorithm.hyperparameters.n_steps = 5
    env = DummyVecEnv([TinyDictEnv])
    with pytest.raises(ValueError, match="does not support n-step replay with Dict"):
        build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path), model_seed=1)
    env.close()


@pytest.mark.parametrize(
    ("config_name", "algorithm_id"),
    [("dqn_1step", "dqn"), ("double_dqn_1step", "double_dqn")],
)
def test_explicit_one_step_algorithm_configs(config_name: str, algorithm_id: str) -> None:
    cfg = config("bbob_dqn_smoke", config_name)
    assert cfg.rl_algorithm_id == algorithm_id
    assert cfg.rl_algorithm.hyperparameters.n_steps == 1


def test_ppo_factory_and_schedule_preserve_existing_configuration(tmp_path: Path) -> None:
    cfg = config("bbob_ppo_pilot")
    schedule = resolve_training_schedule(cfg)
    assert (schedule.n_envs, schedule.n_steps, schedule.batch_size, schedule.total_timesteps) == (4, 2, 8, 16)
    env = DummyVecEnv([TinyDictEnv for _ in range(4)])
    model = build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path), model_seed=5)
    assert type(model) is PPO
    assert model.gamma == 1.0
    assert model.n_steps == 2
    assert model.batch_size == 8
    env.close()


def test_algorithm_metadata_survives_policy_bundle_export(tmp_path: Path) -> None:
    cfg = config("bbob_dqn_smoke")
    cfg.task_id = "fixture"
    cfg.experiment.total_timesteps = 16
    run_root = tmp_path / "run"
    (run_root / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, run_root / ".hydra" / "config.yaml")
    checkpoint = run_root / "validation" / "frequent" / "checkpoints" / "step_16_model"
    checkpoint.parent.mkdir(parents=True)
    env = DummyVecEnv([TinyDictEnv])
    model = build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path), model_seed=7)
    model.save(checkpoint)
    history = {
        "panel_id": "fixture-panel",
        "panel_hash": "fixture-hash",
        "checkpoints": [
            {
                "training_step": 16,
                "model_path": str(checkpoint.with_suffix(".zip")),
                "normalization_path": None,
                "scores": {"balanced": 0.0},
            }
        ],
    }
    (run_root / "validation" / "frequent" / "history.json").write_text(json.dumps(history), encoding="utf-8")
    output = tmp_path / "policy"
    inventory = tmp_path / "inventory.json"
    create_ppo_eval_configs(run_root, output, "final", inventory)
    payload = json.loads(inventory.read_text(encoding="utf-8"))
    assert payload["policies"][0]["algorithm_id"] == "dqn"
    assert payload["policies"][0]["algorithm_class"] == "stable_baselines3.DQN"
    policy_cfg = OmegaConf.load(payload["policies"][0]["config_path"])
    assert policy_cfg.policy_bundle.algorithm_id == "dqn"
    assert policy_cfg.optimizer.policy_kwargs.model_class == "stable_baselines3.DQN"
    assert policy_cfg.optimizer.policy_kwargs.algorithm_id == "dqn"
    env.close()


def test_policy_bridge_rejects_wrong_algorithm_metadata(tmp_path: Path) -> None:
    """An explicit algorithm ID cannot silently load a different class."""
    cfg = config("bbob_dqn_smoke")
    env = DummyVecEnv([TinyDictEnv])
    model = build_sb3_algorithm(cfg, env, tensorboard_log=str(tmp_path), model_seed=7)
    model_path = tmp_path / "dqn"
    model.save(model_path)
    bare_env = TinyDictEnv()
    with pytest.raises(ValueError, match=r"requires dacboenv\.rl\.double_dqn\.DoubleDQN"):
        SB3DiscretePolicy(
            bare_env,
            str(model_path.with_suffix(".zip")),
            model_class="stable_baselines3.DQN",
            algorithm_id="double_dqn",
        )
    env.close()
    bare_env.close()


@pytest.mark.parametrize("algorithm", ["dqn", "double_dqn"])
def test_off_policy_validation_saves_step_zero_and_frequent_checkpoints(algorithm: str, tmp_path: Path) -> None:
    """The common validation callback accepts either off-policy algorithm."""
    cfg = config("bbob_dqn_smoke", algorithm)
    training_env = DummyVecEnv([TinyDictEnv])
    evaluation_env = DummyVecEnv([TinyDictEnv])
    model = build_sb3_algorithm(cfg, training_env, tensorboard_log=str(tmp_path), model_seed=19)
    step_zero = run_step_zero_validation(
        model,
        training_env,
        evaluation_env,
        task_ids=["bbob/2/3/0"],
        inner_seeds=[1],
        save_path=tmp_path / algorithm / "step_zero",
        panel_id="unit-panel",
        panel_hash="unit-hash",
    )
    assert np.isfinite(step_zero.balanced_score)
    assert (tmp_path / algorithm / "step_zero" / "untrained_model.zip").is_file()
    callback = ProtocolEvalCallback(
        evaluation_env,
        n_eval_episodes=1,
        eval_freq=2,
        deterministic=True,
        log_path=str(tmp_path / algorithm / "frequent"),
        warn=False,
        manifest_task_ids=["bbob/2/3/0"],
        manifest_inner_seeds=[1],
        protocol_save_path=tmp_path / algorithm,
        panel_id="unit-panel",
        panel_hash="unit-hash",
    )
    model.learn(total_timesteps=4, callback=callback, progress_bar=False)
    assert callback.frequent_history
    assert all(entry["training_step"] > 0 for entry in callback.frequent_history)
    assert not (tmp_path / algorithm / "best_balanced_model.zip").exists()
    training_env.close()
    evaluation_env.close()
