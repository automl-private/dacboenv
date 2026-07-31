"""Regression tests for selecting and exporting trained PPO policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dacboenv.experiment.collect_ppo import (
    _uses_structured_reference_free_mdp,
    _uses_structured_training_mdp,
    create_ppo_eval_configs,
    gather_trained_ppo,
)
from dacboenv.policy.sb3_model import ModelPolicy
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO


class TinyPolicyEnv(Env):
    """Minimal structured environment for testing the SB3-to-CARPS handoff."""

    def __init__(self) -> None:
        self.observation_space = Dict(
            {
                "global_state": Box(-1.0, 1.0, shape=(13,), dtype=np.float32),
                "action_features": Box(
                    -1.0,
                    1.0,
                    shape=(5, 4),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Discrete(5)

    def _observation(self) -> dict[str, np.ndarray]:
        """Return one valid structured observation."""
        return {
            "global_state": np.zeros(13, dtype=np.float32),
            "action_features": np.zeros((5, 4), dtype=np.float32),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset the one-step environment."""
        super().reset(seed=seed)
        return self._observation(), {}

    def step(
        self,
        action: int,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Finish the one-step environment."""
        return self._observation(), 0.0, True, False, {}


def _write_run_config(
    run_directory: Path,
    *,
    structured: bool = True,
    vecnormalize: bool = False,
    true_regret: bool = False,
) -> DictConfig:
    """Create the minimal saved Hydra config consumed by the collector."""
    structured_reward_id = "true-regret-improvement" if true_regret else "reference-free-improvement"
    structured_reward_key = "true_regret_improvement" if true_regret else "reference_free_improvement"
    cfg = OmegaConf.create(
        {
            "optimizer_id": "PPO-Structured-MLP",
            "task_id": "structured-task",
            "seed": 7,
            "observation_space_id": "structured" if structured else "sawei",
            "reward_id": (structured_reward_id if structured or true_regret else "symlogregret-reference"),
            "experiment": {"vecnormalize": vecnormalize},
            "dacboenv": {
                "task_ids": ["training-task"],
                "inner_seeds": [0],
                "reward_keys": ([structured_reward_key] if structured or true_regret else ["symlogregret"]),
            },
        }
    )
    config_path = run_directory / ".hydra" / "config.yaml"
    config_path.parent.mkdir(parents=True)
    OmegaConf.save(cfg, config_path)
    return cfg


def test_gather_prefers_validation_model_with_legacy_fallbacks(
    tmp_path: Path,
) -> None:
    """Each run contributes its validation best, final, or newest checkpoint."""
    best_run = tmp_path / "PPO" / "DACBO" / "best-task" / "1"
    _write_run_config(best_run)
    (best_run / "validation").mkdir()
    (best_run / "validation" / "best_model.zip").touch()
    (best_run / "model.zip").touch()
    (best_run / "rl_model_200_steps.zip").touch()

    final_run = tmp_path / "PPO" / "DACBO" / "final-task" / "2"
    _write_run_config(final_run)
    (final_run / "model.zip").touch()
    (final_run / "rl_model_300_steps.zip").touch()

    checkpoint_run = tmp_path / "PPO" / "DACBO" / "checkpoint-task" / "3"
    _write_run_config(checkpoint_run)
    (checkpoint_run / "rl_model_100_steps.zip").touch()
    (checkpoint_run / "rl_model_900_steps.zip").touch()
    (checkpoint_run / "rl_model_invalid_steps.zip").touch()

    # A zip outside a Hydra run is not a trained run and must not be exported.
    stray_model = tmp_path / "unrelated" / "validation" / "best_model.zip"
    stray_model.parent.mkdir(parents=True)
    stray_model.touch()

    assert set(gather_trained_ppo(tmp_path)) == {
        (best_run / "validation" / "best_model.zip").resolve(),
        (final_run / "model.zip").resolve(),
        (checkpoint_run / "rl_model_900_steps.zip").resolve(),
    }


@pytest.mark.parametrize(
    ("structured", "true_regret"),
    [(True, False), (True, True), (False, True)],
)
def test_create_eval_config_uses_best_model_and_training_mdp_semantics(
    tmp_path: Path,
    structured: bool,
    true_regret: bool,
) -> None:
    """Potential rewards export without changing training MDP timing."""
    run_directory = tmp_path / "runs" / "PPO" / "DACBO" / "task" / "7"
    _write_run_config(
        run_directory,
        structured=structured,
        vecnormalize=True,
        true_regret=true_regret,
    )
    best_model = run_directory / "validation" / "best_model.zip"
    best_model.parent.mkdir()
    best_model.touch()
    normalization_wrapper = run_directory / "vecnormalize.pkl"
    normalization_wrapper.touch()

    configs_path = tmp_path / "policy-configs"
    create_ppo_eval_configs(tmp_path / "runs", configs_path=configs_path)

    generated_path = configs_path / "PPO-Structured-MLP" / "structured-task" / "seed7.yaml"
    generated = OmegaConf.load(generated_path)

    assert generated.optimizer.policy_kwargs.model == str(best_model.resolve())
    assert generated.optimizer.policy_kwargs.normalization_wrapper == str(normalization_wrapper)
    assert generated.dacboenv.evaluation_mode is False
    assert generated.dacboenv.terminate_after_reference_performance_reached is False


def test_generated_policy_config_loads_through_carps_policy_factory(
    tmp_path: Path,
) -> None:
    """The exported artifact loads via the factory used by DACBOEnvOptimizer."""
    run_directory = tmp_path / "runs" / "PPO" / "DACBO" / "task" / "7"
    _write_run_config(run_directory)
    best_model = run_directory / "validation" / "best_model.zip"
    best_model.parent.mkdir()

    training_env = TinyPolicyEnv()
    model = PPO(
        "MultiInputPolicy",
        training_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [4], "vf": [4]}},
    )
    model.save(best_model)
    training_env.close()

    configs_path = tmp_path / "policy-configs"
    create_ppo_eval_configs(tmp_path / "runs", configs_path=configs_path)
    generated = OmegaConf.load(configs_path / "PPO-Structured-MLP" / "structured-task" / "seed7.yaml")

    policy_factory = instantiate(generated.optimizer.policy_class)
    policy_kwargs = OmegaConf.to_container(
        generated.optimizer.policy_kwargs,
        resolve=True,
    )
    assert isinstance(policy_kwargs, dict)

    evaluation_env = TinyPolicyEnv()
    policy = policy_factory(env=evaluation_env, **policy_kwargs)
    observation, _ = evaluation_env.reset()
    action = policy(observation)

    assert isinstance(policy, ModelPolicy)
    assert evaluation_env.action_space.contains(action)
    policy._vec_env.close()


@pytest.mark.parametrize(
    ("observation_space_id", "reward_keys", "expected"),
    [
        ("structured", ["reference_free_improvement"], True),
        ("structured-quantile", ["reference_free_improvement"], True),
        ("structured-af-selection", ["reference_free_improvement"], True),
        ("structured", ["true_regret_improvement"], True),
        ("structured-quantile", ["true_regret_improvement"], True),
        ("structured-af-selection", ["true_regret_improvement"], True),
        ("structured", ["symlogregret"], False),
        ("sawei", ["reference_free_improvement"], False),
        ("sawei", ["true_regret_improvement"], True),
    ],
)
def test_structured_training_mdp_detection_is_narrow(
    observation_space_id: str,
    reward_keys: list[str],
    expected: bool,
) -> None:
    """Legacy reference-based policies retain their established eval mode."""
    cfg = DictConfig(
        {
            "observation_space_id": observation_space_id,
            "reward_id": "other",
            "dacboenv": {"reward_keys": reward_keys},
        }
    )

    assert _uses_structured_training_mdp(cfg) is expected
    assert _uses_structured_reference_free_mdp(cfg) is expected
