"""Real CARPS/BBOB-to-SB3 integration coverage."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from carps.utils.running import make_task
from dacboenv.experiment.ppo import make_env_factory
from dacboenv.optimizer import DACBOEnvOptimizer
from dacboenv.policy.sb3_model import ModelPolicy
from dacboenv.utils.carps_optimizer import get_bbob_n_trials, get_task_config
from hydra import compose, initialize_config_module
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


def _real_bbob_config(
    training_config: str = "structured_ppo_f1",
) -> DictConfig:
    """Compose a small but otherwise real structured BBOB environment."""
    with initialize_config_module(
        config_module="dacboenv.configs",
        version_base=None,
    ):
        cfg = compose(
            config_name=None,
            overrides=[f"+training={training_config}"],
        )

    cfg.seed = 0
    cfg.dacboenv.task_ids = ["bbob/2/3/0"]
    cfg.dacboenv.inner_seeds = [0]
    # Keep repeated resets from Gymnasium's checker inexpensive while retaining
    # the real IOH objective, CARPS adapter, SMAC facade, and surrogate path.
    initial_design = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.initial_design
    initial_design.n_configs = 2
    initial_design.max_ratio = 1.0
    return cfg


@pytest.mark.parametrize(
    ("training_config", "feature_key", "feature_shape"),
    [
        ("lcb_quantile_ppo_f1", "action_features", (5, 4)),
        ("ucb_quantile_ppo_f1", "action_features", (5, 4)),
        ("af_selection_ppo_f1", "af_action_features", (5, 10)),
    ],
)
def test_new_controller_configs_build_real_bbob_environments(
    training_config: str,
    feature_key: str,
    feature_shape: tuple[int, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydra, CARPS, SMAC, actions, and structured rows agree end to end."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config(training_config)
    env = make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path / training_config,
    )()

    try:
        observation, _info = env.reset()

        assert env.action_space.n == 5
        assert env.observation_space.contains(observation)
        assert observation[feature_key].shape == feature_shape
        assert np.isfinite(observation[feature_key]).all()

        next_observation, reward, terminated, truncated, _info = env.step(4)

        assert not terminated
        assert not truncated
        assert np.isfinite(float(reward))
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(next_observation[feature_key]).all()
        if feature_key == "action_features":
            acquisition_function = (
                env._smac_instance.intensifier.config_selector._acquisition_function
            )
            assert acquisition_function._posterior_quantile == pytest.approx(
                cfg.dacboenv.action_space_kwargs.quantile_levels[4],
            )
        else:
            assert env._action_space.selected_mode == "maximum_variance"
    finally:
        env.close()


def test_mixed_dimension_instance_set_rebuilds_each_bbob_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each selected task dimension gets its own CARP-S/SMAC trial horizon."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config()
    cfg.dacboenv.task_ids = ["bbob/2/3/0", "bbob/5/3/0"]
    cfg.dacboenv.instance_selector_class = {
        "_target_": "dacboenv.env.instance.RoundRobinInstanceSelector",
        "_partial_": True,
        "offset": 0,
    }
    env = make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path / "mixed-dimension-smac",
    )()

    try:
        expected_episodes = [
            ("bbob/2/3/0", 2),
            ("bbob/5/3/0", 5),
            ("bbob/2/3/0", 2),
        ]
        for task_id, dimension in expected_episodes:
            observation, _info = env.reset()

            assert env.current_task_id == task_id
            assert env._smac_instance._scenario.n_trials == get_bbob_n_trials(dimension)
            assert len(env._smac_instance._scenario.configspace) == dimension
            assert env.observation_space.contains(observation)
    finally:
        env.close()


def test_real_bbob_environment_trains_and_runs_as_carps_optimizer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise BBOB -> PPO artifact -> CARP-S optimizer deployment."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config()
    env = make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path / "smac",
    )()

    # This includes reset-seed determinism, space containment, and one real
    # environment transition. It must not be replaced with a toy environment.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Your observation action_features has an unconventional shape.*",
        )
        check_env(env, skip_render_check=True)

    vec_env = DummyVecEnv([lambda: env])
    try:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_kwargs={"net_arch": {"pi": [4], "vf": [4]}},
            seed=0,
            verbose=0,
        )
        model.learn(total_timesteps=2)

        model_path = tmp_path / "model"
        model.save(model_path)
        loaded = PPO.load(model_path, env=vec_env)

        observation = vec_env.reset()
        action, _state = loaded.predict(observation, deterministic=True)

        assert (tmp_path / "model.zip").is_file()
        assert action.dtype == np.int64
        assert action.shape == (1,)
        assert 0 <= int(action[0]) < 5
        for key, value in observation.items():
            assert value.dtype == np.float32
            assert np.isfinite(value).all(), key
    finally:
        vec_env.close()

    deployment_env = make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path / "carps-smac",
    )()
    task_cfg = get_task_config("bbob/2/3/0")
    task_cfg.seed = 0
    # Two initial-design evaluations plus two policy-controlled evaluations.
    task_cfg.task.optimization_resources.n_trials = 4
    task = make_task(task_cfg)
    optimizer = DACBOEnvOptimizer(
        task=task,
        dacboenv=deployment_env,
        seed=0,
        policy_class=ModelPolicy,
        policy_kwargs={
            "model": str(tmp_path / "model.zip"),
            "model_class": "stable_baselines3.PPO",
        },
    )

    try:
        incumbent = optimizer.run()
        _trial_info, trial_value = incumbent

        assert optimizer.trial_counter == 4
        assert deployment_env.get_n_finished_trials() == 4
        assert np.isfinite(float(trial_value.cost))
        assert (tmp_path / "DACBOEnvActions.jsonl").is_file()
        assert (tmp_path / "DACBOEnvLogs.jsonl").is_file()
    finally:
        deployment_env.close()
