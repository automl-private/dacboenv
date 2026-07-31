"""Real CARPS/BBOB-to-SB3 integration coverage."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from carps.utils.running import make_task
from dacboenv.experiment.collect_ppo import create_ppo_eval_configs
from dacboenv.experiment.ppo import make_env_factory
from dacboenv.optimizer import DACBOEnvOptimizer
from dacboenv.utils.carps_optimizer import get_bbob_n_trials, get_task_config
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv


def _real_bbob_config(
    training_config: str = "structured_ppo_f1",
    *,
    true_regret: bool = False,
) -> DictConfig:
    """Compose a small but otherwise real structured BBOB environment."""
    overrides = [f"+training={training_config}"]
    if true_regret:
        overrides.append("+env/reward=true_regret_improvement")

    with initialize_config_module(
        config_module="dacboenv.configs",
        version_base=None,
    ):
        cfg = compose(
            config_name=None,
            overrides=overrides,
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
            acquisition_function = env._smac_instance.intensifier.config_selector._acquisition_function
            assert acquisition_function._posterior_quantile == pytest.approx(
                cfg.dacboenv.action_space_kwargs.quantile_levels[4],
            )
        else:
            assert env._action_space.selected_mode == "maximum_variance"
    finally:
        env.close()


def test_true_regret_reward_uses_live_bbob_optimum_on_every_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The privileged optimum is reward-only state from the active objective."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config("af_selection_ppo_f1", true_regret=True)
    cfg.dacboenv.task_ids = ["bbob/2/3/0", "bbob/2/4/1"]
    cfg.dacboenv.instance_selector_class = {
        "_target_": "dacboenv.env.instance.RoundRobinInstanceSelector",
        "_partial_": True,
        "offset": 0,
    }
    env = make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path / "true-regret-smac",
    )()

    try:
        observed_minima = []
        for expected_task_id in cfg.dacboenv.task_ids:
            observation, _info = env.reset()
            live_minimum = float(env._carps_solver.task.objective_function.f_min)

            assert env.current_task_id == expected_task_id
            assert env._objective_minimum == pytest.approx(live_minimum)
            assert env._reward._objective_minimum == pytest.approx(live_minimum)
            assert "objective_minimum" not in observation
            assert "true_regret" not in observation

            _next_observation, reward, terminated, truncated, _info = env.step(4)
            assert np.isfinite(float(reward))
            assert float(reward) >= 0.0
            assert not terminated
            assert not truncated
            observed_minima.append(live_minimum)

        assert observed_minima[0] != observed_minima[1]
    finally:
        env.close()


def test_mixed_dimension_instance_set_rebuilds_each_bbob_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each selected task dimension gets its own CARP-S/SMAC trial horizon."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config()
    cfg.dacboenv.task_ids = ["bbob/2/3/0", "bbob/4/3/0"]
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
            ("bbob/4/3/0", 4),
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


def test_selected_inner_seed_controls_real_smac_initial_design(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same inner seed replays Sobol independently of the worker seed."""
    monkeypatch.chdir(tmp_path)

    def build_seeded_episode(outer_seed: int, inner_seed: int, label: str) -> tuple[list[dict[str, float]], int]:
        cfg = _real_bbob_config("af_selection_ppo_f1", true_regret=True)
        cfg.seed = outer_seed
        cfg.dacboenv.inner_seeds = [inner_seed]
        env = make_env_factory(
            cfg,
            worker_id=0,
            output_directory=tmp_path / label,
        )()
        try:
            env.reset()
            scenario_seed = int(env._smac_instance._scenario.seed)
            assert env.current_seed == inner_seed
            assert env.instance == (inner_seed, "bbob/2/3/0")
            assert scenario_seed == inner_seed
            assert int(env._carps_solver.task.seed) == inner_seed
            assert env._smac_instance.intensifier.config_selector._random_design._seed == inner_seed
            initial_design = [
                dict(configuration)
                for configuration in env._smac_instance.intensifier.config_selector._initial_design_configs
            ]
            return initial_design, scenario_seed
        finally:
            env.close()

    first_design, first_seed = build_seeded_episode(outer_seed=17, inner_seed=123, label="first")
    replayed_design, replayed_seed = build_seeded_episode(outer_seed=91, inner_seed=123, label="replayed")
    different_design, different_seed = build_seeded_episode(outer_seed=17, inner_seed=124, label="different")

    assert first_seed == replayed_seed == 123
    assert different_seed == 124
    assert replayed_design == first_design
    assert different_design != first_design


def test_streamed_inner_seed_replays_real_bbob_and_resets_control_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One outer seed replays fresh episodes without mutable SMAC leakage."""
    monkeypatch.chdir(tmp_path)

    def collect_stream(outer_seed: int, label: str) -> list[tuple[int, list[dict[str, float]]]]:
        cfg = _real_bbob_config("af_selection_ppo_f1", true_regret=True)
        cfg.seed = outer_seed
        cfg.dacboenv.inner_seeds = [None]
        env = make_env_factory(
            cfg,
            worker_id=0,
            output_directory=tmp_path / label,
        )()
        episodes: list[tuple[int, list[dict[str, float]]]] = []
        try:
            for episode in range(2):
                env.reset()
                inner_seed = env.current_seed
                acquisition_function = env._smac_instance.intensifier.config_selector._acquisition_function
                assert env.instance == (inner_seed, "bbob/2/3/0")
                assert int(env._smac_instance._scenario.seed) == inner_seed
                assert int(env._carps_solver.task.seed) == inner_seed
                assert env._smac_instance.intensifier.config_selector._random_design._seed == inner_seed
                assert acquisition_function.mode == "expected_improvement"

                initial_design = [
                    dict(configuration)
                    for configuration in env._smac_instance.intensifier.config_selector._initial_design_configs
                ]
                episodes.append((inner_seed, initial_design))

                if episode == 0:
                    env.step(0)
                    assert acquisition_function.mode == "posterior_mean"
        finally:
            env.close()
        return episodes

    first_stream = collect_stream(outer_seed=17, label="stream-first")
    replayed_stream = collect_stream(outer_seed=17, label="stream-replayed")
    different_stream = collect_stream(outer_seed=18, label="stream-different")

    assert replayed_stream == first_stream
    assert first_stream[0][0] != first_stream[1][0]
    assert different_stream != first_stream


def test_real_bbob_environment_trains_and_runs_as_carps_optimizer(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise true-regret BBOB -> PPO export -> CARP-S deployment."""
    monkeypatch.chdir(tmp_path)
    cfg = _real_bbob_config("af_selection_ppo_f1", true_regret=True)
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
            message=r"Your observation .*action_features has an unconventional shape.*",
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

        run_directory = tmp_path / "runs" / "PPO-Structured-MLP" / "DACBO" / "training-task" / "0"
        config_path = run_directory / ".hydra" / "config.yaml"
        config_path.parent.mkdir(parents=True)
        OmegaConf.save(cfg, config_path)

        model_path = run_directory / "model"
        model.save(model_path)
        loaded = PPO.load(model_path, env=vec_env)

        observation = vec_env.reset()
        action, _state = loaded.predict(observation, deterministic=True)

        assert (run_directory / "model.zip").is_file()
        assert action.dtype == np.int64
        assert action.shape == (1,)
        assert 0 <= int(action[0]) < 5
        for key, value in observation.items():
            assert value.dtype == np.float32
            assert np.isfinite(value).all(), key
    finally:
        vec_env.close()

    exported_config_root = tmp_path / "exported-policy"
    create_ppo_eval_configs(tmp_path / "runs", configs_path=exported_config_root)
    exported_config_path = next(exported_config_root.rglob("seed0.yaml"))
    exported = OmegaConf.load(exported_config_path)

    task_cfg = get_task_config("bbob/2/3/0")
    task_cfg.seed = 0
    # Two initial-design evaluations plus two policy-controlled evaluations.
    task_cfg.task.optimization_resources.n_trials = 4
    task = make_task(task_cfg)

    runtime = OmegaConf.merge(
        {
            "task": {
                "name": task.name,
                "optimization_resources": {"n_trials": 4},
            },
            "seed": 0,
            "outdir": str(tmp_path / "carps-smac"),
        },
        exported,
    )
    deployment_env = instantiate(runtime.dacboenv)
    policy_class = instantiate(runtime.optimizer.policy_class)
    policy_kwargs = OmegaConf.to_container(
        runtime.optimizer.policy_kwargs,
        resolve=True,
    )
    assert isinstance(policy_kwargs, dict)
    assert list(runtime.dacboenv.reward_keys) == ["true_regret_improvement"]
    assert runtime.dacboenv.evaluation_mode is False
    assert policy_kwargs["model"] == str((run_directory / "model.zip").resolve())

    optimizer = DACBOEnvOptimizer(
        task=task,
        dacboenv=deployment_env,
        seed=0,
        policy_class=policy_class,
        policy_kwargs=policy_kwargs,
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
