"""Contracts for the structured SB3 training and baseline matrix."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from carps.utils.running import make_task as make_carps_task
from dacboenv.env.instance import RandomInstanceSelector
from dacboenv.experiment import (
    ppo as ppo_module,
    ppo_norm,
    ppo_norm_alphanet,
)
from dacboenv.experiment.ppo import resolve_training_schedule
from dacboenv.utils.carps_optimizer import get_task_config, is_bbob_task_id
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from smac.acquisition.function.expected_improvement import EI
from smac.facade.blackbox_facade import BlackBoxFacade
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn


class TinyStructuredEnv(Env):
    """Small Dict/Discrete environment for an end-to-end runner smoke test."""

    def __init__(self) -> None:
        self.observation_space = Dict(
            {
                "global_state": Box(-1.0, 1.0, shape=(13,), dtype=np.float32),
                "action_features": Box(-1.0, 1.0, shape=(5, 4), dtype=np.float32),
            }
        )
        self.action_space = Discrete(5)
        self.instance = (0, "tiny")
        self._step_count = 0

    def _observation(self) -> dict[str, np.ndarray]:
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
        """Reset the two-step toy episode."""
        super().reset(seed=seed)
        self._step_count = 0
        return self._observation(), {}

    def step(
        self,
        action: int,  # noqa: ARG002
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Advance one toy transition."""
        self._step_count += 1
        terminated = self._step_count == 2
        return self._observation(), 1.0, terminated, False, {}


def compose_config(*overrides: str) -> DictConfig:
    """Compose one package config without invoking a training process."""
    with initialize_config_module(
        config_module="dacboenv.configs",
        version_base=None,
    ):
        return compose(config_name=None, overrides=list(overrides))


def test_smac_logging_path_is_concrete_in_saved_hydra_config() -> None:
    """Hydra serialization retains a usable repository-relative logging path."""
    cfg = compose_config("+env=base", "+env/opt=base")
    logging_path = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.logging_level._args_[0]

    assert logging_path == "dacboenv/configs/logging/smac_internal.yaml"
    assert "${dacboenv_config:" not in OmegaConf.to_yaml(cfg, resolve=False)


@pytest.mark.parametrize(
    ("config_name", "frequency", "total_timesteps", "n_updates", "collected"),
    [
        ("structured_ppo_f1", 1, 409600, 200, 409600),
        ("structured_ppo_f5", 5, 81920, 40, 81920),
        ("structured_ppo_f10", 10, 40960, 20, 40960),
    ],
)
def test_structured_training_configs(
    config_name: str,
    frequency: int,
    total_timesteps: int,
    n_updates: int,
    collected: int,
) -> None:
    """Each frequency composes to the accepted PPO setup and fair BO budget."""
    cfg = compose_config(f"+training={config_name}")
    schedule = resolve_training_schedule(cfg)

    assert cfg.dacboenv.interaction_frequency == frequency
    assert cfg.action_space_id == f"WEI-discrete-f{frequency}"
    assert cfg.dacboenv.action_space_kwargs.param_levels == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert cfg.experiment.n_workers == 32
    assert cfg.optimizer.n_steps == 64
    assert cfg.optimizer.batch_size == 256
    assert cfg.optimizer.n_epochs == 5
    assert cfg.optimizer.gamma == pytest.approx(1.0)
    assert cfg.optimizer.gae_lambda == pytest.approx(0.95)
    assert cfg.optimizer.clip_range == pytest.approx(0.15)
    assert cfg.optimizer.ent_coef == pytest.approx(0.01)
    assert cfg.optimizer.vf_coef == pytest.approx(0.5)
    assert cfg.optimizer.target_kl == pytest.approx(0.015)
    assert list(cfg.optimizer.policy_kwargs.net_arch.pi) == [64, 64, 64]
    assert list(cfg.optimizer.policy_kwargs.net_arch.vf) == [64, 64, 64]
    assert not cfg.optimizer.policy_kwargs.share_features_extractor
    assert cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.intensifier.max_config_calls == 1
    assert len(cfg.dacboenv.task_ids) == 12
    assert list(cfg.dacboenv.inner_seeds) == [None]
    assert len(cfg.experiment.validation.task_ids) * len(cfg.experiment.validation.inner_seeds) == 40
    assert (
        cfg.experiment.validation.instance_selector_class._target_ == "dacboenv.env.instance.RoundRobinInstanceSelector"
    )
    assert schedule.rollout_size == 2048
    assert schedule.total_timesteps == total_timesteps
    assert schedule.n_updates == n_updates
    assert schedule.collected_timesteps == collected
    assert schedule.collected_timesteps * frequency == 409600
    assert cfg.experiment.checkpoint_freq == 20480 // frequency
    assert cfg.experiment.validation.eval_freq == 20480 // frequency


@pytest.mark.parametrize(
    (
        "training_prefix",
        "action_prefix",
        "action_target",
        "observation_space_id",
        "observation_keys",
        "acquisition_target",
        "levels",
        "run_directory",
    ),
    [
        (
            "lcb_quantile_ppo",
            "LCB-quantile-discrete",
            "dacboenv.env.action.PosteriorQuantileActionSpace",
            "structured-quantile",
            ["global_state", "action_features"],
            "dacboenv.utils.confidence_bound.LCB",
            [0.5, 0.25, 0.1, 0.025, 0.005],
            "runs_lcb_quantile",
        ),
        (
            "ucb_quantile_ppo",
            "UCB-quantile-discrete",
            "dacboenv.env.action.PosteriorQuantileActionSpace",
            "structured-quantile",
            ["global_state", "action_features"],
            "dacboenv.utils.confidence_bound.UCB",
            [0.5, 0.75, 0.9, 0.975, 0.995],
            "runs_ucb_quantile",
        ),
        (
            "af_selection_ppo",
            "AF-select",
            "dacboenv.env.action.PosteriorModeActionSpace",
            "structured-af-selection",
            ["global_state", "af_action_features"],
            "dacboenv.utils.posterior_decision.PosteriorModeAcquisition",
            None,
            "runs_af_selection",
        ),
    ],
)
def test_structured_controller_training_matrices_compose(
    training_prefix: str,
    action_prefix: str,
    action_target: str,
    observation_space_id: str,
    observation_keys: list[str],
    acquisition_target: str,
    levels: list[float] | None,
    run_directory: str,
) -> None:
    """New controllers reuse the WEI curriculum and schedule without merging stale kwargs."""
    for frequency in (1, 5, 10):
        cfg = compose_config(f"+training={training_prefix}_f{frequency}")
        schedule = resolve_training_schedule(cfg)

        assert cfg.action_space_id == f"{action_prefix}-f{frequency}"
        assert cfg.dacboenv.action_space_class._target_ == action_target
        assert cfg.observation_space_id == observation_space_id
        assert list(cfg.dacboenv.observation_keys) == observation_keys
        assert cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.acquisition_function._target_ == acquisition_target
        assert cfg.dacboenv.interaction_frequency == frequency
        assert cfg.baserundir == run_directory
        assert cfg.optimizer_id == "PPO-Structured-MLP"
        assert len(cfg.dacboenv.task_ids) == 12
        assert list(cfg.dacboenv.inner_seeds) == [None]
        assert len(cfg.experiment.validation.task_ids) * len(cfg.experiment.validation.inner_seeds) == 40
        assert schedule.rollout_size == 2048
        assert schedule.collected_timesteps * frequency == 409600
        assert cfg.experiment.checkpoint_freq == 20480 // frequency
        assert cfg.experiment.validation.eval_freq == 20480 // frequency

        if levels is None:
            assert dict(cfg.dacboenv.action_space_kwargs) == {}
            acquisition_cfg = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.acquisition_function
            assert acquisition_cfg.mode == "expected_improvement"
            assert acquisition_cfg.xi == pytest.approx(0.0)
            assert acquisition_cfg.lower_quantile == pytest.approx(0.1)
        else:
            assert list(cfg.dacboenv.action_space_kwargs.quantile_levels) == levels
            acquisition_cfg = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.acquisition_function
            assert acquisition_cfg.update_beta is False
            assert acquisition_cfg.nu == pytest.approx(1.0)


@pytest.mark.parametrize(
    "training_prefix",
    [
        "structured_ppo",
        "lcb_quantile_ppo",
        "ucb_quantile_ppo",
        "af_selection_ppo",
    ],
)
@pytest.mark.parametrize("frequency", [1, 5, 10])
def test_true_regret_training_override_composes_for_every_controller(
    training_prefix: str,
    frequency: int,
) -> None:
    """The BBOB-only reward is usable across the structured PPO matrix."""
    cfg = compose_config(
        f"+training={training_prefix}_f{frequency}",
        "+env/reward=true_regret_improvement",
    )

    assert cfg.reward_id == "true-regret-improvement"
    assert list(cfg.dacboenv.reward_keys) == ["true_regret_improvement"]
    assert "_Rtrue-regret-improvement_" in cfg.task_id
    assert all(is_bbob_task_id(str(task_id)) for task_id in cfg.dacboenv.task_ids)
    assert all(is_bbob_task_id(str(task_id)) for task_id in cfg.experiment.validation.task_ids)
    assert cfg.optimizer.gamma == pytest.approx(1.0)
    assert cfg.experiment.vecnormalize is False


def test_training_random_selector_does_not_inherit_round_robin_offset() -> None:
    """The composed worker selector is directly instantiable as configured."""
    cfg = compose_config("+training=structured_ppo_f1")
    selector_cfg = cfg.dacboenv.instance_selector_class

    assert "offset" not in selector_cfg
    selector_factory = instantiate(selector_cfg)
    selector = selector_factory(
        task_ids=list(cfg.dacboenv.task_ids),
        seeds=list(cfg.dacboenv.inner_seeds),
    )

    assert isinstance(selector, RandomInstanceSelector)


def test_training_workers_get_independent_outer_seeds_and_stream_inner_seeds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run/worker pairs derive stable streams without additive collisions."""
    cfg = compose_config("+training=structured_ppo_f1")
    cfg.seed = 7
    captured_configs: list[DictConfig] = []

    def fake_make_task(worker_cfg: DictConfig) -> SimpleNamespace:
        captured_configs.append(worker_cfg)
        return SimpleNamespace(
            objective_function=SimpleNamespace(_env=TinyStructuredEnv()),
        )

    monkeypatch.setattr(ppo_module, "make_task", fake_make_task)
    for worker_id in (0, 1, 31):
        ppo_module.make_env_factory(
            cfg,
            worker_id=worker_id,
            output_directory=tmp_path / f"worker_{worker_id}",
        )()

    worker_ids = [0, 1, 31]
    expected_seeds = [ppo_module.derive_worker_seed(7, worker_id) for worker_id in worker_ids]
    assert [int(worker_cfg.seed) for worker_cfg in captured_configs] == expected_seeds
    assert len(set(expected_seeds)) == len(expected_seeds)
    assert ppo_module.derive_worker_seed(0, 1) != ppo_module.derive_worker_seed(1, 0)
    five_run_seeds = {
        ppo_module.derive_worker_seed(run_seed, worker_id) for run_seed in range(5) for worker_id in range(32)
    }
    assert len(five_run_seeds) == 5 * 32
    assert all(
        list(worker_cfg.dacboenv.inner_seeds) == list(cfg.dacboenv.inner_seeds) for worker_cfg in captured_configs
    )
    assert list(cfg.dacboenv.inner_seeds) == [None]
    assert all(list(worker_cfg.dacboenv.task_ids) == list(cfg.dacboenv.task_ids) for worker_cfg in captured_configs)


def test_vector_env_stages_the_same_hierarchical_worker_seeds() -> None:
    """SB3's first reset must not replace the factory's worker seeds."""
    vec_env = ppo_module.HierarchicalSeedDummyVecEnv([TinyStructuredEnv, TinyStructuredEnv])
    try:
        assert vec_env.seed(7) == [
            ppo_module.derive_worker_seed(7, 0),
            ppo_module.derive_worker_seed(7, 1),
        ]
    finally:
        vec_env.close()


def test_actor_and_critic_architectures_are_independently_overridable() -> None:
    """Hydra overrides can change pi and vf layouts independently."""
    cfg = compose_config(
        "+training=structured_ppo_f1",
        "optimizer.policy_kwargs.net_arch.pi=[128,64]",
        "optimizer.policy_kwargs.net_arch.vf=[256,128,64]",
    )

    assert list(cfg.optimizer.policy_kwargs.net_arch.pi) == [128, 64]
    assert list(cfg.optimizer.policy_kwargs.net_arch.vf) == [256, 128, 64]


def test_sb3_builds_separate_actor_and_critic_networks_from_config() -> None:
    """The structured config reaches SB3 as separate pi and vf MLPs."""
    cfg = compose_config("+training=structured_ppo_f1")
    observation_space = Dict(
        {
            "global_state": Box(-1.0, 1.0, shape=(13,), dtype=np.float32),
            "action_features": Box(-1.0, 1.0, shape=(5, 4), dtype=np.float32),
        }
    )
    policy = MultiInputActorCriticPolicy(
        observation_space=observation_space,
        action_space=Discrete(5),
        lr_schedule=lambda _progress: 2.0e-4,
        net_arch=dict(cfg.optimizer.policy_kwargs.net_arch),
        share_features_extractor=False,
    )

    actor_linears = [module for module in policy.mlp_extractor.policy_net if isinstance(module, nn.Linear)]
    critic_linears = [module for module in policy.mlp_extractor.value_net if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in actor_linears] == [
        (33, 64),
        (64, 64),
        (64, 64),
    ]
    assert [(layer.in_features, layer.out_features) for layer in critic_linears] == [
        (33, 64),
        (64, 64),
        (64, 64),
    ]
    assert policy.action_net.out_features == 5
    assert policy.value_net.out_features == 1


@pytest.mark.parametrize(
    ("training_config", "feature_key", "feature_shape", "flat_width"),
    [
        ("lcb_quantile_ppo_f1", "action_features", (5, 4), 33),
        ("ucb_quantile_ppo_f1", "action_features", (5, 4), 33),
        ("af_selection_ppo_f1", "af_action_features", (5, 10), 63),
    ],
)
def test_new_controller_spaces_build_five_action_sb3_policies(
    training_config: str,
    feature_key: str,
    feature_shape: tuple[int, int],
    flat_width: int,
) -> None:
    """Every new structured Dict reaches a categorical five-action PPO head."""
    cfg = compose_config(f"+training={training_config}")
    observation_space = Dict(
        {
            "global_state": Box(-1.0, 1.0, shape=(13,), dtype=np.float32),
            feature_key: Box(
                -1.0,
                1.0,
                shape=feature_shape,
                dtype=np.float32,
            ),
        }
    )
    policy = MultiInputActorCriticPolicy(
        observation_space=observation_space,
        action_space=Discrete(5),
        lr_schedule=lambda _progress: 2.0e-4,
        net_arch=dict(cfg.optimizer.policy_kwargs.net_arch),
        share_features_extractor=False,
    )

    actor_linears = [module for module in policy.mlp_extractor.policy_net if isinstance(module, nn.Linear)]
    critic_linears = [module for module in policy.mlp_extractor.value_net if isinstance(module, nn.Linear)]
    assert actor_linears[0].in_features == flat_width
    assert critic_linears[0].in_features == flat_width
    assert policy.action_net.out_features == 5


def test_ppo_runner_smoke_without_vecnormalize(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The no-normalization path trains, saves, evaluates, and closes cleanly."""
    cfg = compose_config("+training=structured_ppo_f1")
    cfg.experiment.n_workers = 1
    cfg.experiment.total_timesteps = 2
    cfg.experiment.checkpoint_freq = 2
    cfg.experiment.progress_bar = False
    cfg.experiment.validation.enabled = False
    cfg.optimizer.n_steps = 2
    cfg.optimizer.batch_size = 2
    cfg.optimizer.n_epochs = 1
    cfg.optimizer.policy_kwargs.net_arch.pi = [4]
    cfg.optimizer.policy_kwargs.net_arch.vf = [4]
    cfg.dacboenv.task_ids = ["tiny"]
    cfg.dacboenv.inner_seeds = [0]

    monkeypatch.setattr(
        ppo_module,
        "make_env_factory",
        lambda *_args, **_kwargs: TinyStructuredEnv,
    )
    monkeypatch.setattr(ppo_module, "get_run_directory", lambda: tmp_path)
    monkeypatch.setattr(ppo_module, "maybe_remove_logs", lambda **_kwargs: None)

    ppo_module.main.__wrapped__(cfg)

    assert (tmp_path / "model.zip").is_file()
    assert not (tmp_path / "vecnormalize.pkl").exists()
    assert (tmp_path / "modeleval.txt").is_file()
    action_log = tmp_path / "tensorboard" / "actions.csv"
    assert action_log.is_file()
    assert "env_0/bo_evaluations" in action_log.read_text(encoding="utf-8").splitlines()[0]


def test_normalized_training_env_is_frozen_for_final_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Final evaluation must not update training observation statistics."""
    cfg = compose_config("+training=structured_ppo_f1")
    cfg.experiment.n_workers = 1
    cfg.experiment.total_timesteps = 2
    cfg.experiment.checkpoint_freq = 2
    cfg.experiment.progress_bar = False
    cfg.experiment.vecnormalize = True
    cfg.experiment.validation.enabled = False
    cfg.optimizer.n_steps = 2
    cfg.optimizer.batch_size = 2
    cfg.optimizer.n_epochs = 1
    cfg.optimizer.policy_kwargs.net_arch.pi = [4]
    cfg.optimizer.policy_kwargs.net_arch.vf = [4]
    cfg.dacboenv.task_ids = ["tiny"]
    cfg.dacboenv.inner_seeds = [0]

    monkeypatch.setattr(
        ppo_module,
        "make_env_factory",
        lambda *_args, **_kwargs: TinyStructuredEnv,
    )
    monkeypatch.setattr(ppo_module, "get_run_directory", lambda: tmp_path)
    monkeypatch.setattr(ppo_module, "maybe_remove_logs", lambda **_kwargs: None)

    def assert_frozen_eval(
        _model: Any,
        env: Any,
        **_kwargs: Any,
    ) -> tuple[float, float]:
        assert isinstance(env, VecNormalize)
        assert not env.training
        assert not env.norm_reward
        return 0.0, 0.0

    monkeypatch.setattr(ppo_module, "evaluate_policy", assert_frozen_eval)
    ppo_module.main.__wrapped__(cfg)

    assert (tmp_path / "vecnormalize.pkl").is_file()


@pytest.mark.parametrize("optimizer_config", ["ppo_old", "ppo_alphanet2", "ppo_alphanet3"])
def test_legacy_ppo_configs_receive_runtime_schedule(
    optimizer_config: str,
) -> None:
    """Historical normalized entry points retain their config interface."""
    cfg = compose_config(
        f"+opt={optimizer_config}",
        "+dacboenv.task_ids=[bbob/2/3/0]",
    )

    ppo_module.populate_legacy_schedule(cfg)
    schedule = resolve_training_schedule(cfg)

    assert cfg.optimizer.n_steps == 77
    assert cfg.optimizer.batch_size == 77
    assert cfg.experiment.total_timesteps == 1540
    assert cfg.experiment.checkpoint_freq == 770
    assert schedule.rollout_size == 154


@pytest.mark.parametrize("compatibility_module", [ppo_norm, ppo_norm_alphanet])
def test_legacy_normalized_entrypoints_use_shared_runner(
    compatibility_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy module CLIs retain their names without retaining divergent PPO code."""
    cfg = compose_config("+training=structured_ppo_f1")
    cfg.experiment.vecnormalize = False
    normalization_values: list[bool] = []
    monkeypatch.setattr(
        compatibility_module,
        "run",
        lambda run_cfg: normalization_values.append(bool(run_cfg.experiment.vecnormalize)),
    )

    compatibility_module.main.__wrapped__(cfg)

    assert normalization_values == [True]


def test_validation_factory_replaces_training_random_selector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Holdout evaluation visits contexts round-robin instead of with replacement."""
    cfg = compose_config("+training=structured_ppo_f1")
    validation_cfg = cfg.experiment.validation
    captured_configs: list[DictConfig] = []

    def fake_make_task(worker_cfg: DictConfig) -> SimpleNamespace:
        captured_configs.append(worker_cfg)
        return SimpleNamespace(
            objective_function=SimpleNamespace(_env=TinyStructuredEnv()),
        )

    monkeypatch.setattr(ppo_module, "make_task", fake_make_task)
    factory = ppo_module.make_env_factory(
        cfg,
        worker_id=0,
        output_directory=tmp_path,
        task_ids=list(validation_cfg.task_ids),
        inner_seeds=list(validation_cfg.inner_seeds),
        instance_set_id=str(validation_cfg.instance_set_id),
        instance_selector_cfg=validation_cfg.instance_selector_class,
    )

    factory()

    selector_cfg = captured_configs[0].dacboenv.instance_selector_class
    assert selector_cfg._target_ == "dacboenv.env.instance.RoundRobinInstanceSelector"
    assert selector_cfg.offset == 0


def test_all_random_and_static_baseline_cells_compose() -> None:
    """The config grid contains 3 random and 15 static frequency-alpha cells."""
    frequencies = {"f1": 1, "f5": 5, "f10": 10}
    levels = {
        "level_000": (0, 0.0),
        "level_025": (1, 0.25),
        "level_050": (2, 0.5),
        "level_075": (3, 0.75),
        "level_100": (4, 1.0),
    }

    for frequency_name, frequency in frequencies.items():
        random_cfg = compose_config(
            "+baseline=structured_random",
            f"+env/interaction_freq={frequency_name}",
        )
        assert random_cfg.policy_id == "Random"
        assert random_cfg.dacboenv.interaction_frequency == frequency
        assert random_cfg.experiment.n_episodes == 40

        for level_name, (action_idx, alpha) in levels.items():
            static_cfg = compose_config(
                "+baseline=structured_static",
                f"policy/static/wei_discrete={level_name}",
                f"+env/interaction_freq={frequency_name}",
            )
            assert static_cfg.dacboenv.interaction_frequency == frequency
            assert static_cfg.static_action_index == action_idx
            assert static_cfg.alpha == pytest.approx(alpha)
            assert static_cfg.optimizer.policy_kwargs.par_val == action_idx


def test_default_policy_composes_as_noop_smac() -> None:
    """Default SMAC uses the structured stack without changing WEI alpha."""
    cfg = compose_config(
        "+eval=base",
        "+env=base",
        "+env/opt=base",
        "+env/action=wei_alpha_discrete",
        "+env/interaction_freq=f1",
        "+env/obs=structured",
        "+env/reward=reference_free_improvement",
        "+policy=defaultaction",
        "+seed=0",
    )

    assert cfg.optimizer_id == "DefaultPolicy"
    assert cfg.policy_id == "DefaultPolicy"
    assert cfg.optimizer.policy_class._target_ == "dacboenv.policy.noop.NoOpPolicy"
    acquisition_function = instantiate(
        cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.acquisition_function,
    )
    assert acquisition_function._alpha == pytest.approx(0.5)


def test_default_policy_wei_is_blackbox_ei_up_to_scale() -> None:
    """WEI at alpha 0.5 has the same maximizer as BlackBoxFacade's EI."""

    class PredictiveModel:
        @staticmethod
        def predict_marginalized(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            means = 0.25 * np.sum(values, axis=1, keepdims=True)
            variances = 0.2 + np.square(values[:, :1] + 0.3)
            return means, variances

    candidates = np.asarray(
        [
            [-2.0, 0.5],
            [-0.5, -1.0],
            [0.0, 0.0],
            [0.75, 1.25],
            [2.0, -0.25],
        ],
    )
    model = PredictiveModel()
    native_ei = BlackBoxFacade.get_acquisition_function(
        scenario=SimpleNamespace(),
    )
    weighted_ei = instantiate(
        {
            "_target_": "dacboenv.utils.weighted_expected_improvement.WEI",
        },
    )
    native_ei.update(model=model, eta=0.4)
    weighted_ei.update(model=model, eta=0.4)

    assert isinstance(native_ei, EI)
    native_values = native_ei._compute(candidates)
    weighted_values = weighted_ei._compute(candidates)
    np.testing.assert_allclose(weighted_values, 0.5 * native_values)
    assert int(np.argmax(weighted_values)) == int(np.argmax(native_values))


def test_baseline_test_override_covers_every_final_context() -> None:
    """Switching to the frozen test split automatically expands to 195 runs."""
    cfg = compose_config(
        "+baseline=structured_random",
        "instance_sets@evaluation_instances=bbob_test_holdout",
    )

    assert cfg.instance_set_id == "bbob-test-holdout"
    assert cfg.experiment.n_episodes == 195


@pytest.mark.parametrize(
    ("task_id", "dimension", "n_trials"),
    [
        ("bbob/3/2/1", 3, math.ceil(20 + 40 * math.sqrt(3))),
        ("bbob/7/21/0", 7, math.ceil(20 + 40 * math.sqrt(7))),
    ],
)
def test_chat_bbob_dimensions_are_built_without_generated_carps_configs(
    task_id: str,
    dimension: int,
    n_trials: int,
) -> None:
    """Dimensions 3 and 7 work even though CARPS pre-generates powers of two."""
    cfg = get_task_config(task_id)

    assert cfg.task.metadata.dimensions == dimension
    assert cfg.task.optimization_resources.n_trials == n_trials
    assert len(cfg.task.input_space.configuration_space.d.hyperparameters) == dimension
    assert cfg.task.objective_function.instance == int(task_id.rsplit("/", maxsplit=1)[1])


def test_dynamic_carps_bbob_config_supports_validation_dimension() -> None:
    """CARP-S can compose and instantiate the dimension-three holdout tasks."""
    cfg = compose_config(
        "+task/BBOB=dynamic",
        "+seed=100",
        "bbob.dimension=3",
        "bbob.function_id=2",
        "bbob.instance_id=1",
    )

    task = make_carps_task(cfg)

    assert task.name == "bbob/3/2/1"
    assert task.seed == 100
    assert task.metadata.dimensions == 3
    assert task.optimization_resources.n_trials == 90
    assert len(task.input_space.configuration_space) == 3
