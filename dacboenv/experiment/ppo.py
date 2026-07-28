"""Train Stable-Baselines3 PPO on DACBOEnv."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from carps.loggers.file_logger import get_run_directory
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

# Register OmegaConf resolvers.
import dacboenv  # noqa: F401
from dacboenv.experiment.ppo_utils import ActionLoggingCallback
from dacboenv.utils.carps_optimizer import get_task_config
from dacboenv.utils.loggingutils import maybe_remove_logs

if TYPE_CHECKING:
    from collections.abc import Callable

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import DACBOEnv

logger = get_logger("PPO")


@dataclass(frozen=True)
class TrainingSchedule:
    """Resolved PPO sampling schedule."""

    n_envs: int
    n_steps: int
    rollout_size: int
    batch_size: int
    total_timesteps: int
    n_updates: int
    collected_timesteps: int


def resolve_training_schedule(cfg: DictConfig) -> TrainingSchedule:
    """Resolve and validate the SB3 rollout schedule from configuration."""
    n_envs = int(cfg.experiment.n_workers)
    n_steps = int(cfg.optimizer.n_steps)
    batch_size = int(cfg.optimizer.batch_size)
    total_timesteps = int(cfg.experiment.total_timesteps)

    if n_envs <= 0 or n_steps <= 0 or batch_size <= 0 or total_timesteps <= 0:
        raise ValueError("n_workers, n_steps, batch_size, and total_timesteps must all be positive.")

    rollout_size = n_envs * n_steps
    if batch_size > rollout_size:
        raise ValueError(f"batch_size ({batch_size}) exceeds rollout size ({rollout_size}).")
    if rollout_size % batch_size:
        raise ValueError(
            "The rollout size must be divisible by batch_size to avoid a "
            f"truncated PPO minibatch, got {rollout_size} % {batch_size}."
        )

    n_updates = math.ceil(total_timesteps / rollout_size)
    return TrainingSchedule(
        n_envs=n_envs,
        n_steps=n_steps,
        rollout_size=rollout_size,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        n_updates=n_updates,
        collected_timesteps=n_updates * rollout_size,
    )


def populate_legacy_schedule(cfg: DictConfig) -> None:
    """Fill schedule fields omitted by the historical normalized PPO configs.

    The former entry points derived these values at runtime from the first
    configured CARPS task. Keeping that behavior here lets those module CLIs
    use the corrected shared runner without requiring config changes.
    """
    optimizer = cfg.optimizer
    experiment = cfg.experiment
    missing_n_steps = optimizer.get("n_steps") is None
    missing_batch_size = optimizer.get("batch_size") is None
    missing_total_timesteps = experiment.get("total_timesteps") is None
    missing_checkpoint_freq = experiment.get("checkpoint_freq") is None
    if not any(
        (
            missing_n_steps,
            missing_batch_size,
            missing_total_timesteps,
            missing_checkpoint_freq,
        )
    ):
        return

    n_workers = int(experiment.n_workers)
    if missing_n_steps:
        task_ids = list(cfg.dacboenv.task_ids)
        if not task_ids:
            raise ValueError("At least one task ID is required to infer PPO n_steps.")
        task_cfg = get_task_config(str(task_ids[0]))
        n_steps = int(task_cfg.task.optimization_resources.n_trials)
    else:
        n_steps = int(optimizer.n_steps)

    rollout_size = n_workers * n_steps
    if missing_batch_size:
        # Retain the historical half-rollout minibatch where possible while
        # ensuring the stricter shared runner never creates a partial batch.
        batch_size = max(rollout_size // 2, 1)
        while rollout_size % batch_size:
            batch_size -= 1
    else:
        batch_size = int(optimizer.batch_size)

    if missing_total_timesteps:
        n_episodes = int(experiment.n_episodes)
        total_timesteps = rollout_size * n_episodes
    else:
        total_timesteps = int(experiment.total_timesteps)

    checkpoint_freq = rollout_size * 5 if missing_checkpoint_freq else int(experiment.checkpoint_freq)

    with open_dict(cfg):
        optimizer.n_steps = n_steps
        optimizer.batch_size = batch_size
        experiment.total_timesteps = total_timesteps
        experiment.checkpoint_freq = checkpoint_freq

    logger.info(
        "Filled legacy PPO schedule: "
        f"n_steps={n_steps}, batch_size={batch_size}, "
        f"total_timesteps={total_timesteps}, checkpoint_freq={checkpoint_freq}."
    )


def _copy_config(cfg: DictConfig) -> DictConfig:
    """Create an independent OmegaConf tree for one environment worker."""
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected the root training configuration to be a mapping.")
    return copied


def make_env_factory(
    cfg: DictConfig,
    *,
    worker_id: int,
    output_directory: Path,
    task_ids: list[str] | None = None,
    inner_seeds: list[int] | None = None,
    instance_set_id: str | None = None,
    instance_selector_cfg: DictConfig | None = None,
) -> Callable[[], DACBOEnv]:
    """Build an isolated DACBO environment factory for a vector worker."""

    def _init() -> DACBOEnv:
        config = _copy_config(cfg)
        config.seed = int(cfg.seed) + worker_id

        if task_ids is not None:
            config.dacboenv.task_ids = list(task_ids)
        if inner_seeds is not None:
            config.dacboenv.inner_seeds = list(inner_seeds)
        if instance_set_id is not None:
            config.instance_set_id = instance_set_id
        if instance_selector_cfg is not None:
            selector = OmegaConf.to_container(instance_selector_cfg, resolve=True)
            if not isinstance(selector, dict):
                raise TypeError("An instance selector config must be a mapping.")
            config.dacboenv.instance_selector_class = selector

        selector_cfg = config.dacboenv.instance_selector_class
        selector_target = str(selector_cfg.get("_target_", "")) if isinstance(selector_cfg, DictConfig) else ""
        if selector_target.endswith("RoundRobinInstanceSelector"):
            selector_cfg.offset = worker_id

        config.dacboenv.optimizer_cfg.smac_cfg.scenario.output_directory = str(output_directory)

        task = make_task(config)
        return task.objective_function._env

    return _init


def _make_vec_env(
    factories: list[Callable[[], DACBOEnv]],
    *,
    start_method: str | None,
) -> VecEnv:
    """Use subprocess workers when useful and a local vector env otherwise."""
    if len(factories) == 1:
        return DummyVecEnv(factories)
    return SubprocVecEnv(factories, start_method=start_method)


def _policy_kwargs_and_optimizer_cfg(cfg: DictConfig) -> tuple[dict[str, Any], DictConfig]:
    """Separate SB3 policy kwargs without mutating Hydra's saved config."""
    converted = OmegaConf.to_container(cfg.optimizer, resolve=True)
    if not isinstance(converted, dict):
        raise TypeError("optimizer must be a mapping.")

    raw_policy_kwargs = converted.pop("policy_kwargs", {})
    if not isinstance(raw_policy_kwargs, dict):
        raise TypeError("optimizer.policy_kwargs must be a mapping.")
    policy_kwargs: dict[str, Any] = raw_policy_kwargs

    optimizer_cfg = OmegaConf.create(converted)
    if not isinstance(optimizer_cfg, DictConfig):
        raise TypeError("optimizer must be a mapping.")
    return policy_kwargs, optimizer_cfg


def _callback_frequency(transition_frequency: int, n_envs: int) -> int:
    """Convert an SB3 transition interval to vector callback calls."""
    if transition_frequency <= 0:
        raise ValueError("Callback transition frequencies must be positive.")
    return max(transition_frequency // n_envs, 1)


def run(cfg: DictConfig) -> None:  # noqa: PLR0915
    """Train and validate PPO on the configured BO-MDP."""
    logger.info(OmegaConf.to_yaml(cfg))

    rundir = Path(get_run_directory())
    maybe_remove_logs(directory=None, overwrite=True, logfile="model.zip", logger=logger)
    populate_legacy_schedule(cfg)
    schedule = resolve_training_schedule(cfg)
    policy_kwargs, optimizer_cfg = _policy_kwargs_and_optimizer_cfg(cfg)

    logger.info(
        "PPO schedule: "
        f"{schedule.n_envs} envs x {schedule.n_steps} steps = "
        f"{schedule.rollout_size} transitions/update; "
        f"{schedule.n_updates} updates collect {schedule.collected_timesteps} "
        f"transitions for requested {schedule.total_timesteps}."
    )

    start_method = cfg.experiment.get("start_method", None)
    training_factories = [
        make_env_factory(
            cfg,
            worker_id=worker_id,
            output_directory=rundir / "smac3_output" / "train" / f"worker_{worker_id}",
        )
        for worker_id in range(schedule.n_envs)
    ]
    vec_env: VecEnv = _make_vec_env(training_factories, start_method=start_method)
    eval_env: VecEnv | None = None

    try:
        if cfg.experiment.vecnormalize:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
            )

        model: BaseAlgorithm = instantiate(optimizer_cfg)(
            env=vec_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(rundir / "tensorboard"),
        )
        logger.info(f"Model: {model.policy}")

        callbacks: list[BaseCallback] = [
            CheckpointCallback(
                save_freq=_callback_frequency(
                    int(cfg.experiment.checkpoint_freq),
                    schedule.n_envs,
                ),
                save_path=str(rundir),
                save_vecnormalize=bool(cfg.experiment.vecnormalize),
            ),
            ActionLoggingCallback(
                n_envs=schedule.n_envs,
                csv_path=str(rundir / "tensorboard" / "actions.csv"),
            ),
        ]

        validation_cfg = cfg.experiment.get("validation", None)
        eval_callback: EvalCallback | None = None
        if validation_cfg is not None and bool(validation_cfg.get("enabled", True)):
            validation_workers = int(validation_cfg.get("n_workers", 1))
            if validation_workers != 1:
                raise ValueError(
                    "Validation currently requires n_workers=1 so round-robin "
                    "evaluation covers every holdout context exactly once."
                )
            validation_factories = [
                make_env_factory(
                    cfg,
                    worker_id=worker_id,
                    output_directory=rundir / "smac3_output" / "validation" / f"worker_{worker_id}",
                    task_ids=list(validation_cfg.task_ids),
                    inner_seeds=list(validation_cfg.inner_seeds),
                    instance_set_id=str(validation_cfg.instance_set_id),
                    instance_selector_cfg=validation_cfg.instance_selector_class,
                )
                for worker_id in range(validation_workers)
            ]
            eval_env = _make_vec_env(validation_factories, start_method=start_method)
            if cfg.experiment.vecnormalize:
                eval_env = VecNormalize(
                    eval_env,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                )
            eval_callback = EvalCallback(
                eval_env,
                n_eval_episodes=int(validation_cfg.n_eval_episodes),
                eval_freq=_callback_frequency(
                    int(validation_cfg.eval_freq),
                    schedule.n_envs,
                ),
                deterministic=True,
                log_path=str(rundir / "validation"),
                best_model_save_path=str(rundir / "validation"),
                warn=False,
            )
            callbacks.append(eval_callback)

        logger.info("⚔ Start training...")
        model.learn(
            total_timesteps=schedule.total_timesteps,
            progress_bar=bool(cfg.experiment.get("progress_bar", True)),
            tb_log_name="tb_log",
            callback=callbacks,
        )
        model.save(rundir / "model")
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(rundir / "vecnormalize.pkl"))
        logger.info("✅ Finished training.")

        logger.info("🧐 Evaluating learned policy...")
        final_eval_env = eval_env if eval_env is not None else vec_env
        if eval_env is not None and isinstance(vec_env, VecNormalize):
            sync_envs_normalization(vec_env, eval_env)
        if isinstance(final_eval_env, VecNormalize):
            final_eval_env.training = False
            final_eval_env.norm_reward = False

        if validation_cfg is not None and bool(validation_cfg.get("enabled", True)):
            n_eval_episodes = int(validation_cfg.n_eval_episodes)
            evaluation_label = "validation"
        else:
            n_eval_episodes = len(cfg.dacboenv.task_ids) * len(cfg.dacboenv.inner_seeds)
            evaluation_label = "training"

        mean_reward, std_reward = evaluate_policy(
            model,
            final_eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        logger.info(f"Learned policy {evaluation_label} reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        with (rundir / "modeleval.txt").open("a", encoding="utf-8") as out:
            out.write(f"Learned policy {evaluation_label} reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    finally:
        if eval_env is not None:
            eval_env.close()
        vec_env.close()


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the shared Stable-Baselines3 PPO runner."""
    run(cfg)


if __name__ == "__main__":
    main()
