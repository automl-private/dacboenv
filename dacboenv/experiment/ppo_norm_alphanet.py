# flake8: noqa: E402
"""Train PPO on DACBOEnv."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


from typing import TYPE_CHECKING

import hydra
from carps.loggers.file_logger import get_run_directory
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

# Register OmegaConf resolvers
import dacboenv  # noqa: F401
from dacboenv.utils.loggingutils import maybe_remove_logs

if TYPE_CHECKING:
    from collections.abc import Callable

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import DACBOEnv

logger = get_logger("PPO")


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Train PPO on DACBOEnv."""
    logger.info(OmegaConf.to_yaml(cfg))

    rundir = get_run_directory()
    maybe_remove_logs(directory=None, overwrite=True, logfile="model.zip", logger=logger)

    # We need pure python types for sb3
    policy_kwargs = {}
    if hasattr(cfg.optimizer, "policy_kwargs"):
        policy_kwargs = OmegaConf.to_container(cfg=cfg.optimizer.policy_kwargs, resolve=True)
        del cfg.optimizer.policy_kwargs

    def make_env(cfg: DictConfig, seed_offset: int = 0) -> Callable:
        def _init() -> DACBOEnv:
            cfg.seed = cfg.seed + seed_offset
            task = make_task(cfg)
            return task.objective_function._env

        return _init

    n_workers = cfg.experiment.n_workers
    n_envs = n_workers
    task = make_task(cfg)

    n_episodes = cfg.experiment.n_episodes

    # Extract n_trials from inner opt
    env = make_env(cfg)()
    env.reset()
    inner_optimizer = env._carps_solver
    len_episode = inner_optimizer.task.optimization_resources.n_trials

    del env

    env_fns = [make_env(cfg=cfg, seed_offset=i) for i in range(n_workers)]
    vec_env = SubprocVecEnv(env_fns)

    # Obs normalization
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
    )

    model: BaseAlgorithm = instantiate(cfg.optimizer)(
        # policy=PerceptronPolicy,
        env=vec_env,
        policy_kwargs=policy_kwargs,
        tensorboard_log=rundir / "tensorboard",
        n_steps=len_episode,
        batch_size=n_workers * len_episode // 2,
    )
    logger.info(f"Model: {model.policy}")

    logger.info("⚔ Start training...")
    save_freq = n_workers * len_episode * 5
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1), save_path=str(rundir), save_vecnormalize=True
    )
    model.learn(
        total_timesteps=n_workers * n_episodes * len_episode,
        progress_bar=True,
        tb_log_name="tb_log",
        callback=checkpoint_callback,
    )
    model.save(rundir / "model")
    vec_env.save(rundir / "vecnormalize.pkl")
    logger.info("✅ Finished training.🥵")

    # Evaluate learned policy
    logger.info("🧐 Evaluating learned policy...")
    n_eval_episodes = len(task.objective_function._env.instance_selector.instances)
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=n_eval_episodes)
    logger.info(f"Learned policy reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    with open(rundir / "modeleval.txt", "a") as out:
        out.write(f"Learned policy reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")


if __name__ == "__main__":
    main()
