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
from carps.utils.running import make_task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

# Register OmegaConf resolvers
import dacboenv  # noqa: F401
from dacboenv.utils.loggingutils import get_logger, maybe_remove_logs

if TYPE_CHECKING:
    from collections.abc import Callable

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import DACBOEnv

logger = get_logger("PPO")


# def evaluate_policy(env: DACBOEnv, policy: Policy, n_episodes: int | None = None):
#     instance_selector = RoundRobinInstanceSelector(
#         task_ids=env.instance_selector.task_ids,
#         seeds=env.instance_selector.seeds,
#         selector_seed=env.instance_selector.selector_seed
#     )
#     env.instance_selector = instance_selector
#     if n_episodes is None:
#         n_episodes = len(env.instance_selector.instances)
#     results = []
#     for _ in range(n_episodes):
#         episode_result = rollout(env=env, policy=policy)
#         results.append(episode_result)
#     rewards = [r["reward_mean"] for r in results]
#     mean_reward = np.mean(rewards)
#     std_reward = np.std(rewards)
#     return mean_reward, std_reward


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

    n_workers = cfg.experiment.n_workers
    task = make_task(cfg)
    len_episode = cfg.optimizer.n_steps

    n_episodes = cfg.experiment.n_episodes

    def make_env(cfg: DictConfig, seed_offset: int = 0) -> Callable:
        def _init() -> DACBOEnv:
            cfg.seed = cfg.seed + seed_offset
            task = make_task(cfg)
            return task.objective_function._env

        return _init

    env_fns = [make_env(cfg=cfg, seed_offset=i) for i in range(n_workers)]
    vec_env = SubprocVecEnv(env_fns)

    model: BaseAlgorithm = instantiate(cfg.optimizer)(
        env=vec_env, policy_kwargs=policy_kwargs, tensorboard_log=rundir / "tensorboard"
    )

    # TODO wrap in obs normalization

    logger.info("⚔ Start training...")
    model.learn(total_timesteps=n_workers * n_episodes * len_episode, progress_bar=True, tb_log_name="tb.log")
    model.save(rundir / "model")
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
