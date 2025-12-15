import os
import hydra
from omegaconf import DictConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["DACBOENV"] = ""
os.environ["OBS"] = ""

import numpy as np
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ioh import get_problem, ProblemClass
from smac import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from functools import partial
from dacboenv.utils.confidence_bound import UCB
from dacboenv.dacboenv import DACBOEnv

observation_keys = ["budget_percentage", "inc_improvement_scaled", "has_categorical_hps", "knn_difference", "ubr_difference", "knn_entropy", "ubr"]

def evaluate_random_policy(env, n_episodes):
    rewards = []
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    return mean_reward, std_reward

@hydra.main(version_base=None, config_path="../dacboenv/configs/perceptron", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    
    SEED = cfg.seed
    RUN_ID = cfg.experiment.experiment_id
    FID, D = map(int, cfg.experiment.task_id.split("_"))
    
    os.environ["FID"] = str(FID)
    
    n_workers = cfg.experiment.n_workers
    len_episode = int(np.ceil(20 + 40 * np.sqrt(D)))
    
    n_episodes = cfg.experiment.n_episodes
    run_name = f"ppo_perceptron_{SEED}__{RUN_ID}"
    
    def evaluate(config_dict, seed):
        f = get_problem(FID, 0, D, ProblemClass.BBOB)
        x = list(config_dict.values())
        return f(x)

    cs = ConfigurationSpace()
    for i in range(D):
        cs.add(
            CSH.UniformFloatHyperparameter(f"x{i}", lower=-5, upper=5, log=False)
        )
    
    def create(offset):
        scenario = Scenario(
            configspace=cs,
            n_trials=len_episode,
            deterministic=True,
            seed=SEED + offset,
            name=run_name
        )

        smac = BlackBoxFacade(
            scenario,
            evaluate,
            overwrite=True,
            logging_level=9999,
            acquisition_function=UCB(update_beta=False)
        )
        return smac

    def make_env(seed_offset=0):
        def _init():
            env = DACBOEnv(create, seed=SEED + seed_offset, action_mode="parameter", observation_keys=observation_keys)
            return env
        return _init

    env_fns = [make_env(i) for i in range(n_workers)] 
    vec_env = SubprocVecEnv(env_fns)

    print("Evaluating random policy...")
    mean_reward, std_reward = evaluate_random_policy(DACBOEnv(create, seed=0, action_mode="parameter", observation_keys=observation_keys), 2)
    print(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open(f"../training/results_{run_name}.txt", "w") as out:
        out.write(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    # Enforce perceptron aka GLM aka no hidden layers
    policy_kwargs = dict(net_arch = dict(pi=[], vf=[]))

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        seed=SEED,
        n_steps=len_episode,
        batch_size=n_workers * len_episode // 2,
        n_epochs=3,
        tensorboard_log=f"../training/dacbo_{run_name}_tensorboard/"
    )
    
    print("START TRAINING")
    model.learn(total_timesteps=n_workers * n_episodes * len_episode, progress_bar=True, tb_log_name=f"dacbo_{run_name}")
    model.save(f"../training/dacbo_{run_name}")
    print("FINISHED TRAINING")

    # Evaluate learned policy
    print("Evaluating learned policy...")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
    print(f"Learned policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open(f"../training/results_{run_name}.txt", "a") as out:
        out.write(f"Learned policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

if __name__ == "__main__":
    main()
