import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

os.environ["DACBOENV"] = "BUCKET"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--reward", type=str, default="", help="Reward function to use: '', 'SQRT'")
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--fid", type=int, default=-1, help="Function ID")
parser.add_argument("--obs", type=str, default="", help="Observation")
args = parser.parse_args()

os.environ["REWARD"] = args.reward
os.environ["OBS"] = args.obs
os.environ["FID"] = str(args.fid)

SEED = args.seed
FID = args.fid

from dacboenv.dacboenv import DACBOEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ioh import get_problem, ProblemClass
from smac import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
import numpy as np
from functools import partial
import psutil
from dacboenv.utils.confidence_bound import UCB

D = 2
cs = ConfigurationSpace()
n_episodes = 150
n_workers = len(psutil.Process().cpu_affinity()) # Number of cores      
len_episode = 77
run_name = f"dqn_bucket_{SEED}_{FID}_{os.environ["OBS"]}"

for i in range(D):
    cs.add(
        CSH.UniformFloatHyperparameter(f"x{i}", lower=-5, upper=5, log=False)
    )

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

def evaluate(config_dict, seed):
    f = get_problem(FID, 0, D, ProblemClass.BBOB)
    x = list(config_dict.values())
    return f(x)

def create(seed):
    scenario = Scenario(
        configspace=cs,
        n_trials=len_episode,
        deterministic=True,
        seed=seed,
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
        env = DACBOEnv(create, seed=SEED + seed_offset)
        return env
    return _init

if __name__ == "__main__": 

    env_fns = [make_env(i + 1) for i in range(n_workers)]
    vec_env = SubprocVecEnv(env_fns)

    print("Evaluating random policy...")
    mean_reward, std_reward = evaluate_random_policy(DACBOEnv(create, seed=0), 2)
    print(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open(f"../training/results_{run_name}.txt", "w") as out:
        out.write(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    policy = "MlpPolicy" if os.environ["OBS"] == "SINGLE" else "MultiInputPolicy"

    model = DQN(
        policy,
        vec_env,
        verbose=1,
        seed=SEED,
        batch_size=n_workers * len_episode // 11,
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
