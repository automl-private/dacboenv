import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from dacboenv.dacboenv import DACBOEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ioh import get_problem, ProblemClass
from smac import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
import numpy as np
from functools import partial
import psutil

D = 2
SEED = 1
cs = ConfigurationSpace()
n_episodes = 38
n_workers = len(psutil.Process().cpu_affinity()) # Number of cores      
len_episode = 77

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
    f = get_problem(1, 1, D, ProblemClass.BBOB)
    x = list(config_dict.values())
    return f(x)

def create(offset):
    scenario = Scenario(
        configspace=cs,
        n_trials=len_episode,
        deterministic=True,
        seed=SEED + offset,
        name="test_single"
    )

    smac = BlackBoxFacade(
        scenario,
        evaluate,
        overwrite=True,
        logging_level=9999
    )
    return smac

def make_env(seed_offset=0):
    def _init():
        env = DACBOEnv(smac_instance_factory=partial(create, seed_offset), reward_keys=["trajectory_auc_alt"], rho=0.0)
        return env
    return _init

if __name__ == "__main__": 

    env_fns = [make_env(i) for i in range(n_workers)] 
    vec_env = SubprocVecEnv(env_fns)

    print("Evaluating random policy...")
    mean_reward, std_reward = evaluate_random_policy(DACBOEnv(smac_instance_factory=partial(create, -1), reward_keys=["trajectory_auc_alt"], rho=0.0), 2)
    print(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open("results_single.txt", "w") as out:
        out.write(f"Random policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")

    model = PPO(
        "MultiInputPolicy",
        vec_env,
        verbose=1,
        seed=SEED,
        n_steps=len_episode,
        batch_size=n_workers * len_episode // 11,
        n_epochs=4
    )
    
    print("START TRAINING")
    model.learn(total_timesteps=n_workers * n_episodes * len_episode, progress_bar=True)
    model.save("ppo_dacbo_single")
    print("FINISHED TRAINING")

    # Evaluate learned policy
    print("Evaluating learned policy...")
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=2)
    print(f"Learned policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    with open("results_single.txt", "a") as out:
        out.write(f"Learned policy mean reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
