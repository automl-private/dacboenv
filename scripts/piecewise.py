import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="Seed")
parser.add_argument("--fid", type=int, default=-1, help="Function ID")
args = parser.parse_args()

SEED = args.seed
FID = args.fid

import numpy as np
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from ioh import get_problem, ProblemClass
from smac import Scenario
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.facade.algorithm_configuration_facade import AlgorithmConfigurationFacade
from smac.callback import Callback
from smac.intensifier.intensifier import Intensifier
from dacboenv.utils.confidence_bound import UCB

D = 2

n_episodes = 1000  
len_episode = 77
run_name = f"piecewise_{SEED}_{FID}"

# Inner BO loop

cs_inner = ConfigurationSpace()

for i in range(D):
    cs_inner.add(
        CSH.UniformFloatHyperparameter(f"x{i}", lower=-5, upper=5, log=False)
    )

def evaluate_inner(config_dict, seed):
    f = get_problem(FID, 0, D, ProblemClass.BBOB)
    x = list(config_dict.values())
    return f(x)

# Outer BO loop

NUM_SEGMENTS = 4
MIN_BOUND = 0
MAX_BOUND = 1

splits = np.linspace(1, len_episode, NUM_SEGMENTS + 1, dtype=int)
cs_outer = ConfigurationSpace()

for i in range(NUM_SEGMENTS + 1):
    cs_outer.add(CSH.UniformFloatHyperparameter(f"splitx{i}", lower=MIN_BOUND, upper=MAX_BOUND, log=False))
    
def evaluate_outer(config_dict, seed):
    
    splity = list(config_dict.values())
    
    class PiecewiseCallback(Callback):
        def __init__(self):
            super().__init__()
            
        def on_iteration_start(self, smbo):
            
            t = len(smbo.runhistory)
            val = np.interp(t, splits, splity)
            
            setattr(
                smbo._intensifier._config_selector._acquisition_function,
                "_beta", # UCB
                val ** 2,
            )
            
            return super().on_iteration_start(smbo)
    
    scenario_inner = Scenario(
        configspace=cs_inner,
        n_trials=len_episode,
        deterministic=True,
        seed=seed,
        name=f"{run_name}_inner"
    )

    smac_inner = BlackBoxFacade(
        scenario_inner,
        evaluate_inner,
        overwrite=True,
        logging_level=9999,
        acquisition_function=UCB(update_beta=False),
        callbacks=[PiecewiseCallback()]
    )
    
    incumbent = smac_inner.optimize()
    return smac_inner.validate(incumbent)

if __name__ == "__main__":
    
    scenario_outer = Scenario(
        configspace=cs_outer,
        n_trials=n_episodes, # 500
        deterministic=False,
        seed=SEED,
        name=f"{run_name}_outer"
    )

    intensifier = Intensifier(
        scenario=scenario_outer,
        max_config_calls=5,
        seed=SEED
    )

    smac_outer = AlgorithmConfigurationFacade(
        scenario_outer,
        evaluate_outer,
        overwrite=True,
        intensifier=intensifier,
    )

    incumbent = smac_outer.optimize()
    
    default_cost = smac_outer.validate(cs_outer.get_default_configuration())
    print(f"Default cost: {default_cost}")
    
    print(incumbent)
    incumbent_cost = smac_outer.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")