import os
import hydra
from omegaconf import DictConfig

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

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

from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# XXX: change inner evals based on D -> YAHPO formula

@hydra.main(version_base=None, config_path="../dacboenv/configs/piecewise", config_name="base.yaml")
def main(cfg: DictConfig) -> None:
    
    SEED = cfg.seed
    RUN_ID = cfg.experiment.experiment_id
    
    instances = cfg.experiment.instances
    n_workers = cfg.experiment.n_workers
    
    n_episodes = cfg.experiment.n_episodes * len(instances) * n_workers
    run_name = f"piecewise_{SEED}__{RUN_ID}"
    
    NUM_SEGMENTS = 4
    MIN_BOUND = 0
    MAX_BOUND = 1 
    
    splitsx = np.linspace(0, 1, NUM_SEGMENTS + 1, dtype=float)

    # Inner BO loop
    def evaluate_outer(config_dict, seed, instance):
        
        fid, d = map(int, instance.split("_")) # Instance = FID_D
        
        #len_episode = int(np.ceil(20 + 40 * np.sqrt(d)))
        len_episode = 77
        
        splits = splitsx * len_episode

        cs_inner = ConfigurationSpace()

        for i in range(d):
            cs_inner.add(
                CSH.UniformFloatHyperparameter(f"x{i}", lower=-5, upper=5, log=False)
            )
        
        def evaluate_inner(config_dict, seed):
            f = get_problem(fid, 0, d, ProblemClass.BBOB)
            x = list(config_dict.values())
            return f(x)    
        
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

    # Outer BO loop

    cs_outer = ConfigurationSpace()

    for i in range(NUM_SEGMENTS + 1):
        cs_outer.add(CSH.UniformFloatHyperparameter(f"splitx{i}", lower=MIN_BOUND, upper=MAX_BOUND, log=False))
    
    scenario_kwargs = dict(
        configspace=cs_outer,
        n_trials=n_episodes * len(instances) * n_workers, 
        deterministic=False,
        seed=SEED,
        name=f"{run_name}_outer",
        instances=instances,
        instance_features={instance: [i] for i, instance in enumerate(instances)}, # Default instance features
    )
    
    if n_workers > 1:
        scenario_kwargs["n_workers"] = n_workers + 1
        
        cluster = SLURMCluster(
            queue="normal",                         # Name of the partition
            cores=1,                                # CPU cores requested
            memory="8 GB",                          # RAM requested
            walltime="48:00:00",                    # Walltime limit for a runner job. 
            processes=1,                            # Number of processes per worker
            log_directory=f"tmp/smac_dask_slurm/{SEED}",    # Logging directory
            nanny=False,                            # False unless you want to use pynisher
            worker_extra_args=[
                "--worker-port",                    # Worker port range 
                "60010:60100"],                     # Worker port range 
            scheduler_options={
                "port": 60001 + SEED,                      # Main Job Port
                "dashboard_address": 40550 + SEED
            },
        )
        cluster.scale(jobs=n_workers + 1)

        # Dask creates n_workers jobs on the cluster which stay open.
        client = Client(
            address=cluster,
        )

        # Dask waits for n_workers workers to be created
        client.wait_for_workers(n_workers=n_workers + 1)

    scenario_outer = Scenario(**scenario_kwargs)

    intensifier = Intensifier(
        scenario=scenario_outer,
        max_config_calls=len(instances) * n_workers,
        seed=SEED
    )

    smac_kwargs = dict(
        scenario=scenario_outer,
        target_function=evaluate_outer,
        overwrite=True,
        intensifier=intensifier,
    )

    if n_workers > 1:
        smac_kwargs["dask_client"] = client

    smac_outer = AlgorithmConfigurationFacade(**smac_kwargs)

    incumbent = smac_outer.optimize()
    
    default_cost = smac_outer.validate(cs_outer.get_default_configuration())
    print(f"Default cost: {default_cost}")
    
    print(incumbent)
    incumbent_cost = smac_outer.validate(incumbent)
    print(f"Incumbent cost: {incumbent_cost}")
    
    if n_workers > 1:
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()