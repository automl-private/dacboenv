"""Optimize DACBO env objective function with CMA-ES."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
from carps.loggers.file_logger import convert_trials
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
from carps.utils.trials import TrialInfo, TrialValue
from ConfigSpace import Configuration
from dask.base import compute
from dask.delayed import delayed
from dask.distributed import Client, LocalCluster, SpecCluster
from dask_jobqueue.slurm import SLURMCluster
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import (
    print as printr,
)

from dacboenv.utils.reference_performance import is_slurm_cluster

if TYPE_CHECKING:
    from cmaes import CMA

logger = get_logger("opt_via_ea")


def worker(x: list[float], config: dict[str, Any]) -> dict[str, Any]:
    """
    Example worker function to evaluate a task.
    Replace with your actual computation.
    """
    cfg = OmegaConf.create(config["cfg"])
    # The instance is selected randomly
    assert (
        cfg.dacboenv.instance_selector._target_ == "dacboenv.env.instance.RandomInstanceSelector"
    ), "Pass dacboenv.env.instance.RandomInstanceSelector!"
    task = make_task(cfg)
    seed, instance = task.objective_function._env.instance
    trial_info = TrialInfo(
        config=Configuration(configuration_space=task.objective_function.configspace, vector=np.array(x)),
        # By setting a seed and instance the instance set in dacboenv will be replaced.
        # This is okay at this point as we create a new env for each new objective function evaluation.
        seed=seed,
        instance=instance,
    )
    # trial_value = task.objective_function.evaluate(trial_info=trial_info) # TODO uncomment
    trial_value = TrialValue(cost=234)
    n_trials = config["n_generation"]
    n_function_calls = None
    info = convert_trials(n_trials, trial_info, trial_value, n_function_calls)
    info["worker_idx"] = config["worker_idx"]
    return info


def run_parallel(
    client: Client,
    cluster: SpecCluster,
    inputs: list[list[float]],
    config: dict[str, Any],
    n_workers: int = 4,
) -> list[dict[str, Any]]:
    """Run tasks in parallel using Dask."""
    printr(f"Running {len(inputs)} tasks on {n_workers} workers using Dask")

    # Wrap tasks with dask.delayed
    delayed_tasks = []
    for i, x in enumerate(inputs):
        _cfg = config.copy()
        _cfg["worker_idx"] = i
        delayed_tasks.append(delayed(worker)(x, _cfg))

    # Compute results in parallel
    results = compute(*delayed_tasks)

    client.close()
    cluster.close()

    return list(results)


def setup_client(n_workers: int, seed: int) -> tuple[SpecCluster, Client]:
    """Setup client and cluster, either local or slurm."""
    if True:
        # TODO: scenario_kwargs["n_workers"] = n_workers + 1
        if is_slurm_cluster():
            cluster = SLURMCluster(
                queue="normal",  # Name of the partition
                cores=1,  # CPU cores requested
                memory="8 GB",  # RAM requested
                walltime="48:00:00",  # Walltime limit for a runner job.
                processes=1,  # Number of processes per worker
                log_directory=f"tmp/smac_dask_slurm/{seed}",  # Logging directory
                nanny=False,  # False unless you want to use pynisher
                worker_extra_args=[
                    "--worker-port",  # Worker port range
                    "60010:60100",
                ],  # Worker port range
                scheduler_options={
                    "port": 60001 + seed,  # Main Job Port
                    "dashboard_address": 40550 + seed,
                },
            )
            cluster.scale(jobs=n_workers + 1)
        else:
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)

        # Dask creates n_workers jobs on the cluster which stay open.
        client = Client(
            address=cluster,
        )

        if is_slurm_cluster():
            # Dask waits for n_workers workers to be created
            logger.info("Waiting for workers to connect...")
            client.wait_for_workers(n_workers=n_workers + 1)
    return cluster, client


def optimize(
    cfg: DictConfig,
    optimizer: CMA,
    client: Client,
    cluster: SpecCluster,
    config: dict[str, Any],
    n_workers: int = 4,
    n_generations: int = 50,
) -> list[dict[str, Any]]:
    """Run ask and tell with CMA-ES."""
    config["cfg"] = OmegaConf.to_container(cfg, resolve=False)
    all_results = []
    for generation in range(n_generations):
        printr(generation)
        config["n_generation"] = generation
        inputs = [optimizer.ask() for _ in range(optimizer.population_size)]
        results = run_parallel(client=client, cluster=cluster, inputs=inputs, config=config, n_workers=n_workers)
        solutions = [(res["trial_info"]["config"], res["trial_value"]["cost"]) for res in results]
        optimizer.tell(solutions)
        all_results.append({"generation": generation, "result": results})

    return all_results


@hydra.main(version_base=None, config_path="configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Hydra-decorated main function."""
    printr("[bold green]Starting Dask parallel evaluations[/bold green]")
    printr(OmegaConf.to_yaml(cfg))

    logger.info("Starting Dask cluster...")

    client, cluster = setup_client(cfg.n_workers, cfg.seed)

    # Run parallel tasks
    logger.info("Run in parallel...")

    results = optimize(
        cfg=cfg,
        optimizer=instantiate(cfg.optimizer),
        client=client,
        cluster=cluster,
        config=dict(cfg.worker_config),
        n_workers=cfg.n_workers,
        n_generations=int(cfg.n_generations),
    )

    printr("[bold blue]Results:[/bold blue]")
    printr(results)

    hydra_cfg = HydraConfig.instance().get()
    rundir = hydra_cfg.run.dir
    try:
        results = OmegaConf.create(results)
        outstr = OmegaConf.to_yaml(results)
        with open(Path(rundir) / "results.yaml", "w") as file:
            file.write(outstr)
    except Exception as e:
        fn = Path(rundir) / "results.pickle"
        with open(fn, "wb") as file:
            pickle.dump(results, file)

        printr(e)
        raise e


if __name__ == "__main__":
    main()
