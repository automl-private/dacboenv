"""Optimize DACBO env objective function with CMA-ES."""

from __future__ import annotations

from itertools import product
from typing import Any

import hydra
import numpy as np
from carps.loggers.file_logger import convert_trials, dump_logs
from carps.utils.running import make_task
from carps.utils.trials import TrialInfo
from ConfigSpace import Configuration
from dask.distributed import Client, LocalCluster, SpecCluster, as_completed
from dask_jobqueue.slurm import SLURMCluster
from omegaconf import DictConfig, OmegaConf
from rich import (
    print as printr,
)

import dacboenv  # noqa: F401
from dacboenv.utils.loggingutils import get_logger, maybe_remove_logs
from dacboenv.utils.reference_performance import is_slurm_cluster

logger = get_logger("opt_via_ea")


def worker(x: tuple[list[float], tuple[int, str]], config: dict[str, Any]) -> dict[str, Any]:
    """
    Example worker function to evaluate a task.
    Replace with your actual computation.
    """
    # Import dacboenv again for custom resolvers
    import dacboenv  # noqa: F401, F811

    x, (seed, task_id) = x  # type: ignore[assignment]

    cfg: DictConfig = OmegaConf.create(config["cfg"])
    # The instance is selected randomly BEFORE
    task = make_task(cfg)
    cs = task.objective_function.configspace
    trial_info = TrialInfo(
        config=Configuration(configuration_space=cs, vector=np.array(x)),
        # By setting a seed and instance the instance set in dacboenv will be replaced.
        # This is okay at this point as we create a new env for each new objective function evaluation.
        seed=seed,
        instance=task_id,
    )
    trial_value = task.objective_function.evaluate(trial_info=trial_info)
    n_function_calls = None
    n_trials = config["idx"]
    return convert_trials(n_trials, trial_info, trial_value, n_function_calls)


def run_parallel(
    inputs: list[tuple[list[float], tuple[int, str]]],
    config: dict[str, Any],
    client: Client,
    n_workers: int = 4,
    logfile: str = "results.jsonl",
) -> list[dict[str, Any]]:
    """Run tasks in parallel using Dask with safe incremental logging."""
    logger.info(f"Running {len(inputs)} tasks on {n_workers} workers using Dask")

    futures = []
    for i, x in enumerate(inputs):
        _cfg = config.copy()
        _cfg["idx"] = i
        future = client.submit(worker, x, _cfg)
        futures.append(future)

    results: list[dict[str, Any]] = []

    # Stream results as they complete
    for future in as_completed(futures):
        try:
            res = future.result()
        except Exception as e:
            logger.exception("Worker failed", exc_info=e)
            continue

        # safe: only driver writes logs
        dump_logs(
            log_data=res,
            filename=logfile,
            directory=None,
        )

        results.append(res)

    return results


def setup_client(n_workers: int, seed: int, use_local: bool = False) -> tuple[SpecCluster, Client]:  # noqa: FBT001, FBT002
    """Setup client and cluster, either local or slurm."""
    if is_slurm_cluster() and not use_local:
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
    client: Client,
    cluster: SpecCluster,
    config: dict[str, Any],
    n_workers: int = 4,
    n_configs: int = 500,
) -> list[dict[str, Any]]:
    """Run ask and tell with CMA-ES."""
    config["cfg"] = OmegaConf.to_container(cfg, resolve=False)
    task = make_task(cfg)
    cs = task.objective_function.configspace
    configs = cs.sample_configuration(size=n_configs)
    configs_as_array = [list(c.values()) for c in configs]
    instances = task.objective_function._env.instance_selector.instances
    inputs = list(product(configs_as_array, instances))

    results = run_parallel(
        inputs=inputs,
        config=config,
        client=client,
        n_workers=n_workers,
        logfile="results.jsonl",
    )
    client.close()
    cluster.close()
    return results


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Hydra-decorated main function."""
    printr("[bold green]Starting Dask parallel evaluations[/bold green]")
    printr(OmegaConf.to_yaml(cfg))
    maybe_remove_logs(directory=None, overwrite=True, logfile="results.jsonl", logger=logger)

    logger.info("Starting Dask cluster...")

    cluster, client = setup_client(cfg.n_workers, cfg.seed, cfg.use_local)

    # Run parallel tasks
    logger.info("Run in parallel...")

    results = optimize(
        cfg=cfg,
        client=client,
        cluster=cluster,
        config=dict(cfg.worker_config),
        n_workers=cfg.n_workers,
        n_configs=cfg.n_configs,
    )

    printr("[bold blue]Results:[/bold blue]")
    printr(results)


if __name__ == "__main__":
    main()
