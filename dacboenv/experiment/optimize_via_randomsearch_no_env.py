"""Optimize DACBO env objective function with CMA-ES."""

from __future__ import annotations

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
from rich import print as printr

import dacboenv  # noqa: F401
from dacboenv.utils.loggingutils import get_logger, maybe_remove_logs
from dacboenv.utils.reference_performance import is_slurm_cluster

logger = get_logger("opt_via_ea")


def worker(x: list[float], config: dict[str, Any]) -> dict[str, Any]:
    """Worker function without instances."""
    import dacboenv  # noqa: F401, F811

    cfg: DictConfig = OmegaConf.create(config["cfg"])
    task = make_task(cfg)
    cs = task.objective_function.configspace

    trial_info = TrialInfo(
        config=Configuration(configuration_space=cs, vector=np.array(x)),
        seed=config["seed"],
        instance=None,  # NO INSTANCES
    )

    trial_value = task.objective_function.evaluate(trial_info=trial_info)
    n_function_calls = None
    n_trials = config["idx"]

    return convert_trials(n_trials, trial_info, trial_value, n_function_calls)


def run_parallel(
    inputs: list[list[float]],
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

    for future in as_completed(futures):
        try:
            res = future.result()
        except Exception as e:
            logger.exception("Worker failed", exc_info=e)
            continue

        dump_logs(
            log_data=res,
            filename=logfile,
            directory=None,
        )

        results.append(res)

    return results


def setup_client(
    n_workers: int,
    seed: int,
    use_local: bool = False,  # noqa: FBT001, FBT002
) -> tuple[SpecCluster, Client]:
    """Setup client and cluster, either local or slurm."""
    if is_slurm_cluster() and not use_local:
        cluster = SLURMCluster(
            queue="normal",
            cores=1,
            memory="16 GB",
            walltime="48:00:00",
            processes=1,
            log_directory=f"tmp/smac_dask_slurm/{seed}",
            nanny=False,
            worker_extra_args=["--worker-port", "60010:60100"],
            scheduler_options={
                "port": 60001 + seed,
                "dashboard_address": 40550 + seed,
            },
        )
        cluster.scale(jobs=n_workers + 1)
    else:
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)

    client = Client(address=cluster)

    if is_slurm_cluster():
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
    """Run evaluations without instances."""
    config["cfg"] = OmegaConf.to_container(cfg, resolve=False)
    config["seed"] = cfg.seed

    task = make_task(cfg)
    cs = task.objective_function.configspace

    configs = cs.sample_configuration(size=n_configs)
    configs_as_array = [list(c.values()) for c in configs]

    results = run_parallel(
        inputs=configs_as_array,
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

    maybe_remove_logs(
        directory=None,
        overwrite=True,
        logfile="results.jsonl",
        logger=logger,
    )

    logger.info("Starting Dask cluster...")
    cluster, client = setup_client(cfg.n_workers, cfg.seed, cfg.use_local)

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
