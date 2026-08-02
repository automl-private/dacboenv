"""Evaluate native SMAC defaults under the protocol's independent seed tree."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from carps.loggers.file_logger import get_run_directory
from carps.objective_functions.bbob import BBOBObjectiveFunction
from carps.utils.loggingutils import get_logger
from omegaconf import DictConfig, OmegaConf

from dacboenv.env.reward import TRUE_REGRET_EPSILON
from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
)
from dacboenv.utils.carps_optimizer import build_carps_optimizer

logger = get_logger("DefaultSMAC")


@dataclass(frozen=True)
class DefaultSMACResult:
    """Comparable outcome from one fixed task/seed context."""

    task_id: str
    inner_seed: int
    initial_incumbent: float
    final_incumbent: float
    reference_optimum: float
    initial_regret: float
    final_regret: float
    normalized_final_regret: float
    telescoping_return: float
    bo_evaluations: int


def _scalar_cost(cost: Any) -> float:
    values = np.asarray(cost, dtype=float).reshape(-1)
    return float(values[0])


def normalized_telescoping_return(initial_regret: float, final_regret: float) -> float:
    """Return the same bounded episode potential difference as the reward."""
    initial_regret = max(float(initial_regret), 0.0)
    final_regret = max(float(final_regret), 0.0)
    regret_scale = max(initial_regret, TRUE_REGRET_EPSILON)
    initial_normalized = initial_regret / regret_scale
    final_normalized = final_regret / regret_scale
    return float(
        (np.log(initial_normalized + TRUE_REGRET_EPSILON) - np.log(final_normalized + TRUE_REGRET_EPSILON))
        / -np.log(TRUE_REGRET_EPSILON)
    )


def run_default_smac_episode(
    task_id: str,
    inner_seed: int,
    *,
    output_directory: Path,
) -> DefaultSMACResult:
    """Run BlackBoxFacade's algorithmic defaults for one native BBOB context.

    The scenario is keyed by ``inner_seed`` while ConfigSpace, initial design,
    random design, and acquisition maximization use independent named child
    streams, as they do in every other protocol cell.
    """
    optimizer = build_carps_optimizer(
        task_id,
        inner_seed,
        optimizer_id="SMAC3-BlackBoxFacade",
        output_directory=output_directory,
    )
    solver = optimizer.solver
    objective = optimizer.task.objective_function
    if not isinstance(objective, BBOBObjectiveFunction) or objective.f_min is None:
        raise ValueError(f"Default-SMAC true-regret evaluation has no exact BBOB reference for {task_id!r}.")

    initial_design_size = len(solver.optimizer.intensifier.config_selector._initial_design_configs)
    initial_incumbent: float | None = None
    while solver.runhistory.finished < solver.scenario.n_trials:
        trial_info = solver.ask()
        _, trial_value = solver.optimizer._runner.run_wrapper(trial_info)
        solver.tell(trial_info, trial_value)
        if initial_incumbent is None and len(solver.runhistory.get_configs()) >= initial_design_size:
            initial_incumbent = _scalar_cost(solver.runhistory.get_min_cost(solver.intensifier.get_incumbent()))

    if initial_incumbent is None:
        raise RuntimeError(f"Default SMAC did not finish its initial design for {task_id!r}.")
    final_incumbent = _scalar_cost(solver.runhistory.get_min_cost(solver.intensifier.get_incumbent()))
    reference_optimum = float(objective.f_min)
    initial_regret = max(initial_incumbent - reference_optimum, 0.0)
    final_regret = max(final_incumbent - reference_optimum, 0.0)
    regret_scale = max(initial_regret, TRUE_REGRET_EPSILON)
    normalized_final_regret = final_regret / regret_scale
    telescoping_return = normalized_telescoping_return(initial_regret, final_regret)
    return DefaultSMACResult(
        task_id=task_id,
        inner_seed=inner_seed,
        initial_incumbent=initial_incumbent,
        final_incumbent=final_incumbent,
        reference_optimum=reference_optimum,
        initial_regret=initial_regret,
        final_regret=final_regret,
        normalized_final_regret=normalized_final_regret,
        telescoping_return=telescoping_return,
        bo_evaluations=int(solver.runhistory.finished),
    )


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Evaluate native default SMAC on every fixed manifest context."""
    logger.info(OmegaConf.to_yaml(cfg))
    manifest = OmegaConf.to_container(cfg.evaluation_instances, resolve=True)
    if not isinstance(manifest, dict):
        raise TypeError("evaluation_instances must be a manifest mapping.")
    validate_manifest_structure(manifest)
    require_runnable_manifest(manifest)
    validate_native_bbob_manifest(manifest)
    if any(seed is None for seed in manifest["inner_seeds"]):
        raise ValueError("Default-SMAC evaluation requires frozen integer inner seeds.")

    rundir = Path(get_run_directory())
    results = [
        run_default_smac_episode(
            task_id,
            int(inner_seed),
            output_directory=rundir / "smac3_output" / f"task_{task_index}" / f"seed_{inner_seed}",
        )
        for inner_seed in manifest["inner_seeds"]
        for task_index, task_id in enumerate(manifest["task_ids"])
    ]
    output_path = rundir / "default_smac_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)
    logger.info(f"Saved {len(results)} default-SMAC episodes to {output_path}.")


if __name__ == "__main__":
    main()
