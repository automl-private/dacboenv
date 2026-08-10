"""Evaluate native SMAC defaults under the protocol's independent seed tree."""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from carps.loggers.file_logger import get_run_directory
from carps.objective_functions.bbob import BBOBObjectiveFunction
from carps.utils.loggingutils import get_logger
from omegaconf import DictConfig, OmegaConf

from dacboenv.env.reward import TRUE_REGRET_EPSILON
from dacboenv.experiment.evaluation_determinism import require_process_determinism, runhistory_fingerprints
from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
)
from dacboenv.reference import (
    BBOBExactReferenceProvider,
    JSONLReferenceBreachRecorder,
    ObjectiveReference,
    ReferenceBreachContext,
    reference_regret,
)
from dacboenv.utils.carps_optimizer import build_carps_optimizer, load_optimizer_config
from dacboenv.utils.seeding import episode_component_seeds

logger = get_logger("DefaultSMAC")


@dataclass(frozen=True)
class DefaultSMACResult:
    """Comparable outcome from one fixed task/seed context."""

    task_id: str
    inner_seed: int
    initial_incumbent: float
    final_incumbent: float
    reference_value: float
    initial_regret: float
    final_regret: float
    normalized_final_regret: float
    normalized_anytime_auc: float
    telescoping_return: float
    bo_evaluations: int
    runtime_seconds: float
    incumbent_trajectory: tuple[float, ...]
    fingerprints: dict[str, Any] = field(default_factory=dict)


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
    objective_reference: ObjectiveReference | None = None,
    context_split: str = "validation",
) -> DefaultSMACResult:
    """Run BlackBoxFacade defaults for one exact/best-known context.

    The scenario is keyed by ``inner_seed`` while ConfigSpace, initial design,
    random design, and acquisition maximization use independent named child
    streams, as they do in every other protocol cell.
    """
    started = time.perf_counter()
    optimizer_cfg = load_optimizer_config("SMAC3-BlackBoxFacade")
    component_seeds = episode_component_seeds(inner_seed)
    # Pair native default SMAC to the DACBO methods' initial-design protocol;
    # only the post-design acquisition controller is method-specific.
    OmegaConf.update(
        optimizer_cfg,
        "optimizer.smac_cfg.smac_kwargs.initial_design",
        {
            "_target_": "smac.initial_design.sobol_design.SobolInitialDesign",
            "_partial_": True,
            "n_configs": None,
            "n_configs_per_hyperparameter": 8,
            "max_ratio": 0.2,
            "seed": component_seeds["initial_design"],
        },
        force_add=True,
    )
    optimizer = build_carps_optimizer(
        task_id,
        inner_seed,
        optimizer_cfg=optimizer_cfg,
        output_directory=output_directory,
        context_split=context_split,
    )
    solver = optimizer.solver
    objective = optimizer.task.objective_function
    if isinstance(objective, BBOBObjectiveFunction):
        objective_reference = BBOBExactReferenceProvider().get_reference(
            task_id,
            objective,
            {
                "runtime_objective_transform": "identity",
                "reporting_objective_transform": "identity",
                "fidelity": "not_applicable",
            },
        )
    elif objective_reference is None:
        raise ValueError(f"Default-SMAC evaluation requires a best-known reference for {task_id!r}.")
    if objective_reference.task_id != task_id:
        raise ValueError("Default-SMAC objective reference belongs to a different task.")
    reference_value = float(objective_reference.value)
    breach_recorder = JSONLReferenceBreachRecorder(output_directory / "reference_breaches.jsonl")
    parts = task_id.split("/")
    scenario = parts[2] if len(parts) >= 4 else parts[0]  # noqa: PLR2004
    instance = parts[3] if len(parts) >= 4 else parts[-1]  # noqa: PLR2004

    initial_design_size = len(solver.optimizer.intensifier.config_selector._initial_design_configs)
    initial_incumbent: float | None = None
    incumbent_trajectory: list[float] = []
    while solver.runhistory.finished < solver.scenario.n_trials:
        trial_info = solver.ask()
        _, trial_value = solver.optimizer._runner.run_wrapper(trial_info)
        solver.tell(trial_info, trial_value)
        reference_regret(
            objective_reference,
            _scalar_cost(trial_value.cost),
            recorder=breach_recorder,
            context=ReferenceBreachContext(
                run_id=f"native-default-smac:{task_id}:seed-{inner_seed}",
                trial=int(solver.runhistory.finished - 1),
                outer_seed=None,
                inner_seed=int(inner_seed),
                scenario=scenario,
                instance=instance,
            ),
        )
        incumbent_trajectory.append(_scalar_cost(solver.runhistory.get_min_cost(solver.intensifier.get_incumbent())))
        if initial_incumbent is None and len(solver.runhistory.get_configs()) >= initial_design_size:
            initial_incumbent = _scalar_cost(solver.runhistory.get_min_cost(solver.intensifier.get_incumbent()))

    if initial_incumbent is None:
        raise RuntimeError(f"Default SMAC did not finish its initial design for {task_id!r}.")
    final_incumbent = _scalar_cost(solver.runhistory.get_min_cost(solver.intensifier.get_incumbent()))
    initial_regret = max(initial_incumbent - reference_value, 0.0)
    final_regret = max(final_incumbent - reference_value, 0.0)
    regret_scale = max(initial_regret, TRUE_REGRET_EPSILON)
    normalized_final_regret = final_regret / regret_scale
    normalized_trajectory = np.maximum(np.asarray(incumbent_trajectory) - reference_value, 0.0) / regret_scale
    telescoping_return = normalized_telescoping_return(initial_regret, final_regret)
    fingerprints = runhistory_fingerprints(
        solver.runhistory,
        solver.scenario.configspace,
        initial_design_size,
    )
    return DefaultSMACResult(
        task_id=task_id,
        inner_seed=inner_seed,
        initial_incumbent=initial_incumbent,
        final_incumbent=final_incumbent,
        reference_value=reference_value,
        initial_regret=initial_regret,
        final_regret=final_regret,
        normalized_final_regret=normalized_final_regret,
        normalized_anytime_auc=float(np.mean(normalized_trajectory)),
        telescoping_return=telescoping_return,
        bo_evaluations=int(solver.runhistory.finished),
        runtime_seconds=float(time.perf_counter() - started),
        incumbent_trajectory=tuple(incumbent_trajectory),
        fingerprints=fingerprints,
    )


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Evaluate native default SMAC on every fixed manifest context."""
    require_process_determinism()
    logger.info(OmegaConf.to_yaml(cfg))
    manifest = OmegaConf.to_container(cfg.evaluation_instances, resolve=True)
    if not isinstance(manifest, dict):
        raise TypeError("evaluation_instances must be a manifest mapping.")
    validate_manifest_structure(manifest)
    require_runnable_manifest(manifest)
    if manifest["split"] == "test" and not bool(cfg.experiment.get("allow_sealed_test", False)):
        raise PermissionError(
            f"Manifest {manifest['id']!r} is sealed test data; set "
            "experiment.allow_sealed_test=true only for an authorized final report."
        )
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
