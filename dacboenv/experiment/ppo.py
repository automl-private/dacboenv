"""Train Stable-Baselines3 PPO on DACBOEnv."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch as th
from carps.loggers.file_logger import get_run_directory
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecNormalize,
    sync_envs_normalization,
)

# Register OmegaConf resolvers.
import dacboenv  # noqa: F401
from dacboenv.experiment.ppo_utils import ActionLoggingCallback, structured_policy_sensitivity
from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
)
from dacboenv.utils.carps_optimizer import get_task_config
from dacboenv.utils.loggingutils import maybe_remove_logs
from dacboenv.utils.seeding import derive_named_seed, run_seed_metadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import DACBOEnv

logger = get_logger("PPO")


def derive_worker_seed(run_seed: int, worker_id: int) -> int:
    """Derive a stable independent child seed for one vector worker."""
    return derive_named_seed(run_seed, "vector_worker", index=worker_id)


def _vector_worker_seeds(run_seed: int | None, n_workers: int) -> list[int]:
    """Return the same child seeds used when constructing vector workers."""
    if run_seed is None:
        run_seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    return [derive_worker_seed(run_seed, worker_id) for worker_id in range(n_workers)]


class HierarchicalSeedDummyVecEnv(DummyVecEnv):
    """DummyVecEnv whose reset seeds match the worker seed hierarchy."""

    def seed(self, seed: int | None = None) -> list[int]:
        """Stage independently derived worker seeds for the next reset."""
        self._seeds = _vector_worker_seeds(seed, self.num_envs)
        return self._seeds


class HierarchicalSeedSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose reset seeds match the worker seed hierarchy."""

    def seed(self, seed: int | None = None) -> list[int]:
        """Stage independently derived worker seeds for the next reset."""
        self._seeds = _vector_worker_seeds(seed, self.num_envs)
        return self._seeds


@dataclass(frozen=True)
class TrainingSchedule:
    """Resolved PPO sampling schedule."""

    n_envs: int
    n_steps: int
    rollout_size: int
    batch_size: int
    total_timesteps: int
    n_updates: int
    collected_timesteps: int


@dataclass(frozen=True)
class WorkerContextAssignment:
    """Persistent domain/dimension assignment for one vector worker."""

    domain: str
    task_ids: list[str]
    selector_target: str
    bbob_dimension: int | None = None


@dataclass(frozen=True)
class ValidationScores:
    """Hierarchically aggregated fixed-manifest validation scores."""

    bbob_score: float | None
    yahpo_score: float | None
    balanced_score: float
    worst_domain_score: float
    per_task: dict[str, float]
    per_scenario: dict[str, float]


def aggregate_validation_scores(  # noqa: C901, PLR0912
    task_ids: list[str],
    inner_seeds: list[int],
    episode_rewards: list[float],
) -> ValidationScores:
    """Aggregate seeds before instances/functions/scenarios and domains."""
    expected = len(task_ids) * len(inner_seeds)
    if len(episode_rewards) != expected:
        raise ValueError(f"Expected {expected} fixed-manifest rewards, got {len(episode_rewards)}.")

    rewards_by_task: dict[str, list[float]] = {task_id: [] for task_id in task_ids}
    reward_index = 0
    for _inner_seed in inner_seeds:
        for task_id in task_ids:
            rewards_by_task[task_id].append(float(episode_rewards[reward_index]))
            reward_index += 1
    per_task = {task_id: float(np.mean(rewards)) for task_id, rewards in rewards_by_task.items()}

    bbob_by_function: dict[tuple[int, int], list[float]] = {}
    yahpo_by_scenario: dict[str, list[float]] = {}
    for task_id, task_score in per_task.items():
        parts = task_id.split("/")
        if parts[0].lower() == "bbob" and len(parts) == 4:  # noqa: PLR2004
            dimension, function_id = int(parts[1]), int(parts[2])
            bbob_by_function.setdefault((dimension, function_id), []).append(task_score)
        elif parts[0].lower() == "yahpo" and len(parts) >= 3:  # noqa: PLR2004
            scenario_index = 2 if parts[1].lower() in {"so", "mo", "momf"} else 1
            yahpo_by_scenario.setdefault(parts[scenario_index], []).append(task_score)
        else:
            raise ValueError(f"Cannot aggregate unsupported validation task ID {task_id!r}.")

    bbob_score: float | None = None
    if bbob_by_function:
        scores_by_dimension_group: dict[tuple[int, int], list[float]] = {}
        for (dimension, function_id), instance_scores in bbob_by_function.items():
            if function_id <= 5:  # noqa: PLR2004
                function_group = 0
            elif function_id <= 9:  # noqa: PLR2004
                function_group = 1
            elif function_id <= 14:  # noqa: PLR2004
                function_group = 2
            elif function_id <= 19:  # noqa: PLR2004
                function_group = 3
            else:
                function_group = 4
            scores_by_dimension_group.setdefault((dimension, function_group), []).append(
                float(np.mean(instance_scores))
            )
        scores_by_dimension: dict[int, list[float]] = {}
        for (dimension, _function_group), function_scores in scores_by_dimension_group.items():
            scores_by_dimension.setdefault(dimension, []).append(float(np.mean(function_scores)))
        bbob_score = float(np.mean([np.mean(scores) for scores in scores_by_dimension.values()]))

    per_scenario = {
        scenario: float(np.mean(instance_scores)) for scenario, instance_scores in yahpo_by_scenario.items()
    }
    yahpo_score = float(np.mean(list(per_scenario.values()))) if per_scenario else None
    domain_scores = [score for score in (bbob_score, yahpo_score) if score is not None]
    balanced_score = float(np.mean(domain_scores))
    return ValidationScores(
        bbob_score=bbob_score,
        yahpo_score=yahpo_score,
        balanced_score=balanced_score,
        worst_domain_score=float(min(domain_scores)),
        per_task=per_task,
        per_scenario=per_scenario,
    )


class ProtocolEvalCallback(EvalCallback):
    """Fixed-manifest evaluation with hierarchical scores and checkpoints."""

    def __init__(
        self,
        *args: Any,
        manifest_task_ids: list[str],
        manifest_inner_seeds: list[int],
        protocol_save_path: Path,
        **kwargs: Any,
    ) -> None:
        # Disable EvalCallback's raw-episode-mean checkpoint. Protocol scores
        # are computed after its reproducible evaluation and saved below.
        kwargs["best_model_save_path"] = None
        super().__init__(*args, **kwargs)
        self._manifest_task_ids = manifest_task_ids
        self._manifest_inner_seeds = manifest_inner_seeds
        self._protocol_save_path = protocol_save_path
        self._best_scores = {"balanced": -np.inf, "bbob": -np.inf, "yahpo": -np.inf}

    def _save_protocol_checkpoint(self, label: str) -> None:
        self._protocol_save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(self._protocol_save_path / f"best_{label}_model")
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None:
            vecnormalize.save(str(self._protocol_save_path / f"best_{label}_vecnormalize.pkl"))
        if label == "balanced":
            # Compatibility alias consumed by the existing evaluation tools.
            self.model.save(self._protocol_save_path / "best_model")

    @staticmethod
    def _metric_tag(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # ``evaluate_policy`` explicitly resets, while DummyVecEnv also
            # auto-resets the final terminal episode.  Rewind first so every
            # checkpoint starts at manifest context zero rather than rotating
            # the reward array relative to its task labels.
            self.eval_env.env_method("restart_fixed_instance_sequence")
        previous_evaluations = len(self.evaluations_results)
        continue_training = super()._on_step()
        if len(self.evaluations_results) == previous_evaluations:
            return continue_training

        rewards = [float(reward) for reward in self.evaluations_results[-1]]
        scores = aggregate_validation_scores(
            self._manifest_task_ids,
            self._manifest_inner_seeds,
            rewards,
        )
        self.logger.record("eval/balanced_score", scores.balanced_score)
        self.logger.record("eval/worst_domain_score", scores.worst_domain_score)
        if scores.bbob_score is not None:
            self.logger.record("eval/bbob_score", scores.bbob_score)
        if scores.yahpo_score is not None:
            self.logger.record("eval/yahpo_score", scores.yahpo_score)
        for task_id, score in scores.per_task.items():
            self.logger.record(f"eval/per_task/{self._metric_tag(task_id)}", score)
        for scenario, score in scores.per_scenario.items():
            self.logger.record(f"eval/per_scenario/{self._metric_tag(scenario)}", score)

        checkpoint_scores = {
            "balanced": scores.balanced_score,
            "bbob": scores.bbob_score,
            "yahpo": scores.yahpo_score,
        }
        for label, score in checkpoint_scores.items():
            if score is not None and score > self._best_scores[label]:
                self._best_scores[label] = score
                self._save_protocol_checkpoint(label)
        self.logger.dump(self.num_timesteps)
        return continue_training


def assign_training_worker_context(
    task_ids: list[str],
    *,
    worker_id: int,
    n_workers: int,
    bbob_fraction: float = 0.6,
) -> WorkerContextAssignment:
    """Assign persistent workers, then leave reset-level sampling hierarchical."""
    if not task_ids:
        raise ValueError("Training requires at least one runnable task ID.")
    if worker_id < 0 or worker_id >= n_workers:
        raise ValueError(f"worker_id must be in [0, {n_workers}), got {worker_id}.")
    if not 0 < bbob_fraction < 1:
        raise ValueError(f"bbob_fraction must be in (0, 1), got {bbob_fraction}.")

    bbob_tasks = sorted(task_id for task_id in task_ids if task_id.lower().startswith("bbob/"))
    yahpo_tasks = sorted(task_id for task_id in task_ids if task_id.lower().startswith("yahpo/"))
    unknown_tasks = sorted(set(task_ids) - set(bbob_tasks) - set(yahpo_tasks))
    if unknown_tasks:
        raise ValueError(f"The protocol sampler only accepts BBOB or YAHPO task IDs, got {unknown_tasks}.")

    local_worker_id = worker_id
    if bbob_tasks and yahpo_tasks:
        if n_workers < 2:  # noqa: PLR2004
            raise ValueError("Mixed BBOB/YAHPO training requires at least two persistent workers.")
        n_bbob_workers = min(max(math.ceil(n_workers * bbob_fraction), 1), n_workers - 1)
        if worker_id >= n_bbob_workers:
            return WorkerContextAssignment(
                domain="yahpo",
                task_ids=yahpo_tasks,
                selector_target="dacboenv.env.instance.HierarchicalYAHPOInstanceSelector",
            )
    elif yahpo_tasks:
        return WorkerContextAssignment(
            domain="yahpo",
            task_ids=yahpo_tasks,
            selector_target="dacboenv.env.instance.HierarchicalYAHPOInstanceSelector",
        )

    dimensions = sorted({int(task_id.split("/")[1]) for task_id in bbob_tasks})
    dimension = dimensions[local_worker_id % len(dimensions)]
    dimension_tasks = [task_id for task_id in bbob_tasks if int(task_id.split("/")[1]) == dimension]
    return WorkerContextAssignment(
        domain="bbob",
        task_ids=dimension_tasks,
        selector_target="dacboenv.env.instance.HierarchicalBBOBInstanceSelector",
        bbob_dimension=dimension,
    )


def resolve_training_schedule(cfg: DictConfig) -> TrainingSchedule:
    """Resolve and validate the SB3 rollout schedule from configuration."""
    n_envs = int(cfg.experiment.n_workers)
    n_steps = int(cfg.optimizer.n_steps)
    batch_size = int(cfg.optimizer.batch_size)
    total_timesteps = int(cfg.experiment.total_timesteps)

    if n_envs <= 0 or n_steps <= 0 or batch_size <= 0 or total_timesteps <= 0:
        raise ValueError("n_workers, n_steps, batch_size, and total_timesteps must all be positive.")

    rollout_size = n_envs * n_steps
    if batch_size > rollout_size:
        raise ValueError(f"batch_size ({batch_size}) exceeds rollout size ({rollout_size}).")
    if rollout_size % batch_size:
        raise ValueError(
            "The rollout size must be divisible by batch_size to avoid a "
            f"truncated PPO minibatch, got {rollout_size} % {batch_size}."
        )

    n_updates = math.ceil(total_timesteps / rollout_size)
    return TrainingSchedule(
        n_envs=n_envs,
        n_steps=n_steps,
        rollout_size=rollout_size,
        batch_size=batch_size,
        total_timesteps=total_timesteps,
        n_updates=n_updates,
        collected_timesteps=n_updates * rollout_size,
    )


def populate_legacy_schedule(cfg: DictConfig) -> None:
    """Fill schedule fields omitted by the historical normalized PPO configs.

    The former entry points derived these values at runtime from the first
    configured CARPS task. Keeping that behavior here lets those module CLIs
    use the corrected shared runner without requiring config changes.
    """
    optimizer = cfg.optimizer
    experiment = cfg.experiment
    missing_n_steps = optimizer.get("n_steps") is None
    missing_batch_size = optimizer.get("batch_size") is None
    missing_total_timesteps = experiment.get("total_timesteps") is None
    missing_checkpoint_freq = experiment.get("checkpoint_freq") is None
    if not any(
        (
            missing_n_steps,
            missing_batch_size,
            missing_total_timesteps,
            missing_checkpoint_freq,
        )
    ):
        return

    n_workers = int(experiment.n_workers)
    if missing_n_steps:
        task_ids = list(cfg.dacboenv.task_ids)
        if not task_ids:
            raise ValueError("At least one task ID is required to infer PPO n_steps.")
        task_cfg = get_task_config(str(task_ids[0]))
        n_steps = int(task_cfg.task.optimization_resources.n_trials)
    else:
        n_steps = int(optimizer.n_steps)

    rollout_size = n_workers * n_steps
    if missing_batch_size:
        # Retain the historical half-rollout minibatch where possible while
        # ensuring the stricter shared runner never creates a partial batch.
        batch_size = max(rollout_size // 2, 1)
        while rollout_size % batch_size:
            batch_size -= 1
    else:
        batch_size = int(optimizer.batch_size)

    if missing_total_timesteps:
        n_episodes = int(experiment.n_episodes)
        total_timesteps = rollout_size * n_episodes
    else:
        total_timesteps = int(experiment.total_timesteps)

    checkpoint_freq = rollout_size * 5 if missing_checkpoint_freq else int(experiment.checkpoint_freq)

    with open_dict(cfg):
        optimizer.n_steps = n_steps
        optimizer.batch_size = batch_size
        experiment.total_timesteps = total_timesteps
        experiment.checkpoint_freq = checkpoint_freq

    logger.info(
        "Filled legacy PPO schedule: "
        f"n_steps={n_steps}, batch_size={batch_size}, "
        f"total_timesteps={total_timesteps}, checkpoint_freq={checkpoint_freq}."
    )


def _copy_config(cfg: DictConfig) -> DictConfig:
    """Create an independent OmegaConf tree for one environment worker."""
    copied = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    if not isinstance(copied, DictConfig):
        raise TypeError("Expected the root training configuration to be a mapping.")
    return copied


def make_env_factory(
    cfg: DictConfig,
    *,
    worker_id: int,
    output_directory: Path,
    task_ids: list[str] | None = None,
    inner_seeds: list[int | None] | None = None,
    instance_set_id: str | None = None,
    instance_selector_cfg: DictConfig | None = None,
    protocol_metadata: dict[str, Any] | None = None,
) -> Callable[[], DACBOEnv]:
    """Build an isolated DACBO environment factory for a vector worker."""

    def _init() -> DACBOEnv:
        config = _copy_config(cfg)
        config.seed = derive_worker_seed(int(cfg.seed), worker_id)

        if task_ids is not None:
            config.dacboenv.task_ids = list(task_ids)
        if inner_seeds is not None:
            config.dacboenv.inner_seeds = list(inner_seeds)
        if instance_set_id is not None:
            config.instance_set_id = instance_set_id
        if instance_selector_cfg is not None:
            selector = OmegaConf.to_container(instance_selector_cfg, resolve=True)
            if not isinstance(selector, dict):
                raise TypeError("An instance selector config must be a mapping.")
            config.dacboenv.instance_selector_class = selector
        if protocol_metadata is not None:
            with open_dict(config.dacboenv):
                config.dacboenv.protocol_metadata = dict(protocol_metadata)

        selector_cfg = config.dacboenv.instance_selector_class
        selector_target = str(selector_cfg.get("_target_", "")) if isinstance(selector_cfg, DictConfig) else ""
        if selector_target.endswith("RoundRobinInstanceSelector"):
            selector_cfg.offset = worker_id

        config.dacboenv.optimizer_cfg.smac_cfg.scenario.output_directory = str(output_directory)

        task = make_task(config)
        return task.objective_function._env

    return _init


def _protocol_metadata_from_config(cfg: DictConfig) -> dict[str, Any]:
    """Extract immutable manifest identifiers for run and episode logs."""
    metadata: dict[str, Any] = {}
    for config_key, label in (
        ("training_instances", "train"),
        ("validation_instances", "validation"),
        ("test_instances", "test"),
    ):
        manifest = cfg.get(config_key, None)
        if manifest is None:
            continue
        plain_manifest = OmegaConf.to_container(manifest, resolve=True)
        if not isinstance(plain_manifest, dict):
            raise TypeError(f"{config_key} must be a manifest mapping.")
        if plain_manifest.get("schema_version", None) is not None:
            validate_manifest_structure(plain_manifest)
            require_runnable_manifest(plain_manifest)
            if plain_manifest.get("domain") == "bbob":
                validate_native_bbob_manifest(plain_manifest)
        if manifest.get("schema_version", None) is not None:
            metadata[f"{label}_manifest/version"] = int(manifest.schema_version)
        if manifest.get("id", None) is not None:
            metadata[f"{label}_manifest_id"] = str(manifest.id)
        if manifest.get("manifest_hash", None) is not None:
            metadata[f"{label}_manifest_hash"] = str(manifest.manifest_hash)
    return metadata


def _make_vec_env(
    factories: list[Callable[[], DACBOEnv]],
    *,
    start_method: str | None,
) -> VecEnv:
    """Use subprocess workers when useful and a local vector env otherwise."""
    if len(factories) == 1:
        return HierarchicalSeedDummyVecEnv(factories)
    return HierarchicalSeedSubprocVecEnv(factories, start_method=start_method)


def stage_training_worker_seeds(vec_env: VecEnv, run_seed: int) -> list[int]:
    """Restore worker-root seeds after SB3 seeds the model and env together.

    SB3 calls ``env.seed(model_seed)`` during model construction.  The policy
    seed intentionally belongs to an independent stream, so stage the vector
    workers again from the outer run seed before the first learning reset.
    """
    expected = _vector_worker_seeds(run_seed, vec_env.num_envs)
    staged = [int(seed) for seed in vec_env.seed(run_seed)]
    if staged != expected:
        raise RuntimeError(f"Vector env staged worker seeds {staged}, expected {expected}.")
    return staged


def _policy_kwargs_and_optimizer_cfg(cfg: DictConfig) -> tuple[dict[str, Any], DictConfig]:
    """Separate SB3 policy kwargs without mutating Hydra's saved config."""
    converted = OmegaConf.to_container(cfg.optimizer, resolve=True)
    if not isinstance(converted, dict):
        raise TypeError("optimizer must be a mapping.")

    raw_policy_kwargs = converted.pop("policy_kwargs", {})
    if not isinstance(raw_policy_kwargs, dict):
        raise TypeError("optimizer.policy_kwargs must be a mapping.")
    policy_kwargs: dict[str, Any] = raw_policy_kwargs

    optimizer_cfg = OmegaConf.create(converted)
    if not isinstance(optimizer_cfg, DictConfig):
        raise TypeError("optimizer must be a mapping.")
    return policy_kwargs, optimizer_cfg


def _callback_frequency(transition_frequency: int, n_envs: int) -> int:
    """Convert an SB3 transition interval to vector callback calls."""
    if transition_frequency <= 0:
        raise ValueError("Callback transition frequencies must be positive.")
    return max(transition_frequency // n_envs, 1)


def _write_policy_sensitivity(model: BaseAlgorithm, rundir: Path) -> None:
    """Evaluate structured-policy interventions on the final real rollout state."""
    observation = getattr(model, "_last_obs", None)
    if not isinstance(observation, dict) or set(observation) != {"global_state", "action_features"}:
        return
    numpy_observation = {key: np.asarray(value).copy() for key, value in observation.items()}

    def probability_function(state: dict[str, np.ndarray]) -> np.ndarray:
        observation_tensor, _vectorized = model.policy.obs_to_tensor(state)
        with th.no_grad():
            distribution = model.policy.get_distribution(observation_tensor).distribution
        probabilities = getattr(distribution, "probs", None)
        if probabilities is None:
            raise TypeError("Structured PPO diagnostics require a categorical policy distribution.")
        return probabilities.detach().cpu().numpy()

    sensitivity = structured_policy_sensitivity(numpy_observation, probability_function)
    (rundir / "policy_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for intervention, metrics in sensitivity.items():
        for metric, value in metrics.items():
            model.logger.record(f"policy_sensitivity/{intervention}/{metric}", value)
    model.logger.dump(model.num_timesteps)


def run(cfg: DictConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    """Train and validate PPO on the configured BO-MDP."""
    logger.info(OmegaConf.to_yaml(cfg))

    rundir = Path(get_run_directory())
    maybe_remove_logs(directory=None, overwrite=True, logfile="model.zip", logger=logger)
    populate_legacy_schedule(cfg)
    schedule = resolve_training_schedule(cfg)
    policy_kwargs, optimizer_cfg = _policy_kwargs_and_optimizer_cfg(cfg)
    seed_metadata = run_seed_metadata(int(cfg.seed), schedule.n_envs)
    optimizer_cfg.seed = int(seed_metadata["policy_model_seed"])
    (rundir / "seed_streams.json").write_text(
        json.dumps(seed_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Seed stream metadata: {seed_metadata}")
    protocol_metadata = _protocol_metadata_from_config(cfg)
    (rundir / "protocol_metadata.json").write_text(
        json.dumps(protocol_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(f"Manifest protocol metadata: {protocol_metadata}")

    logger.info(
        "PPO schedule: "
        f"{schedule.n_envs} envs x {schedule.n_steps} steps = "
        f"{schedule.rollout_size} transitions/update; "
        f"{schedule.n_updates} updates collect {schedule.collected_timesteps} "
        f"transitions for requested {schedule.total_timesteps}."
    )

    start_method = cfg.experiment.get("start_method", None)
    sampler_cfg = cfg.experiment.get("training_sampler", None)
    use_protocol_sampler = sampler_cfg is not None and bool(sampler_cfg.get("enabled", True))
    training_factories: list[Callable[[], DACBOEnv]] = []
    worker_assignments: list[WorkerContextAssignment] = []
    for worker_id in range(schedule.n_envs):
        task_ids: list[str] | None = None
        selector_cfg: DictConfig | None = None
        if use_protocol_sampler:
            assignment = assign_training_worker_context(
                list(cfg.dacboenv.task_ids),
                worker_id=worker_id,
                n_workers=schedule.n_envs,
                bbob_fraction=float(sampler_cfg.get("bbob_fraction", 0.6)),
            )
            worker_assignments.append(assignment)
            task_ids = assignment.task_ids
            selector_cfg = OmegaConf.create(
                {
                    "_target_": assignment.selector_target,
                    "_partial_": True,
                }
            )
            assert isinstance(selector_cfg, DictConfig)
        training_factories.append(
            make_env_factory(
                cfg,
                worker_id=worker_id,
                output_directory=rundir / "smac3_output" / "train" / f"worker_{worker_id}",
                task_ids=task_ids,
                instance_selector_cfg=selector_cfg,
                protocol_metadata=protocol_metadata,
            )
        )
    if worker_assignments:
        logger.info(f"Persistent training worker assignments: {worker_assignments}")
    vec_env: VecEnv = _make_vec_env(training_factories, start_method=start_method)
    eval_env: VecEnv | None = None

    try:
        if cfg.experiment.vecnormalize:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
            )

        model: BaseAlgorithm = instantiate(optimizer_cfg)(
            env=vec_env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(rundir / "tensorboard"),
        )
        staged_worker_seeds = stage_training_worker_seeds(vec_env, int(cfg.seed))
        logger.info(f"Staged training worker seeds: {staged_worker_seeds}")
        logger.info(f"Model: {model.policy}")

        callbacks: list[BaseCallback] = [
            CheckpointCallback(
                save_freq=_callback_frequency(
                    int(cfg.experiment.checkpoint_freq),
                    schedule.n_envs,
                ),
                save_path=str(rundir),
                save_vecnormalize=bool(cfg.experiment.vecnormalize),
            ),
            ActionLoggingCallback(
                n_envs=schedule.n_envs,
                csv_path=str(rundir / "tensorboard" / "actions.csv"),
            ),
        ]

        validation_cfg = cfg.experiment.get("validation", None)
        eval_callback: EvalCallback | None = None
        if validation_cfg is not None and bool(validation_cfg.get("enabled", True)):
            validation_workers = int(validation_cfg.get("n_workers", 1))
            if validation_workers != 1:
                raise ValueError(
                    "Validation currently requires n_workers=1 so round-robin "
                    "evaluation covers every holdout context exactly once."
                )
            validation_factories = [
                make_env_factory(
                    cfg,
                    worker_id=worker_id,
                    output_directory=rundir / "smac3_output" / "validation" / f"worker_{worker_id}",
                    task_ids=list(validation_cfg.task_ids),
                    inner_seeds=list(validation_cfg.inner_seeds),
                    instance_set_id=str(validation_cfg.instance_set_id),
                    instance_selector_cfg=validation_cfg.instance_selector_class,
                    protocol_metadata=protocol_metadata,
                )
                for worker_id in range(validation_workers)
            ]
            eval_env = _make_vec_env(validation_factories, start_method=start_method)
            if cfg.experiment.vecnormalize:
                eval_env = VecNormalize(
                    eval_env,
                    norm_obs=True,
                    norm_reward=False,
                    training=False,
                )
            manifest_inner_seeds = [int(seed) for seed in validation_cfg.inner_seeds if seed is not None]
            if len(manifest_inner_seeds) != len(validation_cfg.inner_seeds):
                raise ValueError("Validation manifests must contain only frozen integer inner seeds.")
            eval_callback = ProtocolEvalCallback(
                eval_env,
                n_eval_episodes=int(validation_cfg.n_eval_episodes),
                eval_freq=_callback_frequency(
                    int(validation_cfg.eval_freq),
                    schedule.n_envs,
                ),
                deterministic=True,
                log_path=str(rundir / "validation"),
                warn=False,
                manifest_task_ids=list(validation_cfg.task_ids),
                manifest_inner_seeds=manifest_inner_seeds,
                protocol_save_path=rundir / "validation",
            )
            callbacks.append(eval_callback)

        logger.info("⚔ Start training...")
        model.learn(
            total_timesteps=schedule.total_timesteps,
            progress_bar=bool(cfg.experiment.get("progress_bar", True)),
            tb_log_name="tb_log",
            callback=callbacks,
        )
        _write_policy_sensitivity(model, rundir)
        model.save(rundir / "model")
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(rundir / "vecnormalize.pkl"))
        logger.info("✅ Finished training.")

        final_evaluation_cfg = cfg.experiment.get("final_evaluation", {})
        if not bool(final_evaluation_cfg.get("enabled", True)):
            logger.info("⏭ Skipping final evaluation as configured.")
            return

        logger.info("🧐 Evaluating learned policy...")
        final_eval_env = eval_env if eval_env is not None else vec_env
        if eval_env is not None and isinstance(vec_env, VecNormalize):
            sync_envs_normalization(vec_env, eval_env)
        if isinstance(final_eval_env, VecNormalize):
            final_eval_env.training = False
            final_eval_env.norm_reward = False

        if validation_cfg is not None and bool(validation_cfg.get("enabled", True)):
            n_eval_episodes = int(validation_cfg.n_eval_episodes)
            evaluation_label = "validation"
        else:
            n_eval_episodes = len(cfg.dacboenv.task_ids) * len(cfg.dacboenv.inner_seeds)
            evaluation_label = "training"

        mean_reward, std_reward = evaluate_policy(
            model,
            final_eval_env,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            warn=False,
        )
        logger.info(f"Learned policy {evaluation_label} reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        with (rundir / "modeleval.txt").open("a", encoding="utf-8") as out:
            out.write(f"Learned policy {evaluation_label} reward: {mean_reward:.2f} +/- {std_reward:.2f}\n")
    finally:
        if eval_env is not None:
            eval_env.close()
        vec_env.close()


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Hydra entry point for the shared Stable-Baselines3 PPO runner."""
    run(cfg)


if __name__ == "__main__":
    main()
