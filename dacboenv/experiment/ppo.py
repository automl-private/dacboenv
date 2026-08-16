"""Train Stable-Baselines3 PPO on DACBOEnv."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch as th
from carps.loggers.file_logger import get_run_directory
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
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
from dacboenv.experiment.paired_evaluator import (
    DistinctStateSubstitutionError,
    PolicyStateSample,
    policy_state_substitution_sensitivity,
)
from dacboenv.experiment.ppo_utils import (
    ActionLoggingCallback,
    categorical_policy_statistics,
    structured_policy_sensitivity,
)
from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
)
from dacboenv.experiment.sb3_algorithms import (
    DQNDiagnosticsCallback,
    GPHyperparameterDiagnosticsCallback,
    build_sb3_algorithm,
    resolve_rl_algorithm_id,
    write_algorithm_metadata,
)
from dacboenv.utils.carps_optimizer import get_task_config
from dacboenv.utils.loggingutils import maybe_remove_logs
from dacboenv.utils.seeding import derive_named_seed, run_seed_metadata

if TYPE_CHECKING:
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
    yahpo_scenario: str | None = None


@dataclass(frozen=True)
class ValidationScores:
    """Hierarchically aggregated fixed-manifest validation scores."""

    bbob_score: float | None
    yahpo_score: float | None
    balanced_score: float
    worst_domain_score: float
    per_task: dict[str, float]
    per_scenario: dict[str, float]
    per_dimension: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FullValidationCandidate:
    """One trained checkpoint nominated for the expensive full panel."""

    candidate_id: str
    training_step: int
    model_path: Path
    normalization_path: Path | None
    nomination_reasons: tuple[str, ...]


def aggregate_validation_scores(
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
    per_dimension: dict[int, float] = {}
    if bbob_by_function:
        scores_by_dimension: dict[int, list[float]] = {}
        for (dimension, _function_id), instance_scores in bbob_by_function.items():
            scores_by_dimension.setdefault(dimension, []).append(float(np.mean(instance_scores)))
        per_dimension = {dimension: float(np.mean(scores)) for dimension, scores in sorted(scores_by_dimension.items())}
        bbob_score = float(np.mean(list(per_dimension.values())))

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
        per_dimension=per_dimension,
    )


def evaluate_protocol_manifest(
    model: BaseAlgorithm,
    eval_env: VecEnv,
    *,
    task_ids: list[str],
    inner_seeds: list[int],
) -> tuple[ValidationScores, list[float], list[int]]:
    """Evaluate a fixed manifest from context zero and aggregate hierarchically."""
    eval_env.env_method("restart_fixed_instance_sequence")
    episode_rewards, episode_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=len(task_ids) * len(inner_seeds),
        deterministic=True,
        return_episode_rewards=True,
        warn=False,
    )
    rewards = [float(value) for value in episode_rewards]
    lengths = [int(value) for value in episode_lengths]
    return aggregate_validation_scores(task_ids, inner_seeds, rewards), rewards, lengths


def run_step_zero_validation(
    model: BaseAlgorithm,
    training_env: VecEnv,
    eval_env: VecEnv,
    *,
    task_ids: list[str],
    inner_seeds: list[int],
    save_path: Path,
    panel_id: str,
    panel_hash: str,
) -> ValidationScores:
    """Evaluate and persist the untrained policy outside model selection."""
    if isinstance(training_env, VecNormalize):
        sync_envs_normalization(training_env, eval_env)
    if isinstance(eval_env, VecNormalize):
        eval_env.training = False
        eval_env.norm_reward = False

    scores, episode_rewards, episode_lengths = evaluate_protocol_manifest(
        model,
        eval_env,
        task_ids=task_ids,
        inner_seeds=inner_seeds,
    )
    save_path.mkdir(parents=True, exist_ok=True)
    model.save(save_path / "untrained_model")
    if isinstance(eval_env, VecNormalize):
        eval_env.save(str(save_path / "vecnormalize.pkl"))
    payload = {
        "rl_algorithm_id": str(getattr(model, "algorithm_id", type(model).__name__.lower())),
        "algorithm_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "selection_eligible": False,
        "panel_tier": "frequent",
        "panel_id": panel_id,
        "panel_hash": panel_hash,
        "num_timesteps": 0,
        "scores": {
            "balanced": scores.balanced_score,
            "bbob": scores.bbob_score,
            "yahpo": scores.yahpo_score,
            "worst_domain": scores.worst_domain_score,
            "per_task": scores.per_task,
            "per_scenario": scores.per_scenario,
            "per_dimension": scores.per_dimension,
        },
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }
    (save_path / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return scores


def build_trained_vs_step_zero_comparison(
    *,
    step_zero_scores: ValidationScores,
    final_scores: ValidationScores,
    selected_full: dict[str, Any] | None,
    frequent_history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare trained checkpoints with step zero on the frequent panel.

    A final checkpoint need not coincide with a periodic frequent-validation
    callback.  Its score is nevertheless available from the mandatory final
    frequent-panel evaluation.  Treating that case as missing used to report a
    false negative for ``best_trained_improves_over_step_zero``.
    """
    selected_frequent_score: float | None = None
    selected_score_source: str | None = None
    if selected_full is not None:
        selected_step = int(selected_full["training_step"])
        selected_frequent = next(
            (entry for entry in frequent_history if int(entry["training_step"]) == selected_step),
            None,
        )
        if selected_frequent is not None:
            selected_frequent_score = float(selected_frequent["scores"]["balanced"])
            selected_score_source = "periodic_frequent_checkpoint"
        elif str(selected_full.get("candidate_id", "")) == "final":
            selected_frequent_score = final_scores.balanced_score
            selected_score_source = "final_frequent_evaluation"

    return {
        "step_zero_is_selection_eligible": False,
        "comparison_panel_tier": "frequent",
        "step_zero_balanced_score": step_zero_scores.balanced_score,
        "final_balanced_score": final_scores.balanced_score,
        "final_improves_over_step_zero": final_scores.balanced_score > step_zero_scores.balanced_score,
        "full_selected_balanced_score": None if selected_full is None else selected_full["score"],
        "full_selected_checkpoint_frequent_score": selected_frequent_score,
        "full_selected_checkpoint_frequent_score_source": selected_score_source,
        "best_trained_improves_over_step_zero": (
            None if selected_frequent_score is None else selected_frequent_score > step_zero_scores.balanced_score
        ),
    }


class ProtocolEvalCallback(EvalCallback):
    """Frequent-panel screening with replayable trained checkpoints."""

    def __init__(
        self,
        *args: Any,
        manifest_task_ids: list[str],
        manifest_inner_seeds: list[int],
        protocol_save_path: Path,
        panel_id: str,
        panel_hash: str,
        **kwargs: Any,
    ) -> None:
        # Disable EvalCallback's raw-episode-mean checkpoint. Protocol scores
        # are computed after its reproducible evaluation and saved below.
        kwargs["best_model_save_path"] = None
        super().__init__(*args, **kwargs)
        self._manifest_task_ids = manifest_task_ids
        self._manifest_inner_seeds = manifest_inner_seeds
        self._protocol_save_path = protocol_save_path
        self._panel_id = panel_id
        self._panel_hash = panel_hash
        self._best_scores = {"balanced": -np.inf, "bbob": -np.inf, "yahpo": -np.inf}
        self.frequent_history: list[dict[str, Any]] = []

    def _save_frequent_checkpoint(self, scores: ValidationScores) -> None:
        checkpoint_directory = self._protocol_save_path / "frequent" / "checkpoints"
        checkpoint_directory.mkdir(parents=True, exist_ok=True)
        checkpoint_stem = checkpoint_directory / f"step_{self.num_timesteps}_model"
        self.model.save(checkpoint_stem)
        vecnormalize = self.model.get_vec_normalize_env()
        normalization_path: Path | None = None
        if vecnormalize is not None:
            normalization_path = checkpoint_directory / f"step_{self.num_timesteps}_vecnormalize.pkl"
            vecnormalize.save(str(normalization_path))
        entry = {
            "rl_algorithm_id": str(getattr(self.model, "algorithm_id", type(self.model).__name__.lower())),
            "algorithm_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "panel_id": self._panel_id,
            "panel_hash": self._panel_hash,
            "training_step": int(self.num_timesteps),
            "model_path": str(checkpoint_stem.with_suffix(".zip")),
            "normalization_path": None if normalization_path is None else str(normalization_path),
            "scores": {
                "balanced": scores.balanced_score,
                "bbob": scores.bbob_score,
                "yahpo": scores.yahpo_score,
                "worst_domain": scores.worst_domain_score,
                "per_task": scores.per_task,
                "per_scenario": scores.per_scenario,
                "per_dimension": scores.per_dimension,
            },
        }
        self.frequent_history.append(entry)
        history_path = self._protocol_save_path / "frequent" / "history.json"
        history_path.write_text(
            json.dumps({"panel_tier": "frequent", "checkpoints": self.frequent_history}, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _metric_tag(value: str) -> str:
        return value.replace("/", "_").replace(" ", "_")

    def _record_validation_scores(self, scores: ValidationScores) -> None:
        """Record aggregates everywhere and high-cardinality metrics losslessly."""
        self.logger.record("eval/balanced_score", scores.balanced_score)
        self.logger.record("eval/worst_domain_score", scores.worst_domain_score)
        if scores.bbob_score is not None:
            self.logger.record("eval/bbob_score", scores.bbob_score)
        if scores.yahpo_score is not None:
            self.logger.record("eval/yahpo_score", scores.yahpo_score)

        # SB3's human-readable writers truncate keys and reject collisions.
        # CSV and TensorBoard retain these complete stable metric names.
        machine_readable_only = ("stdout", "log")
        for task_id, score in scores.per_task.items():
            self.logger.record(
                f"eval/per_task/{self._metric_tag(task_id)}",
                score,
                exclude=machine_readable_only,
            )
        for scenario, score in scores.per_scenario.items():
            self.logger.record(
                f"eval/per_scenario/{self._metric_tag(scenario)}",
                score,
                exclude=machine_readable_only,
            )

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
        self._record_validation_scores(scores)

        checkpoint_scores = {
            "balanced": scores.balanced_score,
            "bbob": scores.bbob_score,
            "yahpo": scores.yahpo_score,
        }
        for label, score in checkpoint_scores.items():
            if score is not None and score > self._best_scores[label]:
                self._best_scores[label] = score
        self._save_frequent_checkpoint(scores)
        self.logger.dump(self.num_timesteps)
        return continue_training


def nominate_full_validation_candidates(  # noqa: C901, PLR0912
    frequent_history: list[dict[str, Any]],
    *,
    final_model_path: Path,
    final_normalization_path: Path | None,
    final_training_step: int,
    top_k: int = 3,
    include_halfway: bool = True,
    include_final: bool = True,
    manual_steps: list[int] | None = None,
) -> tuple[FullValidationCandidate, ...]:
    """Nominate top frequent, halfway, final, and explicit trained checkpoints."""
    if top_k < 0:
        raise ValueError("full validation top_k must be non-negative.")
    if final_training_step <= 0:
        raise ValueError("Full validation requires a positive trained final step.")
    by_step: dict[int, dict[str, Any]] = {}
    for entry in frequent_history:
        step = int(entry["training_step"])
        if step <= 0 or step in by_step:
            raise ValueError("Frequent validation history must contain unique positive training steps.")
        by_step[step] = entry

    reasons: dict[int, set[str]] = {}
    ranked = sorted(
        frequent_history,
        key=lambda entry: (-float(entry["scores"]["balanced"]), int(entry["training_step"])),
    )
    for entry in ranked[:top_k]:
        reasons.setdefault(int(entry["training_step"]), set()).add("top_frequent")
    if include_halfway and by_step:
        halfway = final_training_step / 2.0
        halfway_step = min(by_step, key=lambda step: (abs(step - halfway), step))
        reasons.setdefault(halfway_step, set()).add("approximately_halfway")
    for step in manual_steps or []:
        if step not in by_step:
            raise ValueError(f"Manual full-validation step {step} has no saved frequent checkpoint.")
        reasons.setdefault(step, set()).add("manual")
    if include_final:
        reasons.setdefault(final_training_step, set()).add("final")

    candidates: list[FullValidationCandidate] = []
    for step in sorted(reasons):
        if step == final_training_step and include_final:
            model_path = final_model_path
            normalization_path = final_normalization_path
            candidate_id = "final"
        else:
            entry = by_step[step]
            model_path = Path(str(entry["model_path"]))
            raw_normalization = entry.get("normalization_path")
            normalization_path = None if raw_normalization is None else Path(str(raw_normalization))
            candidate_id = f"step_{step}"
        if not model_path.is_file():
            raise FileNotFoundError(f"Nominated full-validation model is missing: {model_path}")
        if final_normalization_path is not None and normalization_path is None:
            raise FileNotFoundError(f"Checkpoint-specific normalization is missing for {candidate_id}.")
        if normalization_path is not None and not normalization_path.is_file():
            raise FileNotFoundError(f"Nominated normalization state is missing: {normalization_path}")
        candidates.append(
            FullValidationCandidate(
                candidate_id=candidate_id,
                training_step=step,
                model_path=model_path,
                normalization_path=normalization_path,
                nomination_reasons=tuple(sorted(reasons[step])),
            )
        )
    if not candidates:
        raise ValueError("No trained checkpoint was nominated for full validation.")
    return tuple(candidates)


def _score_payload(scores: ValidationScores) -> dict[str, Any]:
    return {
        "balanced": scores.balanced_score,
        "bbob": scores.bbob_score,
        "yahpo": scores.yahpo_score,
        "worst_domain": scores.worst_domain_score,
        "per_task": scores.per_task,
        "per_scenario": scores.per_scenario,
        "per_dimension": scores.per_dimension,
    }


def _select_full_result(results: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    eligible = [result for result in results if result["scores"].get(label) is not None]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda result: (-float(result["scores"][label]), int(result["training_step"]), result["candidate_id"]),
    )[0]


def run_full_panel_validation(
    model: BaseAlgorithm,
    cfg: DictConfig,
    *,
    validation_cfg: DictConfig,
    frequent_history: list[dict[str, Any]],
    rundir: Path,
    protocol_metadata: dict[str, Any],
    start_method: str | None,
) -> dict[str, Any]:
    """Evaluate only nominated trained checkpoints and export full-panel bests."""
    full_task_ids = list(validation_cfg.full_task_ids)
    full_inner_seeds = [int(seed) for seed in validation_cfg.full_inner_seeds if seed is not None]
    if len(full_inner_seeds) != len(validation_cfg.full_inner_seeds):
        raise ValueError("Full validation manifests must contain only frozen integer inner seeds.")
    expected_episodes = len(full_task_ids) * len(full_inner_seeds)
    if expected_episodes != int(validation_cfg.full_n_eval_episodes):
        raise ValueError("Full validation episode count does not match its frozen task/seed product.")

    final_model_path = rundir / "model.zip"
    final_normalization_path = rundir / "vecnormalize.pkl" if isinstance(model.get_env(), VecNormalize) else None
    candidates = nominate_full_validation_candidates(
        frequent_history,
        final_model_path=final_model_path,
        final_normalization_path=final_normalization_path,
        final_training_step=int(model.num_timesteps),
        top_k=int(validation_cfg.get("full_top_k", 3)),
        include_halfway=bool(validation_cfg.get("full_include_halfway", True)),
        include_final=bool(validation_cfg.get("full_include_final", True)),
        manual_steps=[int(step) for step in validation_cfg.get("full_manual_steps", [])],
    )

    results: list[dict[str, Any]] = []
    for candidate in candidates:
        full_factories = [
            make_env_factory(
                cfg,
                worker_id=0,
                output_directory=rundir / "smac3_output" / "validation_full" / candidate.candidate_id,
                task_ids=full_task_ids,
                inner_seeds=full_inner_seeds,
                instance_set_id=str(validation_cfg.full_instance_set_id),
                instance_selector_cfg=validation_cfg.instance_selector_class,
                protocol_metadata=protocol_metadata,
                context_split="validation",
            )
        ]
        candidate_env: VecEnv = _make_vec_env(full_factories, start_method=start_method)
        try:
            if candidate.normalization_path is not None:
                candidate_env = VecNormalize.load(str(candidate.normalization_path), candidate_env)
                candidate_env.training = False
                candidate_env.norm_reward = False
            candidate_model = type(model).load(str(candidate.model_path))
            scores, rewards, lengths = evaluate_protocol_manifest(
                candidate_model,
                candidate_env,
                task_ids=full_task_ids,
                inner_seeds=full_inner_seeds,
            )
        finally:
            candidate_env.close()
        result = {
            "rl_algorithm_id": str(getattr(model, "algorithm_id", type(model).__name__.lower())),
            "algorithm_class": f"{type(model).__module__}.{type(model).__qualname__}",
            "candidate_id": candidate.candidate_id,
            "training_step": candidate.training_step,
            "model_path": str(candidate.model_path),
            "normalization_path": (None if candidate.normalization_path is None else str(candidate.normalization_path)),
            "nomination_reasons": list(candidate.nomination_reasons),
            "scores": _score_payload(scores),
            "episode_rewards": rewards,
            "episode_lengths": lengths,
        }
        results.append(result)
        candidate_directory = rundir / "validation" / "full" / candidate.candidate_id
        candidate_directory.mkdir(parents=True, exist_ok=True)
        (candidate_directory / "metrics.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    selections: dict[str, Any] = {}
    for label in ("balanced", "bbob", "yahpo"):
        selected = _select_full_result(results, label)
        if selected is None:
            continue
        selections[label] = {
            "candidate_id": selected["candidate_id"],
            "training_step": selected["training_step"],
            "score": selected["scores"][label],
        }
        destination = rundir / "validation" / f"best_{label}_model.zip"
        shutil.copy2(selected["model_path"], destination)
        if selected["normalization_path"] is not None:
            shutil.copy2(
                selected["normalization_path"],
                rundir / "validation" / f"best_{label}_vecnormalize.pkl",
            )
        if label == "balanced":
            shutil.copy2(selected["model_path"], rundir / "validation" / "best_model.zip")

    payload = {
        "panel_tier": "full",
        "manifest_id": str(validation_cfg.full_instance_set_id),
        "manifest_hash": str(validation_cfg.full_manifest_hash),
        "episode_count": expected_episodes,
        "trained_checkpoints_only": True,
        "results": results,
        "selections": selections,
    }
    full_directory = rundir / "validation" / "full"
    full_directory.mkdir(parents=True, exist_ok=True)
    (full_directory / "selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def assign_training_worker_context(  # noqa: C901
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
            local_worker_id = worker_id - n_bbob_workers
            scenarios = sorted({task_id.split("/")[2] for task_id in yahpo_tasks})
            n_yahpo_workers = n_workers - n_bbob_workers
            if n_yahpo_workers < len(scenarios):
                raise ValueError(
                    "Persistent mixed YAHPO sampling requires at least one worker per scenario; "
                    f"got {n_yahpo_workers} workers for {len(scenarios)} scenarios."
                )
            scenario = scenarios[local_worker_id % len(scenarios)]
            return WorkerContextAssignment(
                domain="yahpo",
                task_ids=[task_id for task_id in yahpo_tasks if task_id.split("/")[2] == scenario],
                selector_target="dacboenv.env.instance.HierarchicalYAHPOInstanceSelector",
                yahpo_scenario=scenario,
            )
    elif yahpo_tasks:
        scenarios = sorted({task_id.split("/")[2] for task_id in yahpo_tasks})
        if n_workers < len(scenarios):
            raise ValueError(
                "Persistent YAHPO sampling requires at least one worker per scenario; "
                f"got {n_workers} workers for {len(scenarios)} scenarios."
            )
        scenario = scenarios[worker_id % len(scenarios)]
        return WorkerContextAssignment(
            domain="yahpo",
            task_ids=[task_id for task_id in yahpo_tasks if task_id.split("/")[2] == scenario],
            selector_target="dacboenv.env.instance.HierarchicalYAHPOInstanceSelector",
            yahpo_scenario=scenario,
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
    algorithm_id = resolve_rl_algorithm_id(cfg)
    if algorithm_id == "ppo":
        n_steps = int(cfg.optimizer.n_steps)
        batch_size = int(cfg.optimizer.batch_size)
    else:
        algorithm_hyperparameters = cfg.rl_algorithm.hyperparameters
        train_freq = algorithm_hyperparameters.get("train_freq", [1, "step"])
        n_steps = int(
            train_freq[0] if isinstance(train_freq, Sequence) and not isinstance(train_freq, str) else train_freq
        )
        batch_size = int(algorithm_hyperparameters.batch_size)
    total_timesteps = int(cfg.experiment.total_timesteps)

    if n_envs <= 0 or n_steps <= 0 or batch_size <= 0 or total_timesteps <= 0:
        raise ValueError("n_workers, n_steps, batch_size, and total_timesteps must all be positive.")

    rollout_size = n_envs * n_steps
    if algorithm_id == "ppo" and batch_size > rollout_size:
        raise ValueError(f"batch_size ({batch_size}) exceeds rollout size ({rollout_size}).")
    if algorithm_id == "ppo" and rollout_size % batch_size:
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
    if resolve_rl_algorithm_id(cfg) != "ppo":
        return
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
    context_split: str = "train",
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
        with open_dict(config.dacboenv):
            config.dacboenv.context_split = context_split
            if context_split != "train":
                config.dacboenv.yahpo_training_budget_multiplier = 1.0

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
        ("frequent_validation_panel", "frequent_validation"),
        ("full_validation_panel", "full_validation"),
    ):
        manifest = cfg.get(config_key, None)
        if manifest is None:
            continue
        plain_manifest = OmegaConf.to_container(manifest, resolve=True)
        if not isinstance(plain_manifest, dict):
            raise TypeError(f"{config_key} must be a manifest mapping.")
        if plain_manifest.get("schema_version", None) is not None:
            validate_manifest_structure(plain_manifest)
            # Test manifests are loaded only to pin their identity in training
            # metadata. A sealed/non-runnable test inventory must not be
            # evaluated or prevent an otherwise valid non-test pilot.
            if config_key != "test_instances":
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


_POLICY_SENSITIVITY_INTERVENTIONS = (
    "zero_global_state",
    "permute_global_features",
    "mean_action_features",
    "permute_action_rows",
    "state_from_another_task",
    "state_from_another_budget_phase",
)
_MATRIX_NDIM = 2


def build_policy_sensitivity_report(  # noqa: C901
    fallback_observation: Mapping[str, np.ndarray],
    probability_function: Callable[[dict[str, np.ndarray]], np.ndarray],
    *,
    state_samples: Sequence[Mapping[str, Any]] = (),
    deterministic_constant_episode_fraction: float | None = None,
) -> dict[str, Any]:
    """Build a complete, fail-closed state-sensitivity artifact.

    Cross-state substitutions are selected through explicit task and budget
    provenance. If the rollout reservoir has no provably distinct source, the
    corresponding intervention is recorded as unavailable rather than being
    synthesized from an arbitrary worker shift.
    """
    fallback = {name: np.asarray(value).copy() for name, value in fallback_observation.items()}
    if set(fallback) != {"global_state", "action_features"}:
        raise ValueError("Policy sensitivity requires global_state and action_features.")
    if fallback["global_state"].ndim < 2 or fallback["action_features"].ndim < 3:  # noqa: PLR2004
        raise ValueError("Fallback policy observations must include a leading batch dimension.")

    samples = [
        PolicyStateSample(
            task_id=str(sample["task_id"]),
            budget_fraction=float(sample["budget_fraction"]),
            observation={name: np.asarray(value).copy() for name, value in sample["observation"].items()},
        )
        for sample in state_samples
    ]

    def single_probability(observation: Mapping[str, np.ndarray]) -> np.ndarray:
        probabilities = np.asarray(
            probability_function({name: np.asarray(value).copy() for name, value in observation.items()}),
            dtype=float,
        )
        if probabilities.ndim == _MATRIX_NDIM and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        return probabilities

    substitution_results = None
    reference_index: int | None = None
    substitution_error = "No provenance-bearing rollout states were captured."
    for candidate_index in range(len(samples)):
        try:
            substitution_results = policy_state_substitution_sensitivity(
                samples,
                single_probability,
                reference_index=candidate_index,
            )
        except DistinctStateSubstitutionError as error:
            substitution_error = str(error)
            continue
        reference_index = candidate_index
        break

    if reference_index is None:
        reference_observation = {name: value[:1].copy() for name, value in fallback.items()}
        another_task = None
        another_phase = None
        provenance: dict[str, Any] = {
            "status": "incomplete",
            "reason": substitution_error,
        }
    else:
        assert substitution_results is not None
        reference_sample = samples[reference_index]

        def batched_sample(source_index: int) -> dict[str, np.ndarray]:
            return {
                name: np.asarray(value)[None, ...].copy() for name, value in samples[source_index].observation.items()
            }

        task_result = substitution_results["state_from_another_task"]
        phase_result = substitution_results["state_from_another_budget_phase"]
        reference_observation = {
            name: np.asarray(value)[None, ...].copy() for name, value in reference_sample.observation.items()
        }
        another_task = batched_sample(task_result.source_index)
        another_phase = batched_sample(phase_result.source_index)
        provenance = {
            "status": "complete",
            "reference_index": reference_index,
            "reference_task_id": reference_sample.task_id,
            "reference_budget_fraction": reference_sample.budget_fraction,
            "reference_budget_phase": reference_sample.budget_phase,
            "state_from_another_task": {
                "source_index": task_result.source_index,
                "source_task_id": task_result.source_task_id,
                "source_budget_fraction": task_result.source_budget_fraction,
                "source_budget_phase": task_result.source_budget_phase,
                "task_changed": task_result.task_changed,
                "budget_phase_changed": task_result.budget_phase_changed,
            },
            "state_from_another_budget_phase": {
                "source_index": phase_result.source_index,
                "source_task_id": phase_result.source_task_id,
                "source_budget_fraction": phase_result.source_budget_fraction,
                "source_budget_phase": phase_result.source_budget_phase,
                "task_changed": phase_result.task_changed,
                "budget_phase_changed": phase_result.budget_phase_changed,
            },
        }

    interventions: dict[str, Any] = structured_policy_sensitivity(
        reference_observation,
        probability_function,
        state_from_another_task=another_task,
        state_from_another_budget_phase=another_phase,
    )
    for name in _POLICY_SENSITIVITY_INTERVENTIONS:
        interventions.setdefault(
            name,
            {
                "status": "unavailable",
                "reason": substitution_error,
            },
        )

    if samples:
        compatible = [
            sample
            for sample in samples
            if all(
                np.asarray(sample.observation[name]).shape == reference_observation[name].shape[1:]
                for name in reference_observation
            )
        ]
        state_batch = (
            {
                name: np.stack([np.asarray(sample.observation[name]) for sample in compatible])
                for name in reference_observation
            }
            if compatible
            else reference_observation
        )
    else:
        state_batch = reference_observation
    probabilities = probability_function(state_batch)
    logit_std = categorical_policy_statistics(probabilities).logit_std_across_states
    return {
        "schema_version": 1,
        "state_sample_count": len(samples),
        "substitution_provenance": provenance,
        "interventions": interventions,
        "summary": {
            "logit_std_across_states": logit_std,
            "deterministic_constant_episode_fraction": deterministic_constant_episode_fraction,
        },
    }


def _write_policy_sensitivity(
    model: BaseAlgorithm,
    rundir: Path,
    diagnostics_callback: ActionLoggingCallback,
) -> None:
    """Evaluate structured-policy interventions on captured real rollout states."""
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

    sensitivity = build_policy_sensitivity_report(
        numpy_observation,
        probability_function,
        state_samples=diagnostics_callback.sensitivity_state_samples,
        deterministic_constant_episode_fraction=(diagnostics_callback.deterministic_constant_episode_fraction),
    )
    (rundir / "policy_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for intervention, metrics in sensitivity["interventions"].items():
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                model.logger.record(f"policy_sensitivity/{intervention}/{metric}", value)
    for metric, value in sensitivity["summary"].items():
        if value is not None:
            model.logger.record(f"policy_sensitivity/summary/{metric}", value)
    model.logger.dump(model.num_timesteps)


def run(cfg: DictConfig) -> None:  # noqa: C901, PLR0912, PLR0915
    """Train and validate a configured SB3 algorithm on the BO-MDP."""
    logger.info(OmegaConf.to_yaml(cfg))

    rundir = Path(get_run_directory())
    maybe_remove_logs(directory=None, overwrite=True, logfile="model.zip", logger=logger)
    populate_legacy_schedule(cfg)
    schedule = resolve_training_schedule(cfg)
    algorithm_id = resolve_rl_algorithm_id(cfg)
    seed_metadata = run_seed_metadata(int(cfg.seed), schedule.n_envs)
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
        f"{algorithm_id} schedule: "
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
        assignment_payload = [
            {
                "worker_id": worker_id,
                "domain": assignment.domain,
                "bbob_dimension": assignment.bbob_dimension,
                "yahpo_scenario": assignment.yahpo_scenario,
                "task_ids": assignment.task_ids,
                "selector_target": assignment.selector_target,
            }
            for worker_id, assignment in enumerate(worker_assignments)
        ]
        (rundir / "training_worker_assignments.json").write_text(
            json.dumps(assignment_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    vec_env: VecEnv = _make_vec_env(training_factories, start_method=start_method)
    eval_env: VecEnv | None = None

    try:
        if cfg.experiment.vecnormalize:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
            )

        model: BaseAlgorithm = build_sb3_algorithm(
            cfg,
            vec_env,
            tensorboard_log=str(rundir / "tensorboard"),
            model_seed=int(seed_metadata["policy_model_seed"]),
        )
        algorithm_metadata = write_algorithm_metadata(cfg, vec_env, rundir / "rl_algorithm_metadata.json")
        staged_worker_seeds = stage_training_worker_seeds(vec_env, int(cfg.seed))
        logger.info(f"Staged training worker seeds: {staged_worker_seeds}")
        logger.info(f"Model: {model.policy}")

        action_logging_callback = ActionLoggingCallback(
            n_envs=schedule.n_envs,
            csv_path=str(rundir / "tensorboard" / "actions.csv"),
        )
        callbacks: list[BaseCallback] = [
            CheckpointCallback(
                save_freq=_callback_frequency(
                    int(cfg.experiment.checkpoint_freq),
                    schedule.n_envs,
                ),
                save_path=str(rundir),
                save_vecnormalize=bool(cfg.experiment.vecnormalize),
                save_replay_buffer=bool(cfg.experiment.get("save_replay_buffer", {}).get("periodic", False)),
            ),
            action_logging_callback,
            GPHyperparameterDiagnosticsCallback(),
        ]
        if algorithm_id in {"dqn", "double_dqn"}:
            callbacks.append(DQNDiagnosticsCallback())

        validation_cfg = cfg.experiment.get("validation", None)
        eval_callback: EvalCallback | None = None
        step_zero_scores: ValidationScores | None = None
        manifest_inner_seeds: list[int] = []
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
                    context_split="validation",
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
                log_path=str(rundir / "validation" / "frequent"),
                warn=False,
                manifest_task_ids=list(validation_cfg.task_ids),
                manifest_inner_seeds=manifest_inner_seeds,
                protocol_save_path=rundir / "validation",
                panel_id=str(validation_cfg.instance_set_id),
                panel_hash=str(validation_cfg.manifest_hash),
            )
            callbacks.append(eval_callback)

            if bool(validation_cfg.get("step_zero", True)):
                step_zero_scores = run_step_zero_validation(
                    model,
                    vec_env,
                    eval_env,
                    task_ids=list(validation_cfg.task_ids),
                    inner_seeds=manifest_inner_seeds,
                    save_path=rundir / "validation" / "step_zero",
                    panel_id=str(validation_cfg.instance_set_id),
                    panel_hash=str(validation_cfg.manifest_hash),
                )
                logger.info(
                    "Step-zero validation (diagnostic, selection-ineligible): "
                    f"balanced={step_zero_scores.balanced_score:.6g}."
                )

        logger.info("⚔ Start training...")
        model.learn(
            total_timesteps=schedule.total_timesteps,
            progress_bar=bool(cfg.experiment.get("progress_bar", True)),
            tb_log_name="tb_log",
            callback=callbacks,
        )
        if int(model.num_timesteps) != schedule.total_timesteps:
            raise RuntimeError(
                "SB3 training did not stop at the configured exact final timestep: "
                f"expected {schedule.total_timesteps}, reached {model.num_timesteps}. "
                "Choose a total_timesteps value divisible by the number of vector environments."
            )
        if algorithm_id == "ppo":
            _write_policy_sensitivity(model, rundir, action_logging_callback)
        model.save(rundir / "model")
        save_replay_buffer = bool(cfg.experiment.get("save_replay_buffer", {}).get("final", False))
        if save_replay_buffer and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(rundir / "replay_buffer.pkl")
        if isinstance(vec_env, VecNormalize):
            vec_env.save(str(rundir / "vecnormalize.pkl"))
        logger.info("✅ Finished training.")
        (rundir / "training_complete.json").write_text(
            json.dumps(
                {
                    "complete": True,
                    "rl_algorithm_id": algorithm_id,
                    "algorithm_class": algorithm_metadata["algorithm_class"],
                    "num_timesteps": int(model.num_timesteps),
                    "expected_final_timesteps": int(schedule.total_timesteps),
                    "model_path": str(rundir / "model.zip"),
                    "replay_buffer_path": str(rundir / "replay_buffer.pkl") if save_replay_buffer else None,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        full_validation_payload: dict[str, Any] | None = None
        if (
            validation_cfg is not None
            and bool(validation_cfg.get("enabled", True))
            and bool(validation_cfg.get("full_enabled", True))
        ):
            if not isinstance(eval_callback, ProtocolEvalCallback):
                raise RuntimeError("Full validation requires the frequent protocol callback history.")
            full_validation_payload = run_full_panel_validation(
                model,
                cfg,
                validation_cfg=validation_cfg,
                frequent_history=eval_callback.frequent_history,
                rundir=rundir,
                protocol_metadata=protocol_metadata,
                start_method=start_method,
            )
            logger.info(
                "Completed full-panel validation for "
                f"{len(full_validation_payload['results'])} nominated trained checkpoints."
            )

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

        if validation_cfg is not None and bool(validation_cfg.get("enabled", True)):
            final_scores, final_rewards, _final_lengths = evaluate_protocol_manifest(
                model,
                final_eval_env,
                task_ids=list(validation_cfg.task_ids),
                inner_seeds=manifest_inner_seeds,
            )
            mean_reward = float(np.mean(final_rewards))
            std_reward = float(np.std(final_rewards))
            if step_zero_scores is not None:
                selected_full = (
                    None if full_validation_payload is None else full_validation_payload["selections"].get("balanced")
                )
                frequent_history = (
                    eval_callback.frequent_history if isinstance(eval_callback, ProtocolEvalCallback) else []
                )
                comparison = build_trained_vs_step_zero_comparison(
                    step_zero_scores=step_zero_scores,
                    final_scores=final_scores,
                    selected_full=selected_full,
                    frequent_history=frequent_history,
                )
                (rundir / "validation" / "trained_vs_step_zero.json").write_text(
                    json.dumps(comparison, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
        else:
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
