"""Utils for PPO."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from carps.loggers.file_logger import get_run_directory
from stable_baselines3.common.callbacks import BaseCallback

MATRIX_NDIM = 2
MIN_CATEGORICAL_ACTIONS = 2


@dataclass(frozen=True)
class CategoricalPolicyStatistics:
    """Low-cost summaries of a batch of categorical policy outputs."""

    mean_probabilities: np.ndarray
    normalized_entropy: float
    max_probability: float
    top1_top2_logit_gap: float
    deterministic_actions: np.ndarray


def _probability_matrix(probabilities: Any, *, name: str) -> np.ndarray:
    """Validate categorical probabilities and flatten leading batch axes."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim == 1:
        values = values[None, :]
    elif values.ndim > MATRIX_NDIM:
        values = values.reshape(-1, values.shape[-1])
    if values.ndim != MATRIX_NDIM or values.shape[1] < MIN_CATEGORICAL_ACTIONS:
        raise ValueError(f"{name} must have shape (..., n_actions) with n_actions >= 2, got {values.shape}.")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError(f"{name} must contain finite, non-negative probabilities.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError(f"Every row of {name} must sum to one.")
    return values


def categorical_policy_statistics(
    probabilities: Any,
    logits: Any | None = None,
) -> CategoricalPolicyStatistics:
    """Summarize categorical outputs across an observation batch.

    The entropy is divided by ``log(n_actions)`` and therefore lies in
    ``[0, 1]``. When logits are unavailable, log-probabilities provide the
    same top-one/top-two gap because softmax only adds a common offset.
    """
    probability_matrix = _probability_matrix(probabilities, name="probabilities")
    if logits is None:
        logit_matrix = np.log(np.maximum(probability_matrix, np.finfo(np.float64).tiny))
    else:
        logit_matrix = np.asarray(logits, dtype=np.float64)
        if logit_matrix.ndim == 1:
            logit_matrix = logit_matrix[None, :]
        elif logit_matrix.ndim > MATRIX_NDIM:
            logit_matrix = logit_matrix.reshape(-1, logit_matrix.shape[-1])
        if logit_matrix.shape != probability_matrix.shape or not np.isfinite(logit_matrix).all():
            raise ValueError("logits must be finite and have the same shape as probabilities.")

    positive = probability_matrix > 0.0
    log_probabilities = np.zeros_like(probability_matrix)
    log_probabilities[positive] = np.log(probability_matrix[positive])
    entropy = -np.sum(probability_matrix * log_probabilities, axis=1)
    normalized_entropy = entropy / np.log(probability_matrix.shape[1])
    top_two_logits = np.partition(logit_matrix, kth=-2, axis=1)[:, -2:]
    top1_top2_gap = top_two_logits[:, 1] - top_two_logits[:, 0]

    return CategoricalPolicyStatistics(
        mean_probabilities=np.mean(probability_matrix, axis=0),
        normalized_entropy=float(np.mean(normalized_entropy)),
        max_probability=float(np.mean(np.max(probability_matrix, axis=1))),
        top1_top2_logit_gap=float(np.mean(top1_top2_gap)),
        deterministic_actions=np.argmax(probability_matrix, axis=1),
    )


def policy_sensitivity_metrics(
    reference_probabilities: Any,
    perturbed_probabilities: Any,
    *,
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Compare categorical policies with directional KL and total variation.

    KL is ``KL(reference || perturbed)`` and both metrics are averaged over
    all leading batch dimensions. Clipping only protects the logarithm at
    exact zero; ordinary softmax probabilities are unaffected.
    """
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}.")
    reference_shape = np.asarray(reference_probabilities).shape
    perturbed_shape = np.asarray(perturbed_probabilities).shape
    if reference_shape != perturbed_shape:
        raise ValueError(
            "reference_probabilities and perturbed_probabilities must have "
            f"the same shape, got {reference_shape} and {perturbed_shape}."
        )

    reference = _probability_matrix(reference_probabilities, name="reference_probabilities")
    perturbed = _probability_matrix(perturbed_probabilities, name="perturbed_probabilities")
    positive = reference > 0.0
    kl_terms = np.zeros_like(reference)
    kl_terms[positive] = reference[positive] * (
        np.log(reference[positive]) - np.log(np.maximum(perturbed[positive], epsilon))
    )
    total_variation = 0.5 * np.sum(np.abs(reference - perturbed), axis=1)
    return {
        "mean_kl": float(np.mean(np.sum(kl_terms, axis=1))),
        "mean_total_variation": float(np.mean(total_variation)),
    }


def structured_policy_sensitivity(
    observation: dict[str, np.ndarray],
    probability_function: Callable[[dict[str, np.ndarray]], np.ndarray],
) -> dict[str, dict[str, float]]:
    """Evaluate the requested structured-state interventions on real states."""
    if set(observation) != {"global_state", "action_features"}:
        raise ValueError("Structured sensitivity requires exactly global_state and action_features.")
    global_state = np.asarray(observation["global_state"])
    action_features = np.asarray(observation["action_features"])
    if global_state.ndim < 2 or action_features.ndim < 3:  # noqa: PLR2004
        raise ValueError("Sensitivity observations must include a leading batch dimension.")
    reference = probability_function(observation)

    interventions: dict[str, tuple[dict[str, np.ndarray], np.ndarray | None]] = {}
    zero_global = {key: value.copy() for key, value in observation.items()}
    zero_global["global_state"] = np.zeros_like(global_state)
    interventions["zero_global_state"] = (zero_global, None)

    permuted_global = {key: value.copy() for key, value in observation.items()}
    permuted_global["global_state"] = global_state[..., ::-1].copy()
    interventions["permute_global_features"] = (permuted_global, None)

    mean_actions = {key: value.copy() for key, value in observation.items()}
    mean_row = np.mean(action_features, axis=-2, keepdims=True)
    mean_actions["action_features"] = np.repeat(mean_row, action_features.shape[-2], axis=-2)
    interventions["mean_action_features"] = (mean_actions, None)

    permutation = np.arange(action_features.shape[-2] - 1, -1, -1)
    permuted_rows = {key: value.copy() for key, value in observation.items()}
    permuted_rows["action_features"] = action_features[..., permutation, :].copy()
    interventions["permute_action_rows"] = (permuted_rows, np.argsort(permutation))

    if global_state.shape[0] > 1:
        other_task = {key: value.copy() for key, value in observation.items()}
        other_task["global_state"] = np.roll(global_state, shift=1, axis=0)
        other_task["action_features"] = np.roll(action_features, shift=1, axis=0)
        interventions["state_from_another_worker"] = (other_task, None)

    results: dict[str, dict[str, float]] = {}
    for name, (perturbed_observation, output_permutation) in interventions.items():
        perturbed = probability_function(perturbed_observation)
        if output_permutation is not None:
            perturbed = perturbed[..., output_permutation]
        results[name] = policy_sensitivity_metrics(reference, perturbed)
    return results


def budget_quartile(budget_fraction: float) -> int | None:
    """Map a finite budget fraction to a one-indexed quartile."""
    if not np.isfinite(budget_fraction):
        return None
    clipped_fraction = float(np.clip(budget_fraction, 0.0, 1.0))
    return min(int(clipped_fraction * 4), 3) + 1


class ActionLoggingCallback(BaseCallback):
    """Callback to log actions.

    For each new episode, log the actions. Will be overwritten.
    Intended for quick inspection.
    """

    def __init__(self, n_envs: int, csv_path: str | None = None, verbose: int = 0) -> None:
        """Init.

        Parameters
        ----------
        n_envs : int
            Number of environments.
        csv_path : str | None, optional
            The target path for the actions file, by default None. Defaults to
            the current run directory / "tensorboard/actions.csv".
        verbose : int, optional
            Verbosity level of the callback, by default 0
        """
        super().__init__(verbose)
        if csv_path is None:
            csv_path = str(get_run_directory() / "tensorboard/actions.csv")
        self.csv_path = csv_path
        self.file: TextIOWrapper | None = None
        self.writer = None
        self.step = 0
        self._n_envs = n_envs
        self._episode_ids = [0] * n_envs
        self._instances: list[object] = []
        self._last_deterministic_actions: list[int | None] = [None] * n_envs
        self._last_stochastic_actions: list[int | None] = [None] * n_envs
        self._episode_first_deterministic_action: list[int | None] = [None] * n_envs
        self._episode_deterministic_constant = [True] * n_envs
        self._episode_diagnostics_complete = [True] * n_envs
        self._raw_budget_fractions: np.ndarray | None = None

    def _open_csv(self) -> None:
        """Open a fresh append-only action log for this training run."""
        if self.file is not None:
            self.file.close()

        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.csv_path, "w", newline="")  # noqa: SIM115
        self.writer = csv.writer(self.file)  # type: ignore[assignment]
        assert self.writer is not None

        header = (
            ["step"]
            + [f"env_{i}/action" for i in range(self._n_envs)]
            + [f"env_{i}/instance" for i in range(self._n_envs)]
            + [f"env_{i}/bo_evaluations" for i in range(self._n_envs)]
            + [f"env_{i}/done" for i in range(self._n_envs)]
            + [f"env_{i}/episode" for i in range(self._n_envs)]
        )
        self.writer.writerow(header)

        self.step = 0
        self._episode_ids = [0] * self._n_envs
        self._reset_policy_diagnostics()

    def _reset_policy_diagnostics(self) -> None:
        """Reset per-environment state used by switch and episode metrics."""
        self._last_deterministic_actions = [None] * self._n_envs
        self._last_stochastic_actions = [None] * self._n_envs
        self._episode_first_deterministic_action = [None] * self._n_envs
        self._episode_deterministic_constant = [True] * self._n_envs
        self._episode_diagnostics_complete = [True] * self._n_envs
        self._raw_budget_fractions = None

    def _on_training_start(self) -> None:
        self._open_csv()
        self._instances = list(self.training_env.get_attr("instance"))
        self._refresh_raw_budget_fractions()

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        """Detach a tensor-like value and return a NumPy array."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _policy_statistics(self) -> CategoricalPolicyStatistics | None:
        """Return categorical policy summaries for the current rollout state."""
        observation = self.locals.get("obs_tensor")
        get_distribution = getattr(self.model.policy, "get_distribution", None)
        if observation is None or get_distribution is None:
            return None
        with th.no_grad():
            policy_distribution = get_distribution(observation)
        distribution = getattr(policy_distribution, "distribution", None)
        probabilities = getattr(distribution, "probs", None)
        logits = getattr(distribution, "logits", None)
        if probabilities is None:
            return None
        return categorical_policy_statistics(
            self._as_numpy(probabilities),
            None if logits is None else self._as_numpy(logits),
        )

    def _budget_fractions_from_observation(self, observation: Any) -> np.ndarray | None:
        """Read the raw budget percentage from the common global state."""
        if not isinstance(observation, dict) or "global_state" not in observation:
            return None
        global_state = self._as_numpy(observation["global_state"])
        if global_state.ndim == 1:
            global_state = global_state[None, :]
        if global_state.ndim != MATRIX_NDIM or global_state.shape[0] != self._n_envs or global_state.shape[1] == 0:
            return None
        return np.asarray(global_state[:, 0], dtype=float)

    def _refresh_raw_budget_fractions(self) -> None:
        """Cache raw observations from VecNormalize for the following action."""
        get_original_obs = getattr(self.training_env, "get_original_obs", None)
        if get_original_obs is None:
            self._raw_budget_fractions = None
            return
        self._raw_budget_fractions = self._budget_fractions_from_observation(get_original_obs())

    @staticmethod
    def _budget_fraction_from_info(info: dict[str, Any]) -> float | None:
        """Read an explicit fraction or evaluation-count denominator from info."""
        for key in ("budget_fraction", "budget_percentage"):
            if key in info:
                value = float(info[key])
                return value if np.isfinite(value) else None

        if "bo_evaluations" not in info:
            return None
        evaluations = float(info["bo_evaluations"])
        for key in ("bo_budget", "total_bo_evaluations", "n_trials"):
            if key in info:
                budget = float(info[key])
                if np.isfinite(evaluations) and np.isfinite(budget) and budget > 0.0:
                    return evaluations / budget
        return None

    def _budget_fractions(self, infos: list[dict[str, Any]]) -> np.ndarray:
        """Resolve pre-action budget fractions with explicit info as fallback."""
        fractions = self._raw_budget_fractions
        if fractions is None:
            fractions = self._budget_fractions_from_observation(self.locals.get("obs_tensor"))
        fractions = np.full(self._n_envs, np.nan) if fractions is None else np.asarray(fractions, dtype=float).copy()
        for env_id, info in enumerate(infos):
            if not np.isfinite(fractions[env_id]):
                info_fraction = self._budget_fraction_from_info(info)
                if info_fraction is not None:
                    fractions[env_id] = info_fraction
        return fractions

    def _record_policy_statistics(self, statistics: CategoricalPolicyStatistics) -> None:
        """Record scalar policy-distribution metrics through the SB3 logger."""
        for action_id, probability in enumerate(statistics.mean_probabilities):
            self.logger.record_mean(f"policy/prob_action_{action_id}", float(probability))
        self.logger.record_mean("policy/normalized_entropy", statistics.normalized_entropy)
        self.logger.record_mean("policy/max_probability", statistics.max_probability)
        self.logger.record_mean("policy/top1_top2_logit_gap", statistics.top1_top2_logit_gap)

    @staticmethod
    def _discrete_actions(actions: Any, n_envs: int, n_actions: int) -> np.ndarray | None:
        """Return scalar categorical action indices, or None for other spaces."""
        values = np.asarray(actions)
        if values.size != n_envs:
            return None
        values = values.reshape(n_envs)
        if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
            return None
        values = values.astype(int)
        if np.any((values < 0) | (values >= n_actions)):
            return None
        return values

    def _record_action_sequences(
        self,
        deterministic_actions: np.ndarray,
        stochastic_actions: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Record switch indicators and completed deterministic episodes."""
        for env_id, (deterministic, stochastic, done) in enumerate(
            zip(deterministic_actions, stochastic_actions, dones, strict=True)
        ):
            previous_deterministic = self._last_deterministic_actions[env_id]
            previous_stochastic = self._last_stochastic_actions[env_id]
            if previous_deterministic is not None:
                self.logger.record_mean(
                    "policy/deterministic_switch_rate",
                    float(int(deterministic) != previous_deterministic),
                )
            if previous_stochastic is not None:
                self.logger.record_mean(
                    "policy/stochastic_switch_rate",
                    float(int(stochastic) != previous_stochastic),
                )

            first_deterministic = self._episode_first_deterministic_action[env_id]
            if first_deterministic is None:
                self._episode_first_deterministic_action[env_id] = int(deterministic)
            elif int(deterministic) != first_deterministic:
                self._episode_deterministic_constant[env_id] = False

            self._last_deterministic_actions[env_id] = int(deterministic)
            self._last_stochastic_actions[env_id] = int(stochastic)
            if done:
                if self._episode_diagnostics_complete[env_id]:
                    self.logger.record_mean(
                        "policy/constant_episode_fraction",
                        float(self._episode_deterministic_constant[env_id]),
                    )
                self._reset_finished_policy_episode(env_id)

    def _reset_finished_policy_episode(self, env_id: int) -> None:
        """Start fresh diagnostic state after one vector worker terminates."""
        self._last_deterministic_actions[env_id] = None
        self._last_stochastic_actions[env_id] = None
        self._episode_first_deterministic_action[env_id] = None
        self._episode_deterministic_constant[env_id] = True
        self._episode_diagnostics_complete[env_id] = True

    def _record_budget_histograms(
        self,
        actions: np.ndarray,
        budget_fractions: np.ndarray,
        n_actions: int,
    ) -> None:
        """Record normalized sampled-action histograms for known quartiles."""
        for action, fraction in zip(actions, budget_fractions, strict=True):
            quartile = budget_quartile(float(fraction))
            if quartile is None:
                continue
            for action_id in range(n_actions):
                self.logger.record_mean(
                    f"policy/action_histogram_by_budget_quartile/q{quartile}_action_{action_id}",
                    float(action == action_id),
                    exclude=("stdout", "log"),
                )

    def _record_domain_histograms(
        self,
        actions: np.ndarray,
        infos: list[dict[str, Any]],
        n_actions: int,
    ) -> None:
        """Record sampled-action frequencies separately for BBOB and YAHPO."""
        for action, info in zip(actions, infos, strict=True):
            domain = str(info.get("domain", "")).lower()
            if domain not in {"bbob", "yahpo"}:
                task_id = str(info.get("task_id", ""))
                domain = task_id.split("/", maxsplit=1)[0].lower()
            if domain not in {"bbob", "yahpo"}:
                continue
            for action_id in range(n_actions):
                self.logger.record_mean(
                    f"policy/action_histogram_by_domain/{domain}_action_{action_id}",
                    float(action == action_id),
                    exclude=("stdout", "log"),
                )

    def _record_action_feature_diagnostics(self, infos: list[dict[str, Any]]) -> None:
        """Forward environment-side proxy-candidate diagnostics to TensorBoard."""
        vector_keys = {
            "action_features/uncertainty_by_action",
            "action_features/novelty_by_action",
            "action_features/z_by_action",
        }
        scalar_keys = {
            "action_features/unique_candidate_count",
            "action_features/duplicate_candidate_fraction",
            "action_features/mean_pairwise_candidate_distance",
            "action_features/spearman_action_vs_uncertainty",
            "action_features/zero_consequence_row_fraction",
        }
        for info in infos:
            for key in scalar_keys:
                if key in info:
                    self.logger.record_mean(key, float(info[key]), exclude=("stdout", "log"))
            for key in vector_keys:
                if key not in info:
                    continue
                for action_id, value in enumerate(info[key]):
                    self.logger.record_mean(f"{key}/action_{action_id}", float(value), exclude=("stdout", "log"))

    def _record_policy_diagnostics(
        self,
        actions: Any,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Collect distribution, switching, episode, and quartile metrics."""
        statistics = self._policy_statistics()
        if statistics is None or statistics.deterministic_actions.size != self._n_envs:
            for env_id, done in enumerate(dones):
                self._episode_diagnostics_complete[env_id] = False
                self._last_deterministic_actions[env_id] = None
                self._last_stochastic_actions[env_id] = None
                if done:
                    self._reset_finished_policy_episode(env_id)
            return

        self._record_policy_statistics(statistics)
        stochastic_actions = self._discrete_actions(
            actions,
            self._n_envs,
            statistics.mean_probabilities.size,
        )
        if stochastic_actions is None:
            return
        self._record_action_sequences(statistics.deterministic_actions, stochastic_actions, dones)
        self._record_budget_histograms(
            stochastic_actions,
            self._budget_fractions(infos),
            statistics.mean_probabilities.size,
        )
        self._record_domain_histograms(stochastic_actions, infos, statistics.mean_probabilities.size)
        self._record_action_feature_diagnostics(infos)

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        dones = np.asarray(self.locals["dones"], dtype=bool)
        infos = self.locals["infos"]

        self._record_policy_diagnostics(actions, dones, infos)

        # Preserve every action component
        row = [self.step]
        for action in actions:
            action_array = np.asarray(action)
            value = action_array.item() if action_array.size == 1 else json.dumps(action_array.tolist())
            row.append(value)

        # VecEnv auto-resets completed environments before this callback. Keep
        # the cached pre-step instance for the terminal action, then refresh it.
        row.extend(self._instances)
        row.extend(info.get("bo_evaluations", "") for info in infos)
        row.extend(bool(done) for done in dones)
        row.extend(self._episode_ids)

        self.writer.writerow(row)  # type: ignore[assignment,attr-defined]
        self.file.flush()  # type: ignore[assignment,union-attr]

        self.step += 1
        if np.any(dones):
            next_instances = self.locals["env"].get_attr("instance")
            for env_id, done in enumerate(dones):
                if done:
                    self._episode_ids[env_id] += 1
                    self._instances[env_id] = next_instances[env_id]

        self._refresh_raw_budget_fractions()

        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
