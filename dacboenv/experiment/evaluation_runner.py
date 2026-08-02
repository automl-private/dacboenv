"""Concrete episode adapters for the manifest-driven paired evaluator.

The paired-evaluator module owns scientific context validation and statistics.
This module supplies the execution adapter shared by DACBO policies, including
SB3, static, random, marginal, and SAWEI policies.  No method receives a
different task, seed, budget, or reference convention through this adapter.
"""

from __future__ import annotations

import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from dacboenv.env.reward import TRUE_REGRET_EPSILON, normalized_reference_regret_potential
from dacboenv.experiment.default_smac import run_default_smac_episode
from dacboenv.experiment.paired_evaluator import (
    EvaluationContext,
    EvaluationMethod,
    EvaluationRecord,
    MethodRunner,
)
from dacboenv.experiment.source_provenance import current_source_revision

if TYPE_CHECKING:
    from dacboenv.reference import ObjectiveReference


class EvaluationPolicy(Protocol):
    """Minimal policy interface accepted by the unified DACBO runner."""

    def __call__(self, observation: Any) -> Any:
        """Return the next environment action."""


PolicyFactory = Callable[[Any, EvaluationContext, EvaluationMethod], EvaluationPolicy]
EnvironmentFactory = Callable[..., Any]


@dataclass(frozen=True)
class EpisodeTrace:
    """Auditable trajectory accompanying one tidy evaluation record."""

    record: EvaluationRecord
    actions: tuple[Any, ...]
    incumbent_trajectory: tuple[float, ...]
    policy_metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "record": self.record.to_tidy_row(),
            "actions": [_json_value(action) for action in self.actions],
            "incumbent_trajectory": list(self.incumbent_trajectory),
            "policy_metadata": _json_value(self.policy_metadata),
        }


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _scalar_cost(value: Any) -> float:
    values = np.asarray(value, dtype=float).reshape(-1)
    return float(values[0]) if values.size else np.inf


def _create_context_environment(env_factory: EnvironmentFactory, context: EvaluationContext) -> Any:
    """Create an environment while retaining compatibility with legacy f1 factories."""
    parameters = inspect.signature(env_factory).parameters.values()
    supports_frequency = "interaction_frequency" in inspect.signature(env_factory).parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
    )
    if supports_frequency:
        return env_factory(
            context.task_id,
            context.inner_seed,
            interaction_frequency=context.interaction_frequency,
        )
    if context.interaction_frequency != 1:
        raise TypeError("Evaluation environment factories must accept interaction_frequency for f5/f10 contexts.")
    return env_factory(context.task_id, context.inner_seed)


def _incumbent_trajectory(env: Any) -> tuple[float, ...]:
    """Return the finite incumbent after every completed BO evaluation."""
    best = np.inf
    trajectory: list[float] = []
    for trial_value in env._smac_instance.runhistory._data.values():
        cost = _scalar_cost(trial_value.cost)
        if np.isfinite(cost):
            best = min(best, cost)
        trajectory.append(float(best))
    if not trajectory or not np.isfinite(trajectory[-1]):
        raise RuntimeError("Evaluation episode produced no finite incumbent.")
    return tuple(trajectory)


def _normalized_regret_trajectory(
    trajectory: Sequence[float],
    *,
    reference_value: float,
    initial_design_size: int,
) -> np.ndarray:
    if initial_design_size <= 0 or initial_design_size > len(trajectory):
        raise RuntimeError(
            f"Invalid initial-design size {initial_design_size} for {len(trajectory)} completed evaluations."
        )
    initial_incumbent = float(trajectory[initial_design_size - 1])
    initial_regret = max(initial_incumbent - float(reference_value), 0.0)
    scale = max(initial_regret, TRUE_REGRET_EPSILON)
    values = np.asarray(trajectory, dtype=float)
    return np.maximum(values - float(reference_value), 0.0) / scale


def _validate_environment_reference(env: Any, context: EvaluationContext) -> None:
    """Prove that the reward-side reference matches the paired context."""
    reference = getattr(env, "_objective_reference", None)
    if reference is None:
        raise RuntimeError(
            "A unified evaluation environment must expose its privileged reward-side ObjectiveReference after reset."
        )
    if str(reference.task_id) != context.task_id:
        raise RuntimeError(
            f"Environment reference task {reference.task_id!r} differs from paired task {context.task_id!r}."
        )
    if str(reference.kind) != context.reference_kind:
        raise RuntimeError(
            f"Environment reference kind {reference.kind!r} differs from paired kind {context.reference_kind!r}."
        )
    if str(reference.runtime_objective_transform) != context.objective_transform:
        raise RuntimeError(
            "Environment runtime objective transform differs from the paired convention: "
            f"{reference.runtime_objective_transform!r} != {context.objective_transform!r}."
        )
    tolerance = max(float(reference.tolerance), 1e-12)
    if not np.isclose(float(reference.value), context.reference_value, rtol=0.0, atol=tolerance):
        raise RuntimeError(
            "Environment reference value differs from the paired convention: "
            f"{reference.value!r} != {context.reference_value!r} (atol={tolerance})."
        )


def _validate_policy_observation(env: Any, observation: Any, *, phase: str) -> None:
    """Reject observations that do not satisfy the environment's public space."""
    observation_space = getattr(env, "observation_space", None)
    contains = getattr(observation_space, "contains", None)
    if callable(contains) and not contains(observation):
        raise RuntimeError(f"The {phase} observation is not contained in the declared observation space.")


def _action_diagnostics(actions: Sequence[Any], action_space: Any) -> tuple[tuple[int, ...], float, bool]:
    """Summarize discrete traces; continuous methods retain their raw trace."""
    n_actions = getattr(action_space, "n", None)
    if isinstance(n_actions, (int, np.integer)) and int(n_actions) > 0:
        integer_actions = [int(np.asarray(action).item()) for action in actions]
        if any(action < 0 or action >= int(n_actions) for action in integer_actions):
            raise RuntimeError("A policy emitted an action outside the discrete action space.")
        histogram = tuple(int(value) for value in np.bincount(integer_actions, minlength=int(n_actions)))
        switch_rate = (
            float(np.mean(np.asarray(integer_actions[1:]) != np.asarray(integer_actions[:-1])))
            if len(integer_actions) > 1
            else 0.0
        )
        return histogram, switch_rate, len(set(integer_actions)) <= 1
    # SAWEI's native action is continuous. Its exact action trace is retained
    # alongside the tidy row rather than silently quantized into fake bins.
    numeric = np.asarray([float(np.asarray(action).item()) for action in actions], dtype=float)
    switch_rate = float(np.mean(~np.isclose(numeric[1:], numeric[:-1]))) if len(numeric) > 1 else 0.0
    return (), switch_rate, bool(len(numeric) <= 1 or np.allclose(numeric, numeric[0]))


def current_commit(repository: Path | None = None) -> str:
    """Return a clean commit or a commit plus dirty-source digest."""
    return current_source_revision(repository)


def run_dacbo_episode(  # noqa: C901, PLR0915
    context: EvaluationContext,
    method: EvaluationMethod,
    *,
    env_factory: EnvironmentFactory,
    policy_factory: PolicyFactory,
    action_family: str,
    checkpoint_type: str,
    outer_ppo_seed: int | None,
    code_commit: str,
    policy_seed: int | None = None,
    policy_metadata: Mapping[str, Any] | None = None,
) -> EpisodeTrace:
    """Run one full, paired DACBO episode and emit the unified schema."""
    env = _create_context_environment(env_factory, context)
    actions: list[Any] = []
    rewards: list[float] = []
    started = time.perf_counter()
    try:
        observation, info = env.reset()
        _validate_policy_observation(env, observation, phase="reset")
        task_id = str(info.get("task_id", getattr(env, "current_task_id", "")))
        inner_seed = int(info.get("inner_seed", getattr(env, "current_seed", -1)))
        if (task_id, inner_seed) != (context.task_id, context.inner_seed):
            raise RuntimeError(
                "Environment factory changed the paired context: "
                f"expected {(context.task_id, context.inner_seed)!r}, got {(task_id, inner_seed)!r}."
            )
        if int(env._n_trials) != context.evaluation_budget:
            raise RuntimeError(
                f"Environment budget {env._n_trials} differs from paired budget {context.evaluation_budget}."
            )
        if int(env.interaction_frequency) != context.interaction_frequency:
            raise RuntimeError(
                "Environment interaction frequency differs from the paired context: "
                f"{env.interaction_frequency} != {context.interaction_frequency}."
            )
        _validate_environment_reference(env, context)
        policy = policy_factory(env, context, method)
        set_seed = getattr(policy, "set_seed", None)
        if callable(set_seed):
            set_seed(policy_seed)

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(observation)
            actions.append(_json_value(action))
            observation, reward, terminated, truncated, _step_info = env.step(action)
            _validate_policy_observation(env, observation, phase="step")
            if not np.isfinite(float(reward)):
                raise RuntimeError(f"Method {method.name!r} emitted non-finite reward {reward!r}.")
            rewards.append(float(reward))

        trajectory = _incumbent_trajectory(env)
        if len(trajectory) != context.evaluation_budget:
            raise RuntimeError(f"Completed {len(trajectory)} BO evaluations, expected {context.evaluation_budget}.")
        initial_design_size = len(env._smac_instance.intensifier.config_selector._initial_design_configs)
        normalized = _normalized_regret_trajectory(
            trajectory,
            reference_value=context.reference_value,
            initial_design_size=initial_design_size,
        )
        final_incumbent = float(trajectory[-1])
        final_regret = max(final_incumbent - context.reference_value, 0.0)
        initial_incumbent = float(trajectory[initial_design_size - 1])
        aligned_return = normalized_reference_regret_potential(
            final_incumbent,
            context.reference_value,
            initial_incumbent,
        ) - normalized_reference_regret_potential(
            initial_incumbent,
            context.reference_value,
            initial_incumbent,
        )
        observed_return = float(np.sum(rewards))
        if not np.isclose(observed_return, aligned_return, rtol=0.0, atol=1e-10):
            raise RuntimeError(
                f"Method {method.name!r} episode return {observed_return} is not the paired "
                f"reference-regret telescoping return {aligned_return}."
            )
        histogram, switch_rate, constant = _action_diagnostics(actions, env.action_space)
        record = EvaluationRecord(
            domain=context.domain,
            scenario_or_function=context.scenario_or_function,
            dimension=context.dimension,
            task_id=context.task_id,
            native_instance=context.native_instance,
            inner_seed=context.inner_seed,
            outer_ppo_seed=outer_ppo_seed,
            method=method.name,
            action_family=action_family,
            checkpoint_type=checkpoint_type,
            evaluation_budget=context.evaluation_budget,
            reference_kind=context.reference_kind,
            reference_value=context.reference_value,
            objective_transform=context.objective_transform,
            final_incumbent=final_incumbent,
            final_reference_regret=final_regret,
            normalized_final_regret=float(normalized[-1]),
            anytime_auc=float(np.mean(normalized)),
            episode_return=float(aligned_return),
            action_histogram=histogram,
            deterministic_switch_rate=switch_rate,
            constant_policy=constant,
            runtime_seconds=float(time.perf_counter() - started),
            manifest_hash=context.manifest_hash,
            code_commit=code_commit,
            interaction_frequency=context.interaction_frequency,
        )
        metadata = dict(policy_metadata or {})
        history = getattr(policy, "_history", None)
        if history is not None:
            metadata.setdefault("native_policy_history", history)
        return EpisodeTrace(
            record=record,
            actions=tuple(actions),
            incumbent_trajectory=trajectory,
            policy_metadata=metadata,
        )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def make_dacbo_method_runner(
    *,
    env_factory: EnvironmentFactory,
    policy_factory: PolicyFactory,
    action_family: str,
    checkpoint_type: str,
    outer_ppo_seed: int | None,
    trace_directory: Path,
    code_commit: str | None = None,
    policy_seed: int | None = None,
    policy_metadata: Mapping[str, Any] | None = None,
) -> MethodRunner:
    """Create a paired-evaluator callback and persist every raw trajectory."""
    commit = code_commit or current_commit()

    def runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        trace = run_dacbo_episode(
            context,
            method,
            env_factory=env_factory,
            policy_factory=policy_factory,
            action_family=action_family,
            checkpoint_type=checkpoint_type,
            outer_ppo_seed=outer_ppo_seed,
            code_commit=commit,
            policy_seed=policy_seed,
            policy_metadata=policy_metadata,
        )
        trace_directory.mkdir(parents=True, exist_ok=True)
        safe_task = context.task_id.replace("/", "__")
        path = trace_directory / f"{method.name}__{safe_task}__seed{context.inner_seed}.json"
        path.write_text(json.dumps(trace.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return trace.record

    return runner


def make_default_smac_method_runner(
    *,
    output_directory: Path,
    trace_directory: Path,
    code_commit: str | None = None,
    objective_references: Mapping[str, ObjectiveReference] | None = None,
    context_split: str = "validation",
) -> MethodRunner:
    """Adapt native default SMAC outcomes to the common tidy evaluator schema.

    Native SMAC has no outer DACBO action, so its action histogram is empty,
    switch rate is zero, and ``constant_policy`` records the absence of a
    changing external controller. Pairing is still enforced through the exact
    task, inner seed, native budget, reference, and manifest context.
    """
    commit = code_commit or current_commit()

    def runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        reference = None if objective_references is None else objective_references.get(context.task_id)
        if context.domain == "bbob":
            if context.reference_kind != "exact" or context.objective_transform != "identity":
                raise ValueError("Native default-SMAC BBOB evaluation requires an exact identity reference.")
        elif context.domain == "yahpo":
            if reference is None:
                raise ValueError("Native default-SMAC YAHPO evaluation requires a preflighted reference table.")
            if (
                reference.kind != context.reference_kind
                or reference.runtime_objective_transform != context.objective_transform
            ):
                raise ValueError("Native default-SMAC YAHPO reference convention differs from the paired context.")
        else:
            raise ValueError(f"Unsupported native default-SMAC context domain {context.domain!r}.")
        context_directory = output_directory / context.task_id.replace("/", "__") / f"seed_{context.inner_seed}"
        result = run_default_smac_episode(
            context.task_id,
            context.inner_seed,
            output_directory=context_directory,
            objective_reference=reference,
            context_split=context_split,
        )
        if result.bo_evaluations != context.evaluation_budget:
            raise RuntimeError(
                f"Native default SMAC completed {result.bo_evaluations} evaluations, "
                f"expected paired budget {context.evaluation_budget}."
            )
        if not np.isclose(result.reference_value, context.reference_value, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "Native default-SMAC reference differs from the paired context: "
                f"{result.reference_value} != {context.reference_value}."
            )
        record = EvaluationRecord(
            domain=context.domain,
            scenario_or_function=context.scenario_or_function,
            dimension=context.dimension,
            task_id=context.task_id,
            native_instance=context.native_instance,
            inner_seed=context.inner_seed,
            outer_ppo_seed=None,
            method=method.name,
            action_family="native_smac",
            checkpoint_type="none",
            evaluation_budget=context.evaluation_budget,
            reference_kind=context.reference_kind,
            reference_value=context.reference_value,
            objective_transform=context.objective_transform,
            final_incumbent=result.final_incumbent,
            final_reference_regret=result.final_regret,
            normalized_final_regret=result.normalized_final_regret,
            anytime_auc=result.normalized_anytime_auc,
            episode_return=result.telescoping_return,
            action_histogram=(),
            deterministic_switch_rate=0.0,
            constant_policy=True,
            runtime_seconds=result.runtime_seconds,
            manifest_hash=context.manifest_hash,
            code_commit=commit,
            interaction_frequency=context.interaction_frequency,
        )
        trace_directory.mkdir(parents=True, exist_ok=True)
        trace_path = trace_directory / (
            f"{method.name}__{context.task_id.replace('/', '__')}__seed{context.inner_seed}.json"
        )
        trace_path.write_text(
            json.dumps(
                {
                    "record": record.to_tidy_row(),
                    "actions": [],
                    "incumbent_trajectory": list(result.incumbent_trajectory),
                    "policy_metadata": {
                        "native_default_smac": True,
                        "external_action_interface": False,
                    },
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return record

    return runner


__all__ = [
    "EnvironmentFactory",
    "EpisodeTrace",
    "EvaluationPolicy",
    "PolicyFactory",
    "current_commit",
    "make_dacbo_method_runner",
    "make_default_smac_method_runner",
    "run_dacbo_episode",
]
