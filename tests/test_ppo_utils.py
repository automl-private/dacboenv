"""Focused tests for inexpensive PPO policy diagnostics."""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
from dacboenv.experiment.ppo_utils import (
    ActionLoggingCallback,
    budget_quartile,
    categorical_policy_statistics,
    policy_sensitivity_metrics,
    structured_policy_sensitivity,
)


class _MeanLogger:
    def __init__(self) -> None:
        self.values: dict[str, list[float]] = defaultdict(list)

    def record_mean(self, key: str, value: float, exclude=None) -> None:  # noqa: ARG002
        self.values[key].append(float(value))

    def mean(self, key: str) -> float:
        return float(np.mean(self.values[key]))


class _CategoricalPolicy:
    def __init__(self, logits: list[np.ndarray]) -> None:
        self._logits = iter(logits)

    def get_distribution(self, _observation):
        distribution = th.distributions.Categorical(logits=th.as_tensor(next(self._logits)))
        return SimpleNamespace(distribution=distribution)


class _VectorEnv:
    def __init__(self) -> None:
        self.instances = [(0, "task-a"), (1, "task-b")]

    def get_attr(self, name: str):
        assert name == "instance"
        return self.instances


def test_categorical_policy_statistics_are_normalized_and_use_logit_gap() -> None:
    probabilities = np.asarray(
        [
            [0.5, 0.2, 0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )
    logits = np.log(probabilities)

    statistics = categorical_policy_statistics(probabilities, logits)

    expected_entropy = -np.sum(probabilities * np.log(probabilities), axis=1) / np.log(5)
    np.testing.assert_allclose(statistics.mean_probabilities, np.mean(probabilities, axis=0))
    assert statistics.normalized_entropy == pytest.approx(np.mean(expected_entropy))
    assert statistics.max_probability == pytest.approx(0.35)
    assert statistics.top1_top2_logit_gap == pytest.approx(np.log(2.5) / 2)
    np.testing.assert_array_equal(statistics.deterministic_actions, [0, 0])


def test_policy_sensitivity_metrics_compute_directional_kl_and_tv() -> None:
    reference = np.asarray([[0.8, 0.2], [0.5, 0.5]])
    perturbed = np.asarray([[0.6, 0.4], [0.5, 0.5]])

    metrics = policy_sensitivity_metrics(reference, perturbed)

    first_kl = 0.8 * np.log(0.8 / 0.6) + 0.2 * np.log(0.2 / 0.4)
    assert metrics["mean_kl"] == pytest.approx(first_kl / 2)
    assert metrics["mean_total_variation"] == pytest.approx(0.1)
    assert policy_sensitivity_metrics(reference, reference) == {
        "mean_kl": pytest.approx(0.0),
        "mean_total_variation": pytest.approx(0.0),
    }


def test_structured_policy_sensitivity_applies_all_state_interventions() -> None:
    observation = {
        "global_state": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        "action_features": np.asarray(
            [
                [[0.0], [1.0], [2.0]],
                [[2.0], [1.0], [0.0]],
            ]
        ),
    }

    def probabilities(state: dict[str, np.ndarray]) -> np.ndarray:
        logits = state["action_features"][..., 0] + state["global_state"][:, :1]
        values = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return values / np.sum(values, axis=1, keepdims=True)

    metrics = structured_policy_sensitivity(observation, probabilities)

    assert set(metrics) == {
        "zero_global_state",
        "permute_global_features",
        "mean_action_features",
        "permute_action_rows",
        "state_from_another_worker",
    }
    assert metrics["mean_action_features"]["mean_total_variation"] > 0.0
    assert metrics["permute_action_rows"]["mean_total_variation"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [
        (0.0, 1),
        (0.249, 1),
        (0.25, 2),
        (0.5, 3),
        (0.75, 4),
        (1.0, 4),
        (np.nan, None),
    ],
)
def test_budget_quartile_boundaries(fraction: float, expected: int | None) -> None:
    assert budget_quartile(fraction) == expected


def test_action_callback_records_policy_switch_episode_and_quartile_metrics(tmp_path) -> None:
    logits = [
        np.asarray([[4.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0]]),
        np.asarray([[0.0, 4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0, 0.0]]),
    ]
    logger = _MeanLogger()
    vector_env = _VectorEnv()
    model = SimpleNamespace(
        policy=_CategoricalPolicy(logits),
        logger=logger,
        get_env=lambda: vector_env,
    )
    callback = ActionLoggingCallback(n_envs=2, csv_path=str(tmp_path / "actions.csv"))
    callback.model = model
    callback._on_training_start()

    callback.locals = {
        "actions": np.asarray([0, 1]),
        "dones": np.asarray([False, False]),
        "infos": [
            {
                "bo_evaluations": 1,
                "domain": "bbob",
                "action_features/unique_candidate_count": 4,
                "action_features/uncertainty_by_action": [0.1, 0.2, 0.3, 0.4, 0.5],
            },
            {"bo_evaluations": 3, "task_id": "yahpo/so/lcbench/1/None"},
        ],
        "obs_tensor": {"global_state": th.as_tensor([[0.1, 0.0], [0.3, 0.0]])},
        "env": vector_env,
    }
    assert callback._on_step()

    vector_env.instances = [(2, "task-c"), (3, "task-d")]
    callback.locals = {
        "actions": np.asarray([1, 0]),
        "dones": np.asarray([True, True]),
        "infos": [{"bo_evaluations": 6}, {"bo_evaluations": 9}],
        "obs_tensor": {"global_state": th.as_tensor([[0.6, 0.0], [0.9, 0.0]])},
        "env": vector_env,
    }
    assert callback._on_step()
    callback._on_training_end()

    for action_id in range(5):
        assert f"policy/prob_action_{action_id}" in logger.values
    assert 0.0 <= logger.mean("policy/normalized_entropy") <= 1.0
    assert logger.mean("policy/max_probability") > 0.9
    assert logger.mean("policy/top1_top2_logit_gap") == pytest.approx(4.0)
    assert logger.mean("policy/deterministic_switch_rate") == pytest.approx(0.5)
    assert logger.mean("policy/stochastic_switch_rate") == pytest.approx(1.0)
    assert logger.mean("policy/constant_episode_fraction") == pytest.approx(0.5)
    assert logger.mean("policy/action_histogram_by_budget_quartile/q1_action_0") == pytest.approx(1.0)
    assert logger.mean("policy/action_histogram_by_budget_quartile/q2_action_1") == pytest.approx(1.0)
    assert logger.mean("policy/action_histogram_by_budget_quartile/q3_action_1") == pytest.approx(1.0)
    assert logger.mean("policy/action_histogram_by_budget_quartile/q4_action_0") == pytest.approx(1.0)
    assert logger.mean("policy/action_histogram_by_domain/bbob_action_0") == pytest.approx(1.0)
    assert logger.mean("policy/action_histogram_by_domain/yahpo_action_1") == pytest.approx(1.0)
    assert logger.mean("action_features/unique_candidate_count") == pytest.approx(4.0)
    assert logger.mean("action_features/uncertainty_by_action/action_4") == pytest.approx(0.5)
