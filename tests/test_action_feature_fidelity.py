"""Contracts for non-destructive action-feature fidelity inspection."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, Float
from dacboenv.env.observation import calculate_candidate_semantic_descriptors
from dacboenv.experiment.action_feature_fidelity import (
    ActionFeatureFidelityRecord,
    FidelityPanelEntry,
    inspect_action_feature_fidelity,
    summarize_action_feature_fidelity,
    write_fidelity_csv,
    write_fidelity_summary,
)
from dacboenv.experiment.build_fidelity_panel import build_panel
from gymnasium.spaces import Discrete


class FakeModel:
    """Deterministic surrogate with candidate-dependent mean and variance."""

    def predict_marginalized(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return deterministic posterior moments for candidate rows."""
        x = np.asarray(values, dtype=float)[:, 0]
        mean = np.square(x - 0.4)
        variance = np.square(0.2 + x)
        return mean[:, None], variance[:, None]


class FakeRunHistory:
    """Separate completed and running configurations like SMAC's runhistory."""

    def __init__(self, configs: list[Configuration]) -> None:
        self.completed = list(configs)
        self.running: list[Configuration] = []
        self.finished = len(configs)

    def get_configs(self) -> list[Configuration]:
        """Return completed and running configurations."""
        return [*self.completed, *self.running]


class FakeSelector:
    """Minimal synchronized ConfigSelector surface used by descriptors."""

    def __init__(self, configs: list[Configuration]) -> None:
        self._model = FakeModel()
        self._acquisition_function = SimpleNamespace(_eta=1.0, _xi=0.1, _alpha=0.5)
        self._configs = configs

    def _collect_data(self) -> tuple[np.ndarray, np.ndarray, list[Configuration]]:
        x = np.vstack([configuration.get_array() for configuration in self._configs])
        y = np.asarray([[0.4], [1.2], *([[0.8]] * max(len(self._configs) - 2, 0))])
        return x, y, list(self._configs)


class FakeSMBO:
    """Candidate generator whose ask registers, but never evaluates, a trial."""

    ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
    ACTUAL_X = (0.15, 0.35, 0.35, 0.75, 0.95)

    def __init__(self, configspace: ConfigurationSpace) -> None:
        completed = [
            Configuration(configspace, values={"x": 0.0}),
            Configuration(configspace, values={"x": 0.05}),
        ]
        self.runhistory = FakeRunHistory(completed)
        self._scenario = SimpleNamespace(configspace=configspace, n_trials=12, seed=7)
        selector = FakeSelector(self.runhistory.completed)
        self.intensifier = SimpleNamespace(config_selector=selector)
        self._intensifier = SimpleNamespace(_config_selector=selector)
        self.ask_calls = 0
        self.objective_calls = 0

    def ask(self) -> SimpleNamespace:
        """Register and return the action-dependent candidate without evaluation."""
        self.ask_calls += 1
        alpha = float(self.intensifier.config_selector._acquisition_function._alpha)
        action = self.ALPHAS.index(alpha)
        candidate = Configuration(self._scenario.configspace, values={"x": self.ACTUAL_X[action]})
        self.runhistory.running.append(candidate)
        return SimpleNamespace(config=candidate)


class FakeFidelityEnv:
    """Replayable controlled BO environment with auditable objective calls."""

    PROXY_X = (0.15, 0.25, 0.45, 0.65, 0.85)
    ALPHAS = FakeSMBO.ALPHAS

    def __init__(self, task_id: str, inner_seed: int, action_space_label: str) -> None:
        self.task_id = task_id
        self.inner_seed = inner_seed
        self.action_space_label = action_space_label
        self.action_space = Discrete(5)
        self.closed = False
        self.update_calls = 0
        self.incumbent = 10.0
        self._build_smbo()

    def _build_smbo(self) -> None:
        configspace = ConfigurationSpace(seed=self.inner_seed, space={"x": Float("x", (0.0, 1.0))})
        self._smac_instance = FakeSMBO(configspace)

    def reset(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reconstruct the fixed deterministic BO context."""
        self.incumbent = 10.0
        self.update_calls = 0
        self._build_smbo()
        return self.get_observation(), {"task_id": self.task_id, "inner_seed": self.inner_seed}

    def update_optimizer(self, action: int) -> None:
        """Apply the same absolute action mapping as discrete WEI control."""
        self.update_calls += 1
        self._smac_instance.intensifier.config_selector._acquisition_function._alpha = self.ALPHAS[action]

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Evaluate exactly once while replaying a recorded history action."""
        self.update_optimizer(action)
        smbo = self._smac_instance
        smbo.objective_calls += 1
        configuration = Configuration(
            smbo._scenario.configspace,
            values={"x": min(0.1 + 0.01 * action + 0.01 * smbo.runhistory.finished, 0.99)},
        )
        smbo.runhistory.completed.append(configuration)
        smbo.intensifier.config_selector._configs.append(configuration)
        smbo.runhistory.finished += 1
        self.incumbent -= 0.1 * (action + 1)
        return self.get_observation(), 0.0, False, False, {}

    def get_observation(self) -> dict[str, np.ndarray]:
        """Return fitted proxy candidates and their Stage-A descriptor rows."""
        smbo = self._smac_instance
        proxies = [Configuration(smbo._scenario.configspace, values={"x": value}) for value in self.PROXY_X]
        setattr(smbo, "_dacboenv_selected_action_feature_candidates", proxies)
        rows: list[list[float]] = []
        for alpha, candidate in zip(self.ALPHAS, proxies, strict=True):
            descriptors = calculate_candidate_semantic_descriptors(
                smbo,
                candidate,
                include_xi=True,
                evaluated_configs=smbo.runhistory.completed,
            )
            rows.append(
                [
                    alpha,
                    descriptors["standardized_improvement"],
                    descriptors["normalized_uncertainty"],
                    descriptors["novelty"],
                ]
            )
        return {
            "global_state": np.zeros(13, dtype=np.float32),
            "action_features": np.asarray(rows, dtype=np.float32),
        }

    def get_n_finished_trials(self) -> int:
        """Return completed objective calls, excluding running asks."""
        return self._smac_instance.runhistory.finished

    def get_incumbent_cost(self) -> float:
        """Return the deterministic incumbent."""
        return self.incumbent

    def close(self) -> None:
        """Record clone cleanup."""
        self.closed = True


class RecordingFactory:
    """Build and retain every independent replay clone."""

    def __init__(self) -> None:
        self.created: list[FakeFidelityEnv] = []

    def __call__(self, task_id: str, inner_seed: int, action_space: str) -> FakeFidelityEnv:
        """Create one fresh fixed-context clone."""
        env = FakeFidelityEnv(task_id, inner_seed, action_space)
        self.created.append(env)
        return env


@pytest.fixture
def fidelity_records() -> tuple[ActionFeatureFidelityRecord, ...]:
    factory = RecordingFactory()
    return inspect_action_feature_fidelity(
        FidelityPanelEntry(
            task_id="bbob/2/3/0",
            inner_seed=17,
            action_space="WEI",
            history_policy="static-2",
            action_history=(2,),
        ),
        factory,
        forbidden_task_ids={"bbob/2/1/2"},
    )


def test_actual_candidate_inspection_uses_disposable_replay_clones() -> None:
    source = FakeFidelityEnv("bbob/2/3/0", 17, "WEI")
    source.reset()
    source_state = (
        source.get_n_finished_trials(),
        source.get_incumbent_cost(),
        source.update_calls,
        source._smac_instance.ask_calls,
        source._smac_instance.objective_calls,
    )
    factory = RecordingFactory()

    records = inspect_action_feature_fidelity(
        FidelityPanelEntry("bbob/2/3/0", 17, "WEI", "static-2", (2,)),
        factory,
        forbidden_task_ids={"bbob/2/1/2"},
    )

    assert len(records) == 5
    assert len(factory.created) == 6  # one proxy clone and one actual-candidate clone per action
    assert all(env.closed for env in factory.created)
    assert source_state == (
        source.get_n_finished_trials(),
        source.get_incumbent_cost(),
        source.update_calls,
        source._smac_instance.ask_calls,
        source._smac_instance.objective_calls,
    )

    proxy_clone, *actual_clones = factory.created
    assert proxy_clone._smac_instance.ask_calls == 0
    assert proxy_clone._smac_instance.objective_calls == 1  # replayed history only
    assert all(env._smac_instance.ask_calls == 1 for env in actual_clones)
    assert all(env._smac_instance.objective_calls == 1 for env in actual_clones)
    assert {record.bo_evaluations for record in records} == {3}
    assert {record.budget_phase for record in records} == {"25%"}
    assert len({record.observation_hash for record in records}) == 1

    assert records[0].exact_candidate_equality
    assert records[0].mixed_space_distance == pytest.approx(0.0)
    assert not records[1].exact_candidate_equality
    assert records[1].mixed_space_distance > 0.0
    assert records[1].actual_candidate_duplicate_count == 1
    assert records[2].actual_candidate_duplicate_count == 1
    assert all(record.control_identity_equality for record in records)
    # The fake SMAC runhistory registers the returned candidate as running.
    # Passing the pre-ask completed set prevents self-novelty from becoming zero.
    assert all(record.actual_novelty > 0.0 for record in records)


def test_forbidden_task_is_rejected_before_factory_use() -> None:
    factory = RecordingFactory()
    entry = FidelityPanelEntry("bbob/8/1/2", 7, "LCB", "uniform-random")

    with pytest.raises(ValueError, match="forbidden/test"):
        inspect_action_feature_fidelity(
            entry,
            factory,
            forbidden_task_ids=set(),
        )

    assert factory.created == []


def test_summary_classifies_rank_fidelity_and_writes_machine_readable_outputs(
    fidelity_records: tuple[ActionFeatureFidelityRecord, ...],
    tmp_path: Path,
) -> None:
    strong_records = tuple(
        replace(
            record,
            actual_standardized_improvement=record.proxy_standardized_improvement,
            actual_normalized_uncertainty=record.proxy_normalized_uncertainty,
            actual_novelty=record.proxy_novelty,
        )
        for record in fidelity_records
    )
    strong = summarize_action_feature_fidelity(strong_records)
    assert strong["fidelity"]["classification"] == "strong"
    assert strong["fidelity"]["median_rank_correlation"] == pytest.approx(1.0)
    assert strong["groupings"]["by_action_space_and_action"]
    assert strong["snapshot_action_order"][0]["standardized_improvement_top_action_agreement"]

    reversed_descriptors = {
        descriptor: [getattr(record, f"proxy_{descriptor}") for record in reversed(strong_records)]
        for descriptor in (
            "standardized_improvement",
            "normalized_uncertainty",
            "novelty",
        )
    }
    weak_records = tuple(
        replace(
            record,
            actual_standardized_improvement=reversed_descriptors["standardized_improvement"][index],
            actual_normalized_uncertainty=reversed_descriptors["normalized_uncertainty"][index],
            actual_novelty=reversed_descriptors["novelty"][index],
        )
        for index, record in enumerate(strong_records)
    )
    weak = summarize_action_feature_fidelity(weak_records)
    assert weak["fidelity"]["classification"] == "weak"

    csv_path = tmp_path / "fidelity.csv"
    summary_path = tmp_path / "fidelity.json"
    write_fidelity_csv(strong_records, csv_path)
    write_fidelity_summary(strong, summary_path)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert json.loads(rows[0]["action_history"]) == [2]
    assert json.loads(summary_path.read_text(encoding="utf-8"))["fidelity"]["classification"] == "strong"


def test_larger_panel_has_requested_strata_and_only_meaningful_default_smac_histories() -> None:
    panel = build_panel()
    assert len(panel) == 162
    assert {row["action_space"] for row in panel} == {
        "wei",
        "lcb_quantile",
        "ucb_quantile",
        "af_selection",
    }
    assert {row["task_id"].split("/")[1] for row in panel} == {"2", "4", "8"}
    default_rows = [row for row in panel if row["history_policy"] == "default_smac_equivalent"]
    assert {row["action_space"] for row in default_rows} == {"wei", "af_selection"}
    assert all(set(row["action_history"]) <= {2} for row in default_rows)
    for action_space in {row["action_space"] for row in panel}:
        for dimension in {row["task_id"].split("/")[1] for row in panel}:
            stratum = [
                row
                for row in panel
                if row["action_space"] == action_space and row["task_id"].split("/")[1] == dimension
            ]
            static_actions = {
                int(row["history_policy"].removeprefix("static_"))
                for row in stratum
                if row["history_policy"].startswith("static_")
            }
            assert static_actions == {0, 1, 2, 3, 4}
            assert sum(row["history_policy"].startswith("uniform_random_seed_") for row in stratum) == 6
