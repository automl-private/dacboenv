"""YAHPO split, budget, task-resolution, and generation contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from dacboenv.env.observation import GLOBAL_STATE_INDEX
from dacboenv.experiment.generate_yahpo_references import source_provenance
from dacboenv.experiment.populate_yahpo_references import assumed_reference
from dacboenv.experiment.real_env import real_structured_yahpo_env
from dacboenv.experiment.yahpo_protocol import (
    YAHPO_TRAIN_COUNTS,
    YAHPO_VALIDATION_COUNTS,
    YAHPOCoverageError,
    apply_yahpo_budget_multiplier,
    deterministic_yahpo_split,
    official_yahpo_task_ids,
    yahpo_task_id,
)
from dacboenv.reference import ASSUMED_BOUND_SOURCE_METHOD, ManifestReferenceProvider
from dacboenv.utils.carps_optimizer import get_task_config


def _synthetic_inventory() -> dict[str, list[str]]:
    return {scenario: [str(index) for index in range(1, 40)] for scenario in YAHPO_TRAIN_COUNTS}


@pytest.mark.parametrize(("scenario", "expected"), [("lcbench", -100.0), ("nb301", -100.0), ("rbv2_super", -1.0)])
def test_assumed_reference_uses_installed_runtime_accuracy_scale(scenario: str, expected: float) -> None:
    row = assumed_reference(
        scenario,
        "instance",
        generation_date="2026-08-09",
        data_identity={"version": "1.0.2", "git_commit": "a" * 40, "config_space_tree_sha256": "b" * 64},
        source_hash="c" * 64,
        source_content_hash="d" * 64,
        source_code_commit="e" * 40,
    )

    assert row["value"] == expected
    assert row["metadata"]["reporting_value"] == 0.0
    assert row["metadata"]["empirical"] is False
    assert row["metadata"]["exactness_proved"] is False


def test_split_requires_provenance_before_returning_any_ids() -> None:
    with pytest.raises(YAHPOCoverageError, match="provenance-complete"):
        deterministic_yahpo_split(_synthetic_inventory(), set())


def test_split_is_reproducible_counted_and_sealed_test_disjoint() -> None:
    inventory = _synthetic_inventory()
    eligible = {
        yahpo_task_id(scenario, instance) for scenario, instances in inventory.items() for instance in instances
    }
    split = deterministic_yahpo_split(inventory, eligible)

    assert split == deterministic_yahpo_split(inventory, eligible)
    assert not set(split.train_task_ids) & set(split.validation_task_ids)
    assert not (set(split.train_task_ids) | set(split.validation_task_ids)) & official_yahpo_task_ids()
    assert Counter(task_id.split("/")[2] for task_id in split.train_task_ids) == YAHPO_TRAIN_COUNTS
    assert Counter(task_id.split("/")[2] for task_id in split.validation_task_ids) == YAHPO_VALIDATION_COUNTS


@pytest.mark.parametrize("multiplier", [0.6, 0.8, 1.0])
def test_budget_multiplier_changes_training_only_and_keeps_initial_design(multiplier: float) -> None:
    training_budget = apply_yahpo_budget_multiplier(126, multiplier, initial_design_size=16, split="train")
    assert training_budget >= 16
    assert training_budget == pytest.approx(int(__import__("math").ceil(126 * multiplier)))

    assert apply_yahpo_budget_multiplier(126, 1.0, initial_design_size=16, split="validation") == 126
    with pytest.raises(ValueError, match="full native budget"):
        apply_yahpo_budget_multiplier(
            126, multiplier if multiplier != 1.0 else 0.8, initial_design_size=16, split="test"
        )


def test_installed_non_test_yahpo_configs_are_resolved_without_writing_carps() -> None:
    lcbench = get_task_config("yahpo/so/lcbench/3945/None")
    conditional = get_task_config("yahpo/so/rbv2_super/28/None")

    assert lcbench.task.name == "yahpo/so/lcbench/3945/None"
    assert lcbench.task.objective_function.instance == "3945"
    assert int(lcbench.task.optimization_resources.n_trials) == 126
    assert conditional.task.name == "yahpo/so/rbv2_super/28/None"
    assert conditional.task.objective_function.instance == "28"
    assert int(conditional.task.optimization_resources.n_trials) == 267
    assert bool(conditional.task.metadata.search_space_has_conditionals)


def test_real_yahpo_reference_reward_has_effective_budget_and_no_leakage(tmp_path: Path) -> None:
    reference_table = Path("dacboenv/experiment/analysis/yahpo_best_known_references.json").resolve()
    env = real_structured_yahpo_env(
        "yahpo/so/lcbench/3945/None",
        881_006,
        initial_design_n_configs=2,
        context_split="train",
        budget_multiplier=0.6,
        random_design_probability=1.0,
        reference_table=reference_table,
        reference_breach_path=tmp_path / "reference-breaches.jsonl",
    )
    try:
        observation, info = env.reset()
        effective_budget = int(__import__("math").ceil(126 * 0.6))
        assert env._n_trials == effective_budget
        assert info["bo_budget"] == effective_budget
        assert info["bo_evaluations"] == 2
        assert observation["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] == pytest.approx(
            2 / effective_budget
        )
        assert np.isfinite(observation["global_state"]).all()
        assert env.observation_space.contains(observation)
        assert env._objective_reference is not None
        assert env._objective_reference.value == -100.0
        assert env._objective_reference.metadata["objective_target"] == "val_accuracy"
        assert all("reference" not in key.lower() for key in observation)
        assert all("reference" not in key.lower() for key in info)
        assert str(env._objective_reference.value) not in str(info)

        next_observation, reward, terminated, truncated, step_info = env.step(0)
        assert np.isfinite(float(reward))
        assert not terminated
        assert not truncated
        assert env.observation_space.contains(next_observation)
        assert all("reference" not in key.lower() for key in step_info)
    finally:
        env.close()


def test_budget_multiplier_does_not_mutate_best_known_reference() -> None:
    reference_path = Path("dacboenv/experiment/analysis/yahpo_best_known_references.json")
    provider = ManifestReferenceProvider(
        reference_path,
        expected_runtime_objective_transform="negative_accuracy",
        expected_reporting_objective_transform="one_minus_accuracy",
        expected_fidelity="fixed_maximum",
    )
    task_id = "yahpo/so/lcbench/3945/None"
    before = provider.get_reference(task_id, None, None)
    assert apply_yahpo_budget_multiplier(126, 0.6, initial_design_size=2, split="train") == 76
    after = provider.get_reference(task_id, None, None)
    assert after is before
    assert after.value == -100.0
    assert after.metadata["provenance_status"] == "complete"
    assert after.metadata["source_method"] == ASSUMED_BOUND_SOURCE_METHOD


def test_checked_table_records_explicit_non_empirical_assumed_bounds() -> None:
    path = Path("dacboenv/experiment/analysis/yahpo_best_known_references.json")
    table = json.loads(path.read_text(encoding="utf-8"))
    rows = table["references"]

    assert table["status"] == "complete"
    assert len(rows) == 608
    assert {row["value"] for row in rows} == {-100.0, -1.0}
    assert all(row["kind"] == "best_known" for row in rows)
    assert all(row["metadata"]["provenance_status"] == "complete" for row in rows)
    assert all(row["metadata"]["source_method"] == ASSUMED_BOUND_SOURCE_METHOD for row in rows)
    assert all(row["metadata"]["empirical"] is False for row in rows)
    assert all(row["metadata"]["exactness_proved"] is False for row in rows)


def test_generator_reports_exact_source_content_and_fail_closed_status() -> None:
    identity = source_provenance()
    generator_path = Path(identity["source_content_path"])
    digest = hashlib.sha256(generator_path.read_bytes()).hexdigest()

    assert identity["source_content_sha256"] == digest
    assert identity["source_repository_status"] in {"clean", "dirty"}
    if identity["provenance_status"] == "complete":
        assert identity["source_repository_status"] == "clean"
        assert identity["source_code_commit_contains_method"] is True
    else:
        assert identity["provenance_status"] == "smoke_only_incomplete"
