"""OptBench exact-reference inventory and Double-DQN launcher coverage."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from carps.utils.running import make_task
from dacboenv.experiment.index_optbench_carps import _validate_index
from dacboenv.experiment.optbench_inventory import audit_optbench_inventory
from dacboenv.experiment.ppo import assign_training_worker_context
from dacboenv.experiment.protocol import load_manifest, manifest_hash
from dacboenv.optimizer import DACBOEnvOptimizer
from dacboenv.reference import OptBenchExactReferenceProvider, ReferenceLookupError
from dacboenv.utils.carps_optimizer import (
    get_installed_optbench_task_configs,
    get_optbench_task_dimension,
    get_task_config,
    task_ids_equivalent,
)
from hydra import compose, initialize_config_module

pytest.importorskip("optbench")

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "dacboenv" / "configs" / "instance_sets" / "optbench_train.yaml"
LAUNCHER_PATH = ROOT / "scripts" / "launch_optbench_ddqn_f5.sh"
EVALUATION_LAUNCHER_PATH = ROOT / "scripts" / "otus" / "otus_eval_final_sb3.sh"
EXPECTED_TASKS = {
    "optbench/Ackley-2",
    "optbench/Ackley-5",
    "optbench/Ackley-10",
    "optbench/Levy-2",
    "optbench/Levy-5",
    "optbench/Levy-10",
    "optbench/Schwefel-2",
    "optbench/Schwefel-5",
    "optbench/Schwefel-10",
    "optbench/Hartmann3",
    "optbench/Hartmann6",
}


def _compose_training(name: str):
    """Compose an OptBench training config without starting SB3."""
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        return compose(config_name=None, overrides=[f"+training={name}"])


def test_optbench_manifest_contains_only_tasks_with_finite_global_minima() -> None:
    """Every selected live objective exposes both a minimizer and finite minimum."""
    manifest = load_manifest(MANIFEST_PATH)

    assert set(manifest["task_ids"]) == EXPECTED_TASKS
    assert manifest["manifest_hash"] == manifest_hash(manifest)
    assert "optbench/Hartmann4" not in manifest["task_ids"]
    assert manifest["selection_protocol"]["excluded_tasks"] == [
        {
            "config": "Hartmann_4",
            "task_id": "optbench/Hartmann4",
            "reason": "objective exposes f_min=None and x_min=None",
        }
    ]

    for task_id in manifest["task_ids"]:
        cfg = get_task_config(task_id)
        cfg.seed = 0
        objective = make_task(cfg).objective_function
        assert objective.x_min is not None
        assert np.asarray(objective.x_min, dtype=np.float64).shape == (get_optbench_task_dimension(task_id),)
        assert math.isfinite(float(objective.f_min))


def test_installed_optbench_inventory_keeps_hartmann_tasks_distinct() -> None:
    """The installed Hartmann-6 config must not accidentally construct Hartmann-4."""
    inventory = get_installed_optbench_task_configs()

    assert set(inventory) == EXPECTED_TASKS | {"optbench/Hartmann4"}
    cfg = get_task_config("optbench/Hartmann6")
    cfg.seed = 0
    objective = make_task(cfg).objective_function
    assert type(objective).__name__ == "Hartmann6"
    assert objective.f_min == pytest.approx(-3.32237)


def test_optbench_canonical_and_carps_display_task_ids_are_equivalent() -> None:
    """Only OptBench's canonical namespace may match its plugin display name."""
    assert task_ids_equivalent("optbench/Ackley-2", "Ackley-2")
    assert task_ids_equivalent("Ackley-2", "optbench/Ackley-2")
    assert task_ids_equivalent("bbob/2/1/0", "bbob/2/1/0")
    assert not task_ids_equivalent("optbench/Ackley-2", "Ackley-5")
    assert not task_ids_equivalent("bbob/2/1/0", "2/1/0")


def test_dacbo_optimizer_cleanup_tolerates_failed_setup() -> None:
    """A primary setup exception must not trigger CARP-S's destructor error."""
    uninitialized = DACBOEnvOptimizer.__new__(DACBOEnvOptimizer)
    uninitialized._solver = None
    uninitialized.__del__()

    callback = Mock()
    initialized = DACBOEnvOptimizer.__new__(DACBOEnvOptimizer)
    initialized._solver = SimpleNamespace(optimizer=SimpleNamespace(_callbacks=[callback]))
    initialized.__del__()
    callback.on_end.assert_called_once_with(initialized._solver.optimizer)
    initialized._solver = None


def test_installed_inventory_audit_matches_the_frozen_finite_subset() -> None:
    """The Otus pre-submit audit validates the editable installation itself."""
    audit = audit_optbench_inventory(MANIFEST_PATH)

    assert audit["selected_task_count"] == 11
    assert audit["excluded_task_count"] == 1
    excluded = [record for record in audit["records"] if not record["selected"]]
    assert excluded == [
        {
            "task_id": "optbench/Hartmann4",
            "dimension": 4,
            "objective_class": "Hartmann4",
            "f_min": None,
            "has_finite_x_min": False,
            "selected": False,
        }
    ]


def test_optbench_carps_index_requires_unique_current_config_paths(tmp_path: Path) -> None:
    """Gathering accepts only exact, unique OptBench index mappings."""
    config = tmp_path / "Ackley_2.yaml"
    config.touch()
    expected = {"Ackley-2": config.resolve()}
    valid = pd.DataFrame([{"task_id": "Ackley-2", "config_fn": str(config)}])

    assert _validate_index(valid, expected) == [{"task_id": "Ackley-2", "config_fn": str(config.resolve())}]
    with pytest.raises(RuntimeError, match="not unique/current"):
        _validate_index(pd.concat([valid, valid], ignore_index=True), expected)
    with pytest.raises(RuntimeError, match="not unique/current"):
        _validate_index(pd.DataFrame([{"task_id": "Ackley-2", "config_fn": str(tmp_path / "stale.yaml")}]), expected)


def test_optbench_exact_reference_provider_fails_closed() -> None:
    """Only namespaced tasks with finite live minima receive exact references."""
    provider = OptBenchExactReferenceProvider(source_hash="a" * 64)
    metadata = {
        "task_id": "optbench/Ackley-2",
        "runtime_objective_transform": "identity",
        "reporting_objective_transform": "identity",
        "fidelity": "not_applicable",
    }
    reference = provider.get_reference("optbench/Ackley-2", SimpleNamespace(f_min=0.0), metadata)

    assert reference.kind == "exact"
    assert reference.value == 0.0
    with pytest.raises(ReferenceLookupError, match="finite global minimum"):
        provider.get_reference("optbench/Hartmann4", SimpleNamespace(f_min=None), None)
    with pytest.raises(ReferenceLookupError, match="does not cover"):
        provider.get_reference("bbob/2/1/0", SimpleNamespace(f_min=0.0), None)


def test_optbench_workers_are_persistently_dimension_balanced() -> None:
    """Persistent workers cycle over OptBench dimensions and no absent task."""
    task_ids = sorted(EXPECTED_TASKS)
    assignments = [
        assign_training_worker_context(task_ids, worker_id=worker_id, n_workers=12) for worker_id in range(12)
    ]

    assert Counter(assignment.optbench_dimension for assignment in assignments) == {2: 3, 3: 3, 5: 2, 6: 2, 10: 2}
    assert all(assignment.domain == "optbench" for assignment in assignments)
    assert all(
        {get_optbench_task_dimension(task_id) for task_id in assignment.task_ids} == {assignment.optbench_dimension}
        for assignment in assignments
    )


@pytest.mark.parametrize(
    ("config_name", "bo_budget", "timesteps", "checkpoint_frequency"),
    [
        ("optbench_wei_double_dqn_f5_d1", 61_440, 12_288, 3_072),
        ("optbench_wei_double_dqn_f5_d1_short", 30_720, 6_144, 1_536),
    ],
)
def test_optbench_ddqn_training_configs(
    config_name: str,
    bo_budget: int,
    timesteps: int,
    checkpoint_frequency: int,
) -> None:
    """OptBench configs preserve the D1 f5 accounting and disable in-sample validation."""
    cfg = _compose_training(config_name)

    assert cfg.rl_algorithm_id == "double_dqn"
    assert cfg.rl_algorithm.algorithm_class == "dacboenv.rl.double_dqn.DoubleDQN"
    assert cfg.experiment.n_workers == 12
    assert cfg.dacboenv.interaction_frequency == 5
    assert cfg.experiment.bo_evaluation_budget == bo_budget
    assert cfg.experiment.total_timesteps == timesteps
    assert cfg.experiment.checkpoint_freq == checkpoint_frequency
    assert cfg.rl_algorithm.hyperparameters.gradient_steps == 3
    assert cfg.rl_algorithm.hyperparameters.gamma == 1.0
    assert cfg.experiment.validation.enabled is False
    assert cfg.experiment.final_evaluation.enabled is False
    assert set(cfg.dacboenv.task_ids) == EXPECTED_TASKS
    assert cfg.dacboenv.reference_provider._target_ == "dacboenv.reference.OptBenchExactReferenceProvider"


def test_optbench_launcher_uses_otus_environment_and_three_outer_seeds() -> None:
    """The launcher follows the current Otus worker and array conventions."""
    text = LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "${repo_root}/.venv/bin/python" in text
    assert ".env/bin/python" not in text
    assert "--array=0-2" in text
    assert "DACBO_N_WORKERS=12" in text
    assert "scripts/opt_ppo.sh" in text
    assert "dacboenv.experiment.optbench_inventory" in text
    assert "optbench_wei_double_dqn_f5_d1" in text
    assert "optbench_wei_double_dqn_f5_d1_short" in text


def test_algorithm_neutral_final_evaluator_has_exact_optbench_mode() -> None:
    """Final Double-DQN bundles can be evaluated on only eligible OptBench tasks."""
    text = EVALUATION_LAUNCHER_PATH.read_text(encoding="utf-8")

    assert "list|bbob|yahpo|optbench|both|all|gather" in text
    assert "+task/OptBench=${optbench_tasks}" in text
    assert r"dacboenv.task_ids=[optbench/\${task.name}]" in text
    assert "pkg://optbench/configs" in text
    assert "dacboenv.experiment.optbench_inventory" in text
    assert "dacboenv.experiment.index_optbench_carps" in text
    assert "optbench_exact" in text
    assert "Hartmann_4" not in text
    assert "Hartmann_3,Hartmann_6" in text
