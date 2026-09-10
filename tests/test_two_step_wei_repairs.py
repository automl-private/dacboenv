"""Regression checks for unsupported selective inference and abstention."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from dacboenv.selective_wei.base_selector import fit_base_selector
from dacboenv.selective_wei.calibration import select_frontier_row
from dacboenv.selective_wei.context import context_from_task_id
from dacboenv.selective_wei.metrics import evaluate_selective_actions
from dacboenv.selective_wei.uncertainty import candidate_equivalence, finite_sample_conformal_quantile


@pytest.mark.parametrize(
    ("n", "delta", "finite"), [(5, 0.1, False), (9, 0.1, True), (18, 0.05, False), (19, 0.05, True), (0, 0.1, False)]
)
def test_conformal_unsupported_rank_abstains(n: int, delta: float, finite: bool) -> None:
    """A missing finite-sample order statistic is not clipped."""
    result = finite_sample_conformal_quantile(np.arange(n, dtype=float), delta)
    assert bool(np.isfinite(result)) == finite


def test_negative_gain_frontier_prefers_base() -> None:
    """A feasible but harmful-average gate cannot beat abstention."""
    rows = [
        {
            "feasible": True,
            "task_balanced_mean_gain": -1.0,
            "harmful_override_fraction": 0.01,
            "override_coverage": 0.1,
            "parameter_hash": "a",
        }
    ]
    selected = select_frontier_row(rows)
    assert selected["task_balanced_mean_gain"] == 0
    assert selected["override_count"] == 0
    assert any(row.get("override_count") == 0 for row in rows)


def test_no_overrides_has_no_conditional_harm_estimate() -> None:
    """Zero observations of an override are not evidence of zero risk."""
    metrics = evaluate_selective_actions(
        np.zeros((2, 5)), np.zeros(2, dtype=int), np.zeros(2, dtype=int), np.asarray(["a", "b"])
    )
    assert metrics["harmful_override_fraction"] is None


def test_equal_features_do_not_prove_delayed_ties() -> None:
    """Compressed candidate statistics cannot define H5 label equivalence."""
    features = np.zeros((5, 4))
    features[:, 0] = np.linspace(0, 1, 5)
    equivalence = candidate_equivalence(features)
    assert not equivalence.equivalent(0, 1)


def test_singleton_lcb_uses_supported_domain() -> None:
    """One task cannot estimate a contextual sampling standard error."""
    selector = fit_base_selector(
        q_values=np.asarray([[0, 0, 1, 0, 0]] * 2),
        task_ids=np.asarray(["bbob/2/1/0", "bbob/8/1/0"]),
        phase_bins=None,
        kind="context_lcb",
        horizon=5,
        source_split="train",
        fit_manifest_hash="m",
        fit_data_hash="d",
        code_revision="r",
    )
    decision = selector.select(context_from_task_id("bbob/2/1/0"))
    assert decision.action == 2
    assert decision.fallback_level == "domain"


def test_real_hydra_cli_first_execution(tmp_path: Path) -> None:
    """Hydra bookkeeping must not make a fresh scientific output nonempty."""
    from test_selective_wei import _dev_tasks, _tasks, _write_branch  # noqa: PLC0415 - shared synthetic fixture

    branch = tmp_path / "branches"
    branch.mkdir()
    _write_branch(branch / "branches_train.npz", _tasks(), "train")
    _write_branch(branch / "branches_dev.npz", _dev_tasks(), "dev")
    output = tmp_path / "scientific"
    output.mkdir()
    command = [
        sys.executable,
        "-m",
        "dacboenv.experiment.fit_selective_wei",
        "+selective_policy=selective_b1_g3",
        "selective_predictor=neural_smoke",
        "selective_predictor.updates=1",
        "selective_evaluation.bootstrap_resamples=20",
        f"selective_data.branch_root={branch}",
        f"selective_output.root={output}",
    ]
    result = subprocess.run(  # noqa: S603 - fixed engineering CLI with pytest-owned paths
        command,
        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "selective_training_complete.json").exists()
    registry = json.loads((output / "base_selector_registry.json").read_text())
    partition = json.loads((output / "selective_train_partition_manifest.json").read_text())
    assert registry["task_ids"] == partition["partitions"]["model_fit"]
    for overrides, success in [
        (["selective_output.resume=true"], True),
        (["selective_output.resume=true", "seed=99"], False),
        ([], False),
    ]:
        resumed = subprocess.run(  # noqa: S603 - fixed command, no shell
            command + overrides,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        )
        assert (resumed.returncode == 0) == success, resumed.stdout + resumed.stderr
