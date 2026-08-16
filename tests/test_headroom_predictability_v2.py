"""Focused protocol-v2 leakage and model-contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from dacboenv.experiment.augment_headroom_with_learned_policies import (
    classify_behavior,
    preflight_run,
)
from dacboenv.experiment.headroom_v2 import enrich, fit_selector, select_nonfeedback
from dacboenv.experiment.richer_predictability_v2 import ShortHistoryGRU


def _rows() -> pd.DataFrame:
    records = []
    for split, tasks in (("train", ("bbob/2/1/0", "bbob/2/2/0")), ("validation", ("bbob/4/3/0",))):
        for task_index, task in enumerate(tasks):
            for snapshot_index in range(3):
                for action in range(5):
                    records.append(
                        {
                            "split": split,
                            "task_id": task,
                            "inner_seed": task_index,
                            "history_generator": "default_smac",
                            "action_space": "wei",
                            "campaign_snapshot_id": f"{split}-{task_index}-{snapshot_index}",
                            "domain": "bbob",
                            "scenario": "bbob",
                            "dimension": int(task.split("/")[1]),
                            "target_budget_fraction": (snapshot_index + 1) / 4,
                            "q_value": float(action == task_index),
                            "action": action,
                        }
                    )
    return enrich(pd.DataFrame(records))


def test_training_selected_nonfeedback_is_validation_independent() -> None:
    rows = _rows()
    training = rows[rows.split == "train"]
    selected, scores = select_nonfeedback(training)
    modified = rows.copy()
    modified.loc[modified.split == "validation", "q_value"] = np.arange(np.sum(modified.split == "validation"))
    assert select_nonfeedback(modified[modified.split == "train"]) == (selected, scores)


def test_deployable_and_privileged_bbob_contexts_are_separate() -> None:
    rows = _rows()
    deployable = fit_selector(rows[rows.split == "train"], "context_dimension")
    privileged = fit_selector(rows[rows.split == "train"], "context_dimension_function_group_privileged")
    assert deployable["columns"] == ["context_dimension"]
    assert privileged["columns"] == ["context_privileged"]


def test_gru_padding_mask_uses_last_available_position() -> None:
    model = ShortHistoryGRU(3, 4).eval()
    sequence = torch.zeros((1, 10, 3))
    sequence[:, 9] = 1
    actions = torch.zeros((1, 5, 4))
    with torch.no_grad():
        late = model(sequence, torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]]), actions)
        early = model(sequence, torch.tensor([[1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), actions)
    assert not torch.equal(late, early)


def test_learned_behavior_requires_value_controls() -> None:
    arguments = {
        "constant_fraction": 0.5,
        "within_episode_variation": True,
        "contextual_dependence": True,
        "phase_explains_actions": False,
        "evolving_state_increment": 0.1,
        "beats_modal": True,
        "beats_marginal": True,
        "beats_nonfeedback": True,
        "captures_positive_residual": True,
        "paired_ci_excludes_zero": True,
    }
    assert classify_behavior(**arguments) == "feedback_dynamic"
    arguments["beats_marginal"] = False
    assert classify_behavior(**arguments) == "unclassified"


def test_incomplete_learned_run_fails_preflight(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Hydra config"):
        preflight_run(tmp_path, "best", "wei")


def test_no_run_root_status_is_machine_readable(tmp_path: Path) -> None:
    payload = {
        "status": "not_executed_no_run_roots",
        "complete_runs": [],
    }
    path = tmp_path / "status.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["complete_runs"] == []
