"""Focused v3 metadata, subgroup, history, and fail-closed tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from dacboenv.experiment.scenario_v3 import _subgroups, attach_metadata, captured_ratio
from dacboenv.experiment.snapshot_branch import BOSnapshot
from dacboenv.experiment.task_metadata import EXPECTED_YAHPO_SCENARIOS, parse_task_metadata


def test_canonical_task_metadata_parses_all_allowed_scenarios() -> None:
    for scenario in EXPECTED_YAHPO_SCENARIOS:
        metadata = parse_task_metadata(f"yahpo/so/{scenario}/dataset/None")
        assert metadata.domain == "yahpo"
        assert metadata.scenario == scenario
        assert metadata.dataset_instance == "dataset"
    bbob = parse_task_metadata("bbob/8/20/1")
    assert (bbob.domain, bbob.scenario, bbob.dimension, bbob.function_id, bbob.native_instance) == (
        "bbob",
        None,
        8,
        20,
        "1",
    )


def test_canonical_parser_rejects_nb301_and_empty_instance() -> None:
    with pytest.raises(PermissionError):
        parse_task_metadata("yahpo/so/nb301/CIFAR10/None")
    with pytest.raises(ValueError, match="instance is empty"):
        parse_task_metadata("yahpo/so/lcbench//None")


def test_metadata_repair_overwrites_empty_legacy_labels() -> None:
    rows = pd.DataFrame({"task_id": ["yahpo/so/rbv2_xgboost/23517/None"], "domain": [""], "scenario": [""]})
    repaired = attach_metadata(rows)
    assert repaired.loc[0, "domain"] == "yahpo"
    assert repaired.loc[0, "scenario"] == "rbv2_xgboost"
    assert repaired.loc[0, "dataset_instance"] == "23517"


def test_exclusion_and_leave_one_scenario_subsets_are_strict() -> None:
    rows = []
    for scenario in EXPECTED_YAHPO_SCENARIOS:
        rows.append(
            {
                "domain": "yahpo",
                "scenario": scenario,
                "gap": 0.1,
                "dimension": np.nan,
                "target_budget_fraction": 0.5,
                "history_generator": "static",
            }
        )
    groups = _subgroups(pd.DataFrame(rows))
    assert len(groups["domain:yahpo"]) == 6
    assert len(groups["yahpo_excluding_rbv2_xgboost"]) == 5
    assert all(len(groups[f"yahpo_leave_out:{scenario}"]) == 5 for scenario in EXPECTED_YAHPO_SCENARIOS)


def test_captured_ratio_uses_local_values_and_undefined_denominator() -> None:
    assert captured_ratio(3.0, 2.0, 4.0) == 0.5
    assert np.isnan(captured_ratio(3.0, 2.0, 2.0))


def test_sawei_continuous_action_cannot_be_silently_serialized_as_discrete_snapshot() -> None:
    with pytest.raises(TypeError, match="integers"):
        BOSnapshot("bbob/2/3/0", 919, action_history=(0.75,))
