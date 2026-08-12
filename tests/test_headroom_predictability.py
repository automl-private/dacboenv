"""Scientific contracts for same-state headroom prediction."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
from dacboenv.experiment.build_headroom_campaign import build_campaign_inventory
from dacboenv.experiment.headroom_predictability import (
    SharedActionScorer,
    add_action_advantages,
    assert_disjoint_context_splits,
    assert_predictor_columns_safe,
    campaign_manifest_hash,
    evaluate_selector_decomposition,
    grouped_bootstrap_mean,
    parse_policy_observation,
    tie_aware_accuracy,
)
from dacboenv.experiment.run_action_predictability import ObservationMatrices, _negative_control


def _manifest(domain: str, split: str) -> dict[str, object]:
    if domain == "bbob":
        tasks = ["bbob/2/1/0", "bbob/4/2/0"] if split == "train" else ["bbob/4/3/1", "bbob/8/4/1"]
    else:
        scenarios = ("lcbench", "rbv2_glmnet", "rbv2_rpart", "rbv2_ranger", "rbv2_xgboost", "rbv2_super")
        suffix = "train" if split == "train" else "validation"
        tasks = [f"yahpo/so/{scenario}/{suffix}/None" for scenario in scenarios]
    return {
        "id": f"{domain}-{split}",
        "domain": domain,
        "split": split,
        "manifest_hash": f"hash-{domain}-{split}",
        "task_ids": tasks,
    }


def _rows(split: str, task: str, q_by_snapshot: list[list[float]]) -> pd.DataFrame:
    records = []
    domain = "yahpo" if task.startswith("yahpo") else "bbob"
    for snapshot_index, values in enumerate(q_by_snapshot):
        for action, value in enumerate(values):
            records.append(
                {
                    "snapshot_id": f"{split}-{snapshot_index}",
                    "task_id": task,
                    "inner_seed": snapshot_index,
                    "history_generator": "static",
                    "history_seed": 7,
                    "domain": domain,
                    "scenario": "lcbench" if domain == "yahpo" else "",
                    "dimension": 2 if domain == "bbob" else np.nan,
                    "function_group": 1,
                    "budget_fraction": (0.25, 0.5, 0.75)[snapshot_index % 3],
                    "action": action,
                    "horizon": 5,
                    "q_value": value,
                }
            )
    return pd.DataFrame(records)


def test_campaign_inventory_is_stable_balanced_and_disjoint() -> None:
    manifests = {
        "bbob_train": _manifest("bbob", "train"),
        "bbob_validation": _manifest("bbob", "validation"),
        "yahpo_train": _manifest("yahpo", "train"),
        "yahpo_validation": _manifest("yahpo", "validation"),
    }
    reference_ids = set(manifests["yahpo_train"]["task_ids"]) | set(manifests["yahpo_validation"]["task_ids"])
    train, validation = build_campaign_inventory(**manifests, reference_ids=reference_ids)
    repeated = build_campaign_inventory(**manifests, reference_ids=reference_ids)
    assert (train, validation) == repeated
    assert len(train) == 480
    assert len(validation) == 240
    assert {row["task_id"] for row in train}.isdisjoint(row["task_id"] for row in validation)
    assert campaign_manifest_hash(train) == campaign_manifest_hash(repeated[0])
    for family in ("wei", "af_selection"):
        family_train = [row for row in train if row["action_family"] == family]
        assert len(family_train) == 240
        assert sum(row["domain"] == "bbob" for row in family_train) == 120


def test_observation_parser_rejects_reference_leakage() -> None:
    payload = {
        "global_state": {"dtype": "float32", "shape": [2], "values": [0.1, 0.2]},
        "action_features": {"dtype": "float32", "shape": [2, 2], "values": [[1, 2], [3, 4]]},
    }
    global_state, action_features = parse_policy_observation(json.dumps(payload))
    assert global_state.shape == (2,)
    assert action_features.shape == (2, 2)
    payload["reference_value"] = 0.0
    with pytest.raises(ValueError, match="Expected only"):
        parse_policy_observation(json.dumps(payload))
    with pytest.raises(ValueError, match="privileged"):
        assert_predictor_columns_safe(["global_state", "best_known_reference"])


def test_shared_action_scorer_is_action_permutation_equivariant() -> None:
    torch.manual_seed(5)
    scorer = SharedActionScorer(global_size=3, action_feature_size=4)
    global_state = torch.randn(7, 3)
    action_features = torch.randn(7, 5, 4)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    original = scorer(global_state, action_features)
    permuted = scorer(global_state, action_features[:, permutation])
    torch.testing.assert_close(permuted, original[:, permutation])


def test_tie_metrics_advantages_and_grouped_bootstrap() -> None:
    values = np.asarray([[1.0, 1.0, 0.0], [0.2, 0.1, 0.19995]])
    assert tie_aware_accuracy(values, np.asarray([1, 2]), 0.0) == pytest.approx(0.5)
    assert tie_aware_accuracy(values, np.asarray([1, 2]), 1e-4) == 1.0
    rows = add_action_advantages(_rows("train", "bbob/2/1/0", values.tolist()))
    assert np.allclose(rows.groupby("snapshot_id")["action_advantage"].sum(), 0.0)
    rows["function"] = 1
    samples = grouped_bootstrap_mean(
        rows,
        "q_value",
        ["function_group", "function", "task_id", "inner_seed", "snapshot_id"],
        n_resamples=20,
        seed=3,
    )
    assert samples.shape == (20,)


def test_selector_decomposition_fits_only_disjoint_training_trajectories() -> None:
    train = _rows("train", "bbob/2/1/0", [[3.0, 0.0], [2.0, 0.0], [0.0, 4.0]])
    validation = _rows("validation", "bbob/4/3/1", [[2.0, 1.0], [0.0, 3.0]])
    values = evaluate_selector_decomposition(train, validation, minimum_support=1)
    assert values.oracle >= values.global_static
    overlapping = validation.copy()
    overlapping[["task_id", "inner_seed", "history_generator", "history_seed"]] = train.loc[
        0, ["task_id", "inner_seed", "history_generator", "history_seed"]
    ].to_numpy()
    with pytest.raises(ValueError, match="share complete trajectories"):
        assert_disjoint_context_splits(train, overlapping)


def test_negative_controls_are_seeded_and_do_not_mutate_source() -> None:
    source = ObservationMatrices(
        ids=("a", "b"),
        global_state=np.arange(6, dtype=np.float32).reshape(2, 3),
        action_features=np.arange(40, dtype=np.float32).reshape(2, 5, 4),
        q_values=np.arange(10, dtype=np.float32).reshape(2, 5),
        groups=("g1", "g2"),
    )
    original = source.action_features.copy()
    first_train, first_validation = _negative_control(source, source, "shared_mismatched_rows")
    second_train, second_validation = _negative_control(source, source, "shared_mismatched_rows")
    np.testing.assert_array_equal(first_train.action_features, second_train.action_features)
    np.testing.assert_array_equal(first_validation.action_features, second_validation.action_features)
    np.testing.assert_array_equal(source.action_features, original)
    assert not np.array_equal(first_train.action_features, source.action_features)
