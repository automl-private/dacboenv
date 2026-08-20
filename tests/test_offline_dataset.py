"""Tests for the CARP-S offline transition dataset workflow."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dacboenv.env.action import AcqParameterActionSpace, WEIDiscreteActionSpace, WEITempoRLActionSpace
from dacboenv.experiment.offline_dataset import (
    OFFLINE_DATASET_SCHEMA_VERSION,
    OFFLINE_OBSERVATION_KEYS,
    canonical_wei_action,
    validate_episode_npz,
)
from dacboenv.experiment.offline_dataset_campaign import POLICY_COUNT, build_policies, consolidate
from dacboenv.policy.random import DoubleRandomWEIPolicy
from gymnasium.spaces import MultiDiscrete
from omegaconf import OmegaConf


def test_double_random_samples_both_axes_reproducibly() -> None:
    """The policy owns an independent seeded RNG for both categorical axes."""
    env = SimpleNamespace(action_space=MultiDiscrete([3, 5]))
    first = DoubleRandomWEIPolicy(env)
    second = DoubleRandomWEIPolicy(env)
    first.set_seed(17)
    second.set_seed(17)

    actions_first = np.stack([first(None) for _ in range(128)])
    actions_second = np.stack([second(None) for _ in range(128)])

    np.testing.assert_array_equal(actions_first, actions_second)
    assert set(actions_first[:, 0]) == {0, 1, 2}
    assert set(actions_first[:, 1]) == {0, 1, 2, 3, 4}


def test_double_random_rejects_non_tempo_action_space() -> None:
    """A one-axis controller cannot be silently interpreted as double random."""
    with pytest.raises(TypeError, match="two-axis WEI TempoRL"):
        DoubleRandomWEIPolicy(SimpleNamespace(action_space=MultiDiscrete([5])))


def test_canonical_wei_action_unifies_supported_controllers() -> None:
    """Static, double-random, and SAWEI actions share two numeric columns."""
    tempo = object.__new__(WEITempoRLActionSpace)
    tempo._step_durations = [1, 5, 10]
    tempo._param_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    tempo_action = canonical_wei_action(tempo, np.asarray([2, 3]), 1)
    assert (tempo_action.alpha, tempo_action.requested_duration) == (0.75, 10)
    assert (tempo_action.alpha_index, tempo_action.duration_index) == (3, 2)

    static = object.__new__(WEIDiscreteActionSpace)
    static._param_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    static_action = canonical_wei_action(static, 1, 5)
    assert (static_action.alpha, static_action.requested_duration) == (0.25, 5)
    assert (static_action.alpha_index, static_action.duration_index) == (1, -1)

    continuous = object.__new__(AcqParameterActionSpace)
    continuous._action = SimpleNamespace(attr="_alpha")
    sawei_action = canonical_wei_action(continuous, 0.63, 1)
    assert (sawei_action.alpha, sawei_action.requested_duration) == (0.63, 1)
    assert (sawei_action.alpha_index, sawei_action.duration_index) == (-1, -1)


def test_policy_registry_has_five_static_and_two_dynamic_rows() -> None:
    """The frozen campaign expands to exactly seven scientifically distinct policies."""
    path = Path("dacboenv/configs/offline_dataset/carps_bbob_yahpo_v1.yaml")
    config = OmegaConf.load(path).offline_dataset
    policies = build_policies(config)

    assert len(policies) == POLICY_COUNT == 7
    assert sum(policy["policy_kind"] == "static" for policy in policies) == 5
    assert sum(policy["policy_kind"] == "double_random" for policy in policies) == 1
    assert sum(policy["policy_kind"] == "sawei" for policy in policies) == 1
    assert {(policy["alpha"], policy["duration"]) for policy in policies if policy["policy_kind"] == "static"} == {
        (alpha, 1) for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]
    }


def _write_valid_shard(path: Path, *, seed: int = 0) -> None:
    arrays: dict[str, np.ndarray] = {}
    shapes = {
        "global_state": (13,),
        "action_features": (5, 4),
        "gp_hp_summary": (24,),
        "gp_hp_change": (8,),
        "gp_hp_raw": (64,),
        "gp_hp_raw_mask": (64,),
        "gp_hp_raw_roles": (64, 4),
    }
    for key, shape in shapes.items():
        arrays[f"observations__{key}"] = np.zeros((2, *shape), dtype=np.float32)
        arrays[f"next_observations__{key}"] = np.ones((2, *shape), dtype=np.float32)
    arrays.update(
        {
            "actions": np.asarray([[0.0, 1.0], [1.0, 5.0]], dtype=np.float32),
            "action_alpha_index": np.asarray([0, 4], dtype=np.int8),
            "action_duration_index": np.asarray([-1, -1], dtype=np.int8),
            "rewards": np.asarray([0.1, -0.2], dtype=np.float64),
            "terminated": np.asarray([False, True]),
            "truncated": np.asarray([False, False]),
            "terminals": np.asarray([False, True]),
            "timeouts": np.asarray([False, False]),
            "requested_duration": np.asarray([1, 5], dtype=np.int16),
            "realized_duration": np.asarray([1, 3], dtype=np.int16),
            "bo_evaluations_before": np.asarray([15, 16], dtype=np.int32),
            "bo_evaluations_after": np.asarray([16, 19], dtype=np.int32),
            "transition_index": np.asarray([0, 1], dtype=np.int32),
            "metadata_json": np.asarray(
                json.dumps(
                    {
                        "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
                        "task_id": "bbob/2/1/0",
                        "seed": seed,
                        "policy_id": "fixture",
                        "transition_count": 2,
                        "observation_keys": list(OFFLINE_OBSERVATION_KEYS),
                    }
                )
            ),
        }
    )
    np.savez(path, **arrays)


def test_npz_validation_is_pickle_free_and_checks_identity(tmp_path: Path) -> None:
    """A portable shard loads with ``allow_pickle=False`` and preserves identity."""
    path = tmp_path / "episode.npz"
    _write_valid_shard(path)
    metadata = validate_episode_npz(
        path,
        expected_task_id="bbob/2/1/0",
        expected_seed=0,
        expected_policy_id="fixture",
    )
    assert metadata["transition_count"] == 2
    with np.load(path, allow_pickle=False) as payload:
        assert all(not payload[key].dtype.hasobject for key in payload.files)


def test_structured_gp_all_configs_have_common_stored_schema() -> None:
    """SAWEI may consume extra signals without changing stored observation keys."""
    ordinary = OmegaConf.load("dacboenv/configs/env/obs/structured_gp_all.yaml")
    sawei = OmegaConf.load("dacboenv/configs/env/obs/structured_gp_all_sawei.yaml")
    assert tuple(ordinary.dacboenv.observation_keys) == OFFLINE_OBSERVATION_KEYS
    assert tuple(sawei.dacboenv.observation_keys[: len(OFFLINE_OBSERVATION_KEYS)]) == OFFLINE_OBSERVATION_KEYS
    assert tuple(sawei.dacboenv.observation_keys[len(OFFLINE_OBSERVATION_KEYS) :]) == (
        "ubr_smoothed_gradient",
        "acq_value_PI",
        "acq_value_WEI_explore",
        "previous_param",
    )


def test_disk_backed_consolidation_creates_standard_npz(tmp_path: Path) -> None:
    """Episode shards consolidate to one ordinary pickle-free NumPy archive."""
    rows = []
    for seed in [0, 1]:
        episode = tmp_path / "runs" / str(seed) / "offline_episode.npz"
        episode.parent.mkdir(parents=True)
        _write_valid_shard(episode, seed=seed)
        rows.append(
            {
                "job_index": seed + 1,
                "scientific_id": f"fixture-{seed}",
                "task_id": "bbob/2/1/0",
                "seed": seed,
                "policy_id": "fixture",
                "output_path": str(episode),
            }
        )
    (tmp_path / "inventory.json").write_text(
        json.dumps(
            {
                "schema_version": OFFLINE_DATASET_SCHEMA_VERSION,
                "manifest_hash": "fixture-manifest",
                "source_revision": "fixture-source",
                "reference_table_sha256": "fixture-reference",
                "observation_keys": list(OFFLINE_OBSERVATION_KEYS),
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    config = OmegaConf.create(
        {
            "output_root": str(tmp_path),
            "allow_incomplete_consolidation": False,
            "consolidated_filename": "dataset.npz",
            "compression": "deflated",
        }
    )

    result = consolidate(config)

    assert result["episode_count"] == 2
    assert result["transition_count"] == 4
    with np.load(tmp_path / "dataset.npz", allow_pickle=False) as dataset:
        np.testing.assert_array_equal(dataset["episode_offsets"], [0, 2, 4])
        np.testing.assert_array_equal(dataset["episode_index"], [0, 0, 1, 1])
        assert dataset["observations__global_state"].shape == (4, 13)
        assert json.loads(str(dataset["dataset_metadata_json"].item()))["episode_count"] == 2
        first_episode = json.loads(str(dataset["episode_metadata_json"][0]))
        assert first_episode["scientific_id"] == "fixture-0"
