from __future__ import annotations

import copy
import itertools
from pathlib import Path

import pytest
from dacboenv.experiment.protocol import (
    BBOB_STRESS_TEST_FUNCTIONS,
    BBOB_STRICT_TEST_FUNCTIONS,
    BBOB_TRAIN_FUNCTIONS,
    BBOB_VALIDATION_FUNCTIONS,
    EXPECTED_NATIVE_BBOB_DIMENSIONS,
    TEST_INNER_MASTER_SEED,
    VALIDATION_INNER_MASTER_SEED,
    ManifestUnavailableError,
    ManifestValidationError,
    bbob_function_ids,
    discover_native_bbob_configs,
    discover_official_yahpo_so_configs,
    expected_bbob_task_ids,
    fixed_contexts,
    frozen_inner_seeds,
    load_legacy_yahpo_references,
    load_manifest,
    manifest_hash,
    official_yahpo_so_task_ids,
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
    validate_official_yahpo_manifest,
)
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "dacboenv" / "configs" / "instance_sets"
YAHPO_REFERENCE_CSV = REPOSITORY_ROOT / "dacboenv" / "experiment" / "analysis" / "yahpo_so_fmin.csv"

BBOB_MANIFEST_NAMES = (
    "bbob_train",
    "bbob_validation",
    "bbob_test_strict",
    "bbob_test_full_2d8d",
    "bbob_test_stress_32d",
)
YAHPO_AND_MIXED_MANIFEST_NAMES = (
    "yahpo_train",
    "yahpo_validation",
    "mixed_train_60_40",
    "mixed_validation",
)


def _load(name: str) -> dict:
    return load_manifest(MANIFEST_ROOT / f"{name}.yaml")


def test_dedicated_validation_and_test_seed_lists_are_frozen_and_disjoint() -> None:
    expected_validation = (1349011988, 2024774586, 595161999, 1294824964)
    expected_test = (4186603198, 1289272855, 3065569299, 2920417400, 1709701155)

    assert frozen_inner_seeds(VALIDATION_INNER_MASTER_SEED, 4) == expected_validation
    assert frozen_inner_seeds(VALIDATION_INNER_MASTER_SEED, 4) == expected_validation
    assert frozen_inner_seeds(TEST_INNER_MASTER_SEED, 5) == expected_test
    assert set(expected_validation).isdisjoint(expected_test)


def test_all_protocol_manifests_have_valid_deterministic_content_hashes() -> None:
    names = (*BBOB_MANIFEST_NAMES, "yahpo_test_official_so", *YAHPO_AND_MIXED_MANIFEST_NAMES)
    manifests = [_load(name) for name in names]

    assert len({manifest["manifest_hash"] for manifest in manifests}) == len(manifests)
    reordered = dict(reversed(list(manifests[0].items())))
    assert manifest_hash(reordered) == manifests[0]["manifest_hash"]

    tampered = copy.deepcopy(manifests[0])
    tampered["id"] = "tampered"
    with pytest.raises(ManifestValidationError, match="hash mismatch"):
        validate_manifest_structure(tampered)


def test_bbob_manifests_exactly_match_the_requested_splits() -> None:
    train = _load("bbob_train")
    validation = _load("bbob_validation")
    strict = _load("bbob_test_strict")
    full = _load("bbob_test_full_2d8d")
    stress = _load("bbob_test_stress_32d")

    assert tuple(train["task_ids"]) == expected_bbob_task_ids(BBOB_TRAIN_FUNCTIONS, (2, 4), 0)
    assert tuple(validation["task_ids"]) == expected_bbob_task_ids(BBOB_VALIDATION_FUNCTIONS, (4, 8), 1)
    assert tuple(strict["task_ids"]) == expected_bbob_task_ids(BBOB_STRICT_TEST_FUNCTIONS, (2, 8, 16), 2)
    assert tuple(full["task_ids"]) == expected_bbob_task_ids(tuple(range(1, 25)), (2, 8), 2)
    assert tuple(stress["task_ids"]) == expected_bbob_task_ids(BBOB_STRESS_TEST_FUNCTIONS, (32,), 2)

    assert train["inner_seeds"] == [None]
    assert tuple(validation["inner_seeds"]) == frozen_inner_seeds(VALIDATION_INNER_MASTER_SEED, 4)
    for manifest in (strict, full, stress):
        assert tuple(manifest["inner_seeds"]) == frozen_inner_seeds(TEST_INNER_MASTER_SEED, 5)

    assert strict["primary_ood"] is True
    assert strict["contains_seen_function_identities"] is False
    assert full["primary_ood"] is False
    assert full["contains_seen_function_identities"] is True


def test_all_bbob_entries_are_native_carps_tasks() -> None:
    native_configs = discover_native_bbob_configs()

    assert tuple(sorted({dimension for dimension, _, _ in native_configs})) == EXPECTED_NATIVE_BBOB_DIMENSIONS
    assert len(native_configs) == 360
    for dimension in EXPECTED_NATIVE_BBOB_DIMENSIONS:
        assert len([key for key in native_configs if key[0] == dimension]) == 72
    for name in BBOB_MANIFEST_NAMES:
        validate_native_bbob_manifest(_load(name))


def test_native_bbob_validation_rejects_non_native_dimensions(tmp_path: Path) -> None:
    for dimension in EXPECTED_NATIVE_BBOB_DIMENSIONS:
        (tmp_path / f"cfg_{dimension}_1_0.yaml").touch()
    invalid = copy.deepcopy(_load("bbob_train"))
    invalid["task_ids"] = ["bbob/3/1/0"]
    invalid["manifest_hash"] = manifest_hash(invalid)

    with pytest.raises(ManifestValidationError, match="no native CARP-S YAML"):
        validate_native_bbob_manifest(invalid, tmp_path)


def test_checked_out_carps_dimension_changes_fail_clearly(tmp_path: Path) -> None:
    (tmp_path / "cfg_2_1_0.yaml").touch()

    with pytest.raises(ManifestValidationError, match="dimensions differ"):
        discover_native_bbob_configs(tmp_path)


def test_primary_bbob_splits_and_seed_streams_are_disjoint_as_intended() -> None:
    train = _load("bbob_train")
    validation = _load("bbob_validation")
    strict = _load("bbob_test_strict")
    full = _load("bbob_test_full_2d8d")

    primary_function_sets = [bbob_function_ids(manifest) for manifest in (train, validation, strict)]
    for left, right in itertools.combinations(primary_function_sets, 2):
        assert left.isdisjoint(right)

    assert bbob_function_ids(full).intersection(bbob_function_ids(train))
    assert bbob_function_ids(full).intersection(bbob_function_ids(validation))
    assert fixed_contexts(validation).isdisjoint(fixed_contexts(strict))
    with pytest.raises(ManifestValidationError, match="dynamic seed stream"):
        fixed_contexts(train)


def test_official_yahpo_manifest_has_all_checked_configs_and_references() -> None:
    manifest = _load("yahpo_test_official_so")
    expected_tasks = official_yahpo_so_task_ids()
    configs = discover_official_yahpo_so_configs()
    references = load_legacy_yahpo_references(YAHPO_REFERENCE_CSV)

    assert len(expected_tasks) == 20
    assert tuple(manifest["task_ids"]) == expected_tasks
    assert set(configs) == set(expected_tasks)
    assert set(references) == set(expected_tasks)
    assert all(task["fidelity"] == "fixed_maximum" for task in manifest["tasks"])
    assert all(0.0 <= reference.one_minus_accuracy <= 1.0 for reference in references.values())
    metadata_by_task = {task["task_id"]: task for task in manifest["tasks"]}
    for task_id, config_path in configs.items():
        config = OmegaConf.load(config_path)
        metadata = metadata_by_task[task_id]
        assert config.task.name == task_id
        assert config.task.objective_function.bench == metadata["scenario"]
        assert str(config.task.objective_function.instance) == metadata["instance"]
        assert list(config.task.objective_function.metric) == [metadata["target"]]
        assert config.task.objective_function.budget_type is None
        assert int(config.task.optimization_resources.n_trials) == metadata["budget"]
    validate_official_yahpo_manifest(manifest, reference_csv=YAHPO_REFERENCE_CSV)


def test_yahpo_and_mixed_manifests_are_ready_and_sealed_test_remains_closed() -> None:
    expected_counts = {
        "yahpo_train": 68,
        "yahpo_validation": 24,
        "mixed_train_60_40": 80,
        "mixed_validation": 34,
    }
    for name in YAHPO_AND_MIXED_MANIFEST_NAMES:
        manifest = _load(name)
        assert manifest["status"] == "ready"
        assert manifest["runnable"] is True
        assert len(manifest["task_ids"]) == expected_counts[name]
        require_runnable_manifest(manifest)

    official_test = _load("yahpo_test_official_so")
    assert official_test["status"] == "defined"
    assert official_test["task_ids"]
    with pytest.raises(ManifestUnavailableError, match="sealed"):
        require_runnable_manifest(official_test)


@pytest.mark.parametrize(
    ("master_seed", "count", "message"),
    [
        (-1, 1, "master_seed"),
        (2**32, 1, "master_seed"),
        (True, 1, "master_seed"),
        (0, 0, "count"),
        (0, True, "count"),
    ],
)
def test_frozen_inner_seed_validation(master_seed: int, count: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        frozen_inner_seeds(master_seed, count)
