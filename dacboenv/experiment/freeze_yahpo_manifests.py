"""Freeze YAHPO and mixed manifests after reference coverage becomes complete."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dacboenv.experiment.protocol import file_sha256, load_manifest, manifest_hash
from dacboenv.experiment.yahpo_protocol import (
    YAHPO_SPLIT_MASTER_SEED,
    deterministic_yahpo_split,
    installed_yahpo_inventory,
    official_yahpo_task_ids,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INSTANCE_ROOT = REPOSITORY_ROOT / "dacboenv/configs/instance_sets"
PANEL_ROOT = REPOSITORY_ROOT / "dacboenv/configs/validation_panels"
REFERENCE_PATH = REPOSITORY_ROOT / "dacboenv/experiment/analysis/yahpo_best_known_references.json"
VALIDATION_SEEDS = [1_349_011_988, 2_024_774_586, 595_161_999, 1_294_824_964]
SCENARIOS = ("lcbench", "rbv2_glmnet", "rbv2_rpart", "rbv2_ranger", "rbv2_xgboost", "rbv2_super")


def _plain(path: Path) -> dict[str, Any]:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(value, dict):
        raise TypeError(f"Expected mapping in {path}.")
    return value


def _write_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    manifest["manifest_hash"] = manifest_hash(manifest)
    OmegaConf.save(OmegaConf.create(manifest), path)
    return manifest


def _scenario_counts(task_ids: list[str]) -> dict[str, int]:
    return dict(Counter(task_id.split("/")[2] for task_id in task_ids))


def _reference_coverage(reference_hash: str, complete_non_test: int, selected: int) -> dict[str, Any]:
    return {
        "table_path": "dacboenv/experiment/analysis/yahpo_best_known_references.json",
        "table_sha256": reference_hash,
        "reference_basis": "assumed_metric_upper_bound",
        "provenance_complete_non_test_count": complete_non_test,
        "required_train_count": 68,
        "required_validation_count": 24,
        "required_total_before_split": 92,
        "selected_task_count": selected,
    }


def _installed_identity() -> dict[str, Any]:
    official = _plain(INSTANCE_ROOT / "yahpo_test_official_so.yaml")
    return dict(official["yahpo_protocol"])


def _yahpo_manifests(
    train_ids: list[str],
    validation_ids: list[str],
    *,
    reference_hash: str,
    complete_non_test: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    installed = _installed_identity()
    train = {
        "schema_version": 1,
        "id": "yahpo-train-v1",
        "domain": "yahpo",
        "split": "train",
        "status": "ready",
        "runnable": True,
        "role": "training",
        "task_ids": train_ids,
        "inner_seeds": [None],
        "reference_coverage": _reference_coverage(reference_hash, complete_non_test, len(train_ids)),
        "installed_benchmark": installed,
        "selection_protocol": {
            "master_seed": YAHPO_SPLIT_MASTER_SEED,
            "generator": "per_scenario_numpy.random.SeedSequence",
            "official_test_instances_excluded": True,
            "sort_before_shuffle": True,
            "scenario_counts": _scenario_counts(train_ids),
            "nb301_instances": 0,
        },
        "training_budget_multipliers": [0.6, 0.8, 1.0],
        "default_training_budget_multiplier": 1.0,
    }
    validation = {
        "schema_version": 1,
        "id": "yahpo-validation-v1",
        "domain": "yahpo",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "role": "checkpoint_selection",
        "task_ids": validation_ids,
        "inner_seeds": VALIDATION_SEEDS,
        "reference_coverage": _reference_coverage(reference_hash, complete_non_test, len(validation_ids)),
        "installed_benchmark": installed,
        "seed_protocol": {
            "kind": "frozen",
            "generator": "numpy.random.SeedSequence.generate_state",
            "master_seed": 3_670_740_482,
            "dtype": "uint32",
            "count": 4,
        },
        "selection_protocol": {
            "master_seed": YAHPO_SPLIT_MASTER_SEED,
            "official_test_instances_excluded": True,
            "scenario_counts": _scenario_counts(validation_ids),
            "full_native_budget": True,
            "nb301_instances": 0,
        },
        "aggregation": {
            "order": ["inner_seed", "dataset_instance", "scenario"],
            "scenario_weights": {scenario: 1 / len(SCENARIOS) for scenario in SCENARIOS},
        },
    }
    return train, validation


def _yahpo_panels(validation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    by_scenario = {
        scenario: [task_id for task_id in validation["task_ids"] if task_id.split("/")[2] == scenario]
        for scenario in SCENARIOS
    }
    frequent_ids = [task_id for scenario in SCENARIOS for task_id in by_scenario[scenario][:2]]
    common = {
        "schema_version": 1,
        "domain": "yahpo",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "scenarios": list(SCENARIOS),
        "source_manifest": {"id": validation["id"], "hash": validation["manifest_hash"]},
        "aggregation": {
            "order": ["inner_seed", "dataset_instance", "scenario"],
            "scenario_weights": {scenario: 1 / len(SCENARIOS) for scenario in SCENARIOS},
        },
    }
    frequent = {
        **common,
        "id": "yahpo-validation-frequent-v1",
        "role": "frequent_checkpoint_screening",
        "task_ids": frequent_ids,
        "inner_seeds": VALIDATION_SEEDS[:2],
        "panel": {
            "tier": "frequent",
            "instances_per_scenario": 2,
            "inner_seeds_per_instance": 2,
            "episode_count": 24,
            "full_native_budget": True,
            "official_test_instances_excluded": True,
            "checkpoint_selection_eligible": False,
            "step_zero_namespace": "step_zero",
        },
    }
    full = {
        **common,
        "id": "yahpo-validation-full-v1",
        "role": "full_checkpoint_selection",
        "task_ids": list(validation["task_ids"]),
        "inner_seeds": VALIDATION_SEEDS,
        "panel": {
            "tier": "full",
            "instances_per_scenario": 4,
            "inner_seeds_per_instance": 4,
            "episode_count": 96,
            "full_native_budget": True,
            "official_test_instances_excluded": True,
            "checkpoint_selection_eligible": True,
            "trained_checkpoints_only": True,
        },
    }
    return frequent, full


def _mixed_manifest(
    *,
    split: str,
    bbob: dict[str, Any],
    yahpo: dict[str, Any],
    reference_hash: str,
    complete_non_test: int,
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "id": f"mixed-{split}-60-40-v1" if split == "train" else "mixed-validation-v1",
        "domain": "mixed",
        "split": split,
        "status": "ready",
        "runnable": True,
        "role": "training" if split == "train" else "checkpoint_selection",
        "task_ids": [*bbob["task_ids"], *yahpo["task_ids"]],
        "inner_seeds": [None] if split == "train" else VALIDATION_SEEDS,
        "reference_coverage": _reference_coverage(reference_hash, complete_non_test, len(yahpo["task_ids"])),
        "composition": {
            "bbob_manifest": bbob["id"],
            "bbob_manifest_hash": bbob["manifest_hash"],
            "yahpo_manifest": yahpo["id"],
            "yahpo_manifest_hash": yahpo["manifest_hash"],
            "balanced_score_weights": {"bbob": 0.5, "yahpo": 0.5},
        },
    }
    if split == "train":
        manifest["composition"].update(
            {
                "target_transition_weights": {"bbob": 0.6, "yahpo": 0.4},
                "preferred_persistent_workers": {"total": 32, "bbob": 20, "yahpo": 12},
                "realized_worker_weights": {"bbob": 0.625, "yahpo": 0.375},
                "bbob_dimension_workers": {"2": 10, "4": 10},
                "yahpo_scenario_workers": dict.fromkeys(SCENARIOS, 2),
            }
        )
    return manifest


def _mixed_panel(bbob: dict[str, Any], yahpo: dict[str, Any], *, tier: str) -> dict[str, Any]:
    episode_count = int(bbob["panel"]["episode_count"]) + int(yahpo["panel"]["episode_count"])
    return {
        "schema_version": 1,
        "id": f"mixed-validation-{tier}-v1",
        "domain": "mixed",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "role": "frequent_checkpoint_screening" if tier == "frequent" else "full_checkpoint_selection",
        "task_ids": [*bbob["task_ids"], *yahpo["task_ids"]],
        "inner_seeds": list(yahpo["inner_seeds"]),
        "panel": {
            "tier": tier,
            "episode_count": episode_count,
            "checkpoint_selection_eligible": tier == "full",
            "trained_checkpoints_only": tier == "full",
        },
        "composition": {
            "bbob": {
                "manifest": bbob["id"],
                "task_ids": list(bbob["task_ids"]),
                "inner_seeds": list(bbob["inner_seeds"]),
                "episode_count": bbob["panel"]["episode_count"],
            },
            "yahpo": {
                "manifest": yahpo["id"],
                "task_ids": list(yahpo["task_ids"]),
                "inner_seeds": list(yahpo["inner_seeds"]),
                "episode_count": yahpo["panel"]["episode_count"],
            },
            "balanced_score_weights": {"bbob": 0.5, "yahpo": 0.5},
        },
    }


def _update_official_manifest(reference_hash: str) -> dict[str, Any]:
    path = INSTANCE_ROOT / "yahpo_test_official_so.yaml"
    manifest = _plain(path)
    legacy_path = REPOSITORY_ROOT / "dacboenv/experiment/analysis/yahpo_so_fmin.csv"
    manifest["reference_source"] = {
        "path": "dacboenv/experiment/analysis/yahpo_so_fmin.csv",
        "sha256": file_sha256(legacy_path),
        "source_method": "assumed_accuracy_upper_bound_v1",
        "kind": "best_known",
        "empirical": False,
        "exactness_proved": False,
    }
    manifest["objective_reference_table"] = {
        "path": "dacboenv/experiment/analysis/yahpo_best_known_references.json",
        "sha256": reference_hash,
    }
    for task in manifest["tasks"]:
        scale = 100.0 if task["scenario"] in {"lcbench", "nb301"} else 1.0
        task["reference_cost"] = 0.0
        task["legacy_minimization_reference"] = -scale
        task["runtime_reference"] = -scale
        task["reference_kind"] = "best_known"
        task["reference_basis"] = "assumed_metric_upper_bound"
    return _write_manifest(path, manifest)


def _update_official_inventory(
    official_manifest: dict[str, Any], reference_hash: str, complete_non_test: int, installed_count: int
) -> None:
    path = REPOSITORY_ROOT / "artifacts/yahpo_official_test_inventory.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["manifest"]["manifest_hash"] = official_manifest["manifest_hash"]
    artifact["reference_status"] = {
        "reference_table_path": "dacboenv/experiment/analysis/yahpo_best_known_references.json",
        "reference_table_sha256": reference_hash,
        "reference_basis": "assumed_metric_upper_bound",
        "kind": "best_known",
        "empirical": False,
        "exactness_proved": False,
        "installed_task_count": installed_count,
        "provenance_complete_non_test_count": complete_non_test,
        "required_non_test_count_before_split": 92,
    }
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    """Freeze ready YAHPO/mixed manifests and their identity artifacts."""
    reference_table = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    complete_ids = {
        row["task_id"]
        for row in reference_table["references"]
        if row["metadata"].get("provenance_status") == "complete"
    }
    sealed = set(official_yahpo_task_ids())
    complete_non_test = len(complete_ids - sealed)
    inventory = installed_yahpo_inventory(SCENARIOS)
    split = deterministic_yahpo_split(inventory, complete_ids)
    train_ids = list(split.train_task_ids)
    validation_ids = list(split.validation_task_ids)
    reference_hash = file_sha256(REFERENCE_PATH)

    train, validation = _yahpo_manifests(
        train_ids,
        validation_ids,
        reference_hash=reference_hash,
        complete_non_test=complete_non_test,
    )
    train = _write_manifest(INSTANCE_ROOT / "yahpo_train.yaml", train)
    validation = _write_manifest(INSTANCE_ROOT / "yahpo_validation.yaml", validation)

    bbob_train = load_manifest(INSTANCE_ROOT / "bbob_train.yaml")
    bbob_validation = load_manifest(INSTANCE_ROOT / "bbob_validation.yaml")
    mixed_train = _mixed_manifest(
        split="train",
        bbob=bbob_train,
        yahpo=train,
        reference_hash=reference_hash,
        complete_non_test=complete_non_test,
    )
    mixed_validation = _mixed_manifest(
        split="validation",
        bbob=bbob_validation,
        yahpo=validation,
        reference_hash=reference_hash,
        complete_non_test=complete_non_test,
    )
    _write_manifest(INSTANCE_ROOT / "mixed_train_60_40.yaml", mixed_train)
    _write_manifest(INSTANCE_ROOT / "mixed_validation.yaml", mixed_validation)

    yahpo_frequent, yahpo_full = _yahpo_panels(validation)
    yahpo_frequent = _write_manifest(PANEL_ROOT / "yahpo_frequent.yaml", yahpo_frequent)
    yahpo_full = _write_manifest(PANEL_ROOT / "yahpo_full.yaml", yahpo_full)
    bbob_frequent = load_manifest(PANEL_ROOT / "bbob_frequent.yaml")
    bbob_full = load_manifest(PANEL_ROOT / "bbob_full.yaml")
    _write_manifest(
        PANEL_ROOT / "mixed_frequent.yaml",
        _mixed_panel(bbob_frequent, yahpo_frequent, tier="frequent"),
    )
    _write_manifest(PANEL_ROOT / "mixed_full.yaml", _mixed_panel(bbob_full, yahpo_full, tier="full"))

    official = _update_official_manifest(reference_hash)
    _update_official_inventory(official, reference_hash, complete_non_test, len(reference_table["references"]))
    print(
        f"Frozen {len(train_ids)} YAHPO train and {len(validation_ids)} validation tasks; "
        f"complete non-test reference coverage is {complete_non_test}."
    )


if __name__ == "__main__":
    main()
