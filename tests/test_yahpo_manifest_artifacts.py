"""Machine-readable YAHPO protocol inventory and overlap contracts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from dacboenv.experiment.build_manifest_overlap_report import (
    MANIFEST_PATHS,
    build_manifest_overlap_report,
)
from dacboenv.experiment.protocol import (
    discover_official_yahpo_so_configs,
    file_sha256,
    load_manifest,
    official_yahpo_so_task_ids,
)
from dacboenv.experiment.yahpo_protocol import installed_yahpo_inventory

REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_ROOT = REPOSITORY_ROOT / "dacboenv" / "configs" / "instance_sets"
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts"
REFERENCE_TABLE = REPOSITORY_ROOT / "dacboenv" / "experiment" / "analysis" / "yahpo_best_known_references.json"

FROZEN_BBOB_HASHES = {
    "bbob_train": "50dda02f306a2cad36af36dfae05962b6a3c0955d8a1404254896ae155a44152",
    "bbob_validation": "36ed3fb56ddc141069b1efad21f4f2ee51d98fed5a0ebaf8c1cdc0d3fcfec196",
    "bbob_test_strict": "8ba80dd92a2422c2569192abc196513f66aa1ed0d5d248d29deffc0dbf1115ae",
}
READY_YAHPO_MANIFESTS = (
    "yahpo_train",
    "yahpo_validation",
    "mixed_train_60_40",
    "mixed_validation",
)


def _load_manifest(name: str) -> dict:
    return load_manifest(MANIFEST_ROOT / f"{name}.yaml")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_task_hash(task_ids: list[str]) -> str:
    payload = json.dumps(
        task_ids,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_reference_table_and_yahpo_manifests_are_complete() -> None:
    reference_table = _load_json(REFERENCE_TABLE)
    assert reference_table["status"] == "complete"
    assert len(reference_table["references"]) == 608
    assert {row["value"] for row in reference_table["references"]} == {-100.0, -1.0}
    assert all(row["metadata"]["provenance_status"] == "complete" for row in reference_table["references"])

    expected_counts = {
        "yahpo_train": 68,
        "yahpo_validation": 24,
        "mixed_train_60_40": 80,
        "mixed_validation": 34,
    }
    for name in READY_YAHPO_MANIFESTS:
        manifest = _load_manifest(name)
        coverage = manifest["reference_coverage"]

        assert manifest["status"] == "ready"
        assert manifest["runnable"] is True
        assert len(manifest["task_ids"]) == expected_counts[name]
        assert coverage["table_sha256"] == file_sha256(REFERENCE_TABLE)
        assert coverage["provenance_complete_non_test_count"] == 588
        assert coverage["required_total_before_split"] == 92


def test_official_inventory_is_exact_sealed_and_never_evaluated() -> None:
    manifest = _load_manifest("yahpo_test_official_so")
    artifact = _load_json(ARTIFACT_ROOT / "yahpo_official_test_inventory.json")
    expected = list(official_yahpo_so_task_ids())
    inventory = artifact["task_inventory"]

    assert manifest["runnable"] is False
    assert artifact["sealed"] is True
    assert artifact["evaluation_scope"]["post_stage_a_task_objective_evaluations"] == []
    assert artifact["evaluation_scope"]["pre_task_installation_probe"] == {
        "task_ids": ["yahpo/so/lcbench/167168/None"],
        "purpose": "runtime_installation_probe",
        "learned_policy_evaluation": False,
        "used_for_model_or_threshold_selection": False,
    }
    assert manifest["task_ids"] == expected
    assert inventory["task_ids"] == expected
    assert inventory["count"] == 20 == len(set(expected))
    assert set(discover_official_yahpo_so_configs()) == set(expected)

    digest = _ordered_task_hash(expected)
    assert digest == "0cdd972cdec55d2ca07eb99005ed30ed9b6fbcada5af0a2574606348e4d3105b"
    assert inventory["task_id_sha256"] == digest
    assert manifest["inventory_report"]["task_id_sha256"] == digest
    assert artifact["manifest"]["manifest_hash"] == manifest["manifest_hash"]
    assert artifact["installed_benchmark"]["yahpo_gym_source_commit"] == ("93f5b151d4e2f44daa5314cd10533aafec37d630")
    assert manifest["yahpo_protocol"]["yahpo_gym_source_commit"] == ("93f5b151d4e2f44daa5314cd10533aafec37d630")

    scenario_counts = Counter(task_id.split("/")[2] for task_id in expected)
    assert inventory["scenario_counts"] == dict(scenario_counts)
    transfer = artifact["transfer_protocol"]
    assert transfer["held_out_instance_transfer_within_seen_scenarios"]["task_count"] == 19
    assert transfer["unseen_scenario_transfer"] == {
        "scenarios": ["nb301"],
        "task_count": 1,
    }


def test_installed_instance_counts_match_the_versioned_inventory() -> None:
    artifact = _load_json(ARTIFACT_ROOT / "yahpo_official_test_inventory.json")
    recorded = artifact["installed_benchmark"]
    scenarios = tuple(recorded["installed_instance_counts"])
    inventory = installed_yahpo_inventory(scenarios)
    installed_counts = {scenario: len(instances) for scenario, instances in inventory.items()}
    official_counts = artifact["task_inventory"]["scenario_counts"]
    non_test_counts = {
        scenario: installed_counts[scenario] - official_counts[scenario] for scenario in installed_counts
    }

    assert installed_counts == recorded["installed_instance_counts"]
    assert non_test_counts == recorded["installed_non_test_instance_counts"]


def test_overlap_hash_report_matches_live_manifests() -> None:
    report = _load_json(ARTIFACT_ROOT / "manifest_overlap_hash_report.json")
    manifests = {report_name: load_manifest(REPOSITORY_ROOT / path) for report_name, path in MANIFEST_PATHS.items()}

    assert report == build_manifest_overlap_report(REPOSITORY_ROOT)
    assert report["schema_version"] == 2

    for report_name, manifest in manifests.items():
        recorded = report["manifests"][report_name]
        assert recorded["manifest_hash"] == manifest["manifest_hash"]
        assert recorded["task_count"] == len(manifest["task_ids"])
        assert recorded["id"] == manifest["id"]
        assert recorded["status"] == manifest["status"]
        assert recorded["runnable"] is manifest["runnable"]

    for name, expected_hash in FROZEN_BBOB_HASHES.items():
        assert manifests[name]["manifest_hash"] == expected_hash

    for left_name, right_name in combinations(MANIFEST_PATHS, 2):
        overlap_name = f"{left_name}__{right_name}"
        overlap = sorted(set(manifests[left_name]["task_ids"]) & set(manifests[right_name]["task_ids"]))
        assert report["pairwise_overlaps"][overlap_name] == {
            "count": len(overlap),
            "task_ids": overlap,
        }

    panel_names = {
        "bbob_validation_frequent",
        "bbob_validation_full",
        "yahpo_validation_frequent",
        "yahpo_validation_full",
        "mixed_validation_frequent",
        "mixed_validation_full",
    }
    assert panel_names.issubset(report["manifests"])
    assert len(report["pairwise_overlaps"]) == len(MANIFEST_PATHS) * (len(MANIFEST_PATHS) - 1) // 2
    for panel_name in panel_names:
        covered_pairs = [name for name in report["pairwise_overlaps"] if panel_name in name.split("__")]
        assert len(covered_pairs) == len(MANIFEST_PATHS) - 1

    all_bbob = set().union(*(set(manifests[name]["task_ids"]) for name in FROZEN_BBOB_HASHES))
    all_yahpo = set().union(
        *(set(manifests[name]["task_ids"]) for name in ("yahpo_train", "yahpo_validation", "yahpo_test_official_so"))
    )
    assert all_bbob.isdisjoint(all_yahpo)
    assert report["aggregate_overlaps"]["all_bbob__all_yahpo"] == {"count": 0, "task_ids": []}

    reference_coverage = report["yahpo_reference_coverage"]
    assert reference_coverage["table_sha256"] == file_sha256(REFERENCE_TABLE)
    assert reference_coverage["installed_reference_count"] == 608
    assert reference_coverage["reference_basis"] == "assumed_metric_upper_bound"
    assert reference_coverage["provenance_complete_non_test_count"] == 588
    assert reference_coverage["smoke_only_incomplete_count"] == 0
    assert reference_coverage["required_train_count"] == 68
    assert reference_coverage["required_validation_count"] == 24
    assert reference_coverage["required_total_before_split"] == 92
    assert reference_coverage["train_validation_all_or_nothing_blocked"] is False
