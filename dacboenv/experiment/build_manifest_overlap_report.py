"""Generate the versioned manifest hash and overlap inventory."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

from dacboenv.experiment.protocol import file_sha256, load_manifest, manifest_hash, official_yahpo_so_task_ids

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_TABLE_PATH = Path("dacboenv/experiment/analysis/yahpo_best_known_references.json")
REQUIRED_YAHPO_TRAIN_REFERENCES = 68
REQUIRED_YAHPO_VALIDATION_REFERENCES = 24
REQUIRED_YAHPO_REFERENCES = REQUIRED_YAHPO_TRAIN_REFERENCES + REQUIRED_YAHPO_VALIDATION_REFERENCES
MANIFEST_PATHS = {
    "bbob_train": Path("dacboenv/configs/instance_sets/bbob_train.yaml"),
    "bbob_validation": Path("dacboenv/configs/instance_sets/bbob_validation.yaml"),
    "bbob_test_strict": Path("dacboenv/configs/instance_sets/bbob_test_strict.yaml"),
    "yahpo_train": Path("dacboenv/configs/instance_sets/yahpo_train.yaml"),
    "yahpo_validation": Path("dacboenv/configs/instance_sets/yahpo_validation.yaml"),
    "yahpo_test_official_so": Path("dacboenv/configs/instance_sets/yahpo_test_official_so.yaml"),
    "mixed_train": Path("dacboenv/configs/instance_sets/mixed_train_60_40.yaml"),
    "mixed_validation": Path("dacboenv/configs/instance_sets/mixed_validation.yaml"),
    "bbob_validation_frequent": Path("dacboenv/configs/validation_panels/bbob_frequent.yaml"),
    "bbob_validation_full": Path("dacboenv/configs/validation_panels/bbob_full.yaml"),
    "yahpo_validation_frequent": Path("dacboenv/configs/validation_panels/yahpo_frequent.yaml"),
    "yahpo_validation_full": Path("dacboenv/configs/validation_panels/yahpo_full.yaml"),
    "mixed_validation_frequent": Path("dacboenv/configs/validation_panels/mixed_frequent.yaml"),
    "mixed_validation_full": Path("dacboenv/configs/validation_panels/mixed_full.yaml"),
}


def _overlap(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    task_ids = sorted(set(left["task_ids"]) & set(right["task_ids"]))
    return {"count": len(task_ids), "task_ids": task_ids}


def _manifest_entry(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    live_hash = manifest_hash(manifest)
    if live_hash != manifest["manifest_hash"]:
        raise ValueError(f"Manifest {path} records {manifest['manifest_hash']}, computed {live_hash}.")
    return {
        "path": path.as_posix(),
        "id": manifest["id"],
        "domain": manifest["domain"],
        "split": manifest["split"],
        "status": manifest["status"],
        "runnable": manifest["runnable"],
        "task_count": len(manifest["task_ids"]),
        "manifest_hash": live_hash,
    }


def build_manifest_overlap_report(repository_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Return an exhaustive, deterministic report for instance and panel manifests."""
    manifests = {name: load_manifest(repository_root / path) for name, path in MANIFEST_PATHS.items()}
    pairwise = {
        f"{left_name}__{right_name}": _overlap(manifests[left_name], manifests[right_name])
        for left_name, right_name in combinations(MANIFEST_PATHS, 2)
    }
    bbob_names = ("bbob_train", "bbob_validation", "bbob_test_strict")
    yahpo_names = ("yahpo_train", "yahpo_validation", "yahpo_test_official_so")
    all_bbob = set().union(*(set(manifests[name]["task_ids"]) for name in bbob_names))
    all_yahpo = set().union(*(set(manifests[name]["task_ids"]) for name in yahpo_names))

    reference_path = repository_root / REFERENCE_TABLE_PATH
    reference_table = json.loads(reference_path.read_text(encoding="utf-8"))
    references = reference_table.get("references", [])
    sealed_yahpo = set(official_yahpo_so_task_ids())
    provenance_complete = sum(
        reference.get("metadata", {}).get("provenance_status") == "complete"
        and reference.get("task_id") not in sealed_yahpo
        for reference in references
    )
    smoke_only_incomplete = sum(
        reference.get("metadata", {}).get("provenance_status") == "smoke_only_incomplete" for reference in references
    )
    return {
        "schema_version": 2,
        "id": "frozen-manifest-overlap-hash-report-v2",
        "manifests": {name: _manifest_entry(path, manifests[name]) for name, path in MANIFEST_PATHS.items()},
        "pairwise_overlaps": pairwise,
        "aggregate_overlaps": {
            "all_bbob__all_yahpo": {
                "count": len(all_bbob & all_yahpo),
                "task_ids": sorted(all_bbob & all_yahpo),
            },
            "mixed_train__sealed_tests": _overlap(
                manifests["mixed_train"],
                {
                    "task_ids": (
                        list(manifests["bbob_test_strict"]["task_ids"])
                        + list(manifests["yahpo_test_official_so"]["task_ids"])
                    )
                },
            ),
            "mixed_validation__sealed_tests": _overlap(
                manifests["mixed_validation"],
                {
                    "task_ids": (
                        list(manifests["bbob_test_strict"]["task_ids"])
                        + list(manifests["yahpo_test_official_so"]["task_ids"])
                    )
                },
            ),
        },
        "yahpo_reference_coverage": {
            "table_path": REFERENCE_TABLE_PATH.as_posix(),
            "table_sha256": file_sha256(reference_path),
            "installed_reference_count": len(references),
            "reference_basis": reference_table.get("reference_convention", {}).get("basis"),
            "provenance_complete_non_test_count": provenance_complete,
            "smoke_only_incomplete_count": smoke_only_incomplete,
            "required_train_count": REQUIRED_YAHPO_TRAIN_REFERENCES,
            "required_validation_count": REQUIRED_YAHPO_VALIDATION_REFERENCES,
            "required_total_before_split": REQUIRED_YAHPO_REFERENCES,
            "train_validation_all_or_nothing_blocked": provenance_complete < REQUIRED_YAHPO_REFERENCES,
        },
    }


def main() -> None:
    """Write the report to an explicit path (the repository artifact by default)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "artifacts/manifest_overlap_hash_report.json",
    )
    args = parser.parse_args()
    report = build_manifest_overlap_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
