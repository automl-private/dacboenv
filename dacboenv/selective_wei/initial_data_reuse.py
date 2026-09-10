"""Conservative compatibility audit without inventing D0 from transition data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_collection import read_verified
from dacboenv.selective_wei.initial_features import InitialFeatureSettings


def audit_initial_data_reuse(path: Path) -> dict[str, Any]:
    """Inspect explicit metadata/schema, refusing protected and ambiguous sources.

    Presence of a final GP or compact transition observation never proves that
    an initial fitted-model context can be reconstructed. No reconstruction or
    objective evaluation is performed by this audit.
    """
    result: dict[str, Any] = {
        "source_sha256": file_sha256(path),
        "source_path": str(path.resolve()),
        "audit_version": "initial-base-reuse-v1",
        "training_eligible": False,
        "complete_terminal_labels": False,
        "raw_D0_available": False,
        "compatible_initial_features": False,
        "reconstruction_performed": False,
    }
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as source:
            keys = set(source.files)
            result["array_keys"] = sorted(keys)
            metadata = (
                json.loads(str(source["dataset_metadata_json"].item())) if "dataset_metadata_json" in keys else {}
            )
        split = metadata.get("context_split", metadata.get("split"))
        result["source_split"] = split
        result["reason"] = (
            "protected_or_unproven_split"
            if split not in {"train", "dev"}
            else "transition_archive_requires_verified_raw_D0_and_full_static_history_import"
        )
        return result
    if path.name != "complete.json":
        return {**result, "reason": "unknown_initial_base_source_schema"}
    group = read_verified(path)
    if group.get("recipe", {}).get("split") not in {"train", "dev"}:
        return {**result, "reason": "protected_or_unproven_split"}
    initialization = read_verified(path.parent / "initialization.json")
    if group.get("status") != "complete" or len(group.get("arm_hashes", [])) != 5:  # noqa: PLR2004
        return {**result, "reason": "incomplete_static_group"}
    for action, digest in enumerate(group["arm_hashes"]):
        arm = read_verified(path.parent / f"arm_{action}.json")
        if canonical_hash(arm) != digest or arm.get("status") != "complete":
            raise ValueError("Reuse audit found an incomplete or conflicting continuation arm.")
    result["complete_terminal_labels"] = "terminal_raw" in group.get("targets", {})
    result["raw_D0_available"] = bool(initialization.get("initial_records"))
    result["compatible_initial_features"] = initialization.get("protocol", {}).get("features", {}).get(
        "version"
    ) == InitialFeatureSettings().version and initialization.get("context_digest") == group.get("context_digest")
    result["training_eligible"] = False
    result["reason"] = "schema_compatible_requires_campaign_hash_and_partition_audit_before_import"
    return result
