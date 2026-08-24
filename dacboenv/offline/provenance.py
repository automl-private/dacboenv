"""Fail-closed provenance checks for offline training inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.protocol import sealed_final_test_task_ids

FORBIDDEN_ROLE_TOKENS = frozenset({"test", "holdout", "learned_policy_validation_headroom", "sealed"})


def reject_training_provenance(metadata: dict[str, Any]) -> None:
    """Reject test, holdout, and learned-policy validation branch provenance."""
    values = {
        str(metadata.get("data_role", "")).lower(),
        str(metadata.get("context_split", metadata.get("split", ""))).lower(),
        str(metadata.get("campaign_role", "")).lower(),
    }
    if values.intersection(FORBIDDEN_ROLE_TOKENS) or any(
        token in value for token in FORBIDDEN_ROLE_TOKENS for value in values
    ):
        raise ValueError(f"Forbidden offline-training provenance: {sorted(values)}.")
    if bool(metadata.get("final_offline_holdout")):
        raise ValueError("Final offline holdout cannot enter training.")
    task_ids = {str(task) for task in metadata.get("task_ids", [])}
    prohibited = sorted(task_ids.intersection(sealed_final_test_task_ids()))
    if prohibited:
        raise ValueError(f"Offline training provenance contains sealed tasks: {prohibited}.")


def headroom_provenance(root: Path | None) -> dict[str, Any] | None:
    """Record external headroom hashes without reading branch rows."""
    if root is None:
        return None
    resolved = root.resolve()
    candidates = {
        "campaign": resolved / "d1_headroom_job_manifest.json",
        "selector_registry": resolved / "nonfeedback_selector_registry.json",
        "summary": resolved / "learned_headroom_summary.csv",
    }
    result: dict[str, Any] = {
        "path": str(resolved),
        "role": "external_validation_evidence_only",
        "training_allowed": False,
        "files": {},
    }
    for name, path in candidates.items():
        if path.is_file():
            result["files"][name] = {"path": str(path), "sha256": file_sha256(path)}
            if path.suffix == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                for key in ("campaign_hash", "registry_hash", "model_hashes"):
                    if key in payload:
                        result[key] = payload[key]
    return result
