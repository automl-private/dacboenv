"""Hashed policy-bundle and atomic artifact helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, cast

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256

POLICY_SCHEMA_VERSION = "dacbo-selective-wei-policy-v2"
INTERACTION_FREQUENCY = 5


def atomic_json(path: Path, payload: Any) -> None:
    """Write JSON atomically in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finalize_policy_bundle(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Validate required deployment identity, hash it, and save atomically."""
    required = {
        "base_selector_registry",
        "base_selector_registry_sha256",
        "predictor_manifest",
        "predictor_manifest_sha256",
        "predictor_model_hashes",
        "normalizer_hash",
        "gate_id",
        "gate_parameters",
        "train_partition_hash",
        "gate_tune_task_hash",
        "uncertainty_calibration_task_hash",
        "feature_schema_hash",
        "action_grid",
        "horizon",
        "interaction_frequency",
        "code_revision",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Selective policy bundle is missing fields: {missing}.")
    if (
        payload["action_grid"] != [0.0, 0.25, 0.5, 0.75, 1.0]
        or payload["interaction_frequency"] != INTERACTION_FREQUENCY
    ):
        raise ValueError("Selective deployment requires the frozen five-action f=5 contract.")
    result = _artifact_paths({"schema_version": POLICY_SCHEMA_VERSION, **payload}, path.parent.resolve(), relative=True)
    result["policy_artifact_hash"] = canonical_sha256(result)
    atomic_json(path, result)
    return result


def load_policy_bundle(path: Path, *, expected_hash: str | None = None) -> dict[str, Any]:
    """Load a bundle and verify its own and referenced artifact hashes."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") not in {POLICY_SCHEMA_VERSION, "dacbo-selective-wei-policy-v1"}:
        raise ValueError("Unsupported selective-policy bundle schema.")
    stored = str(payload.pop("policy_artifact_hash"))
    if canonical_sha256(payload) != stored or (expected_hash is not None and stored != expected_hash):
        raise ValueError("Selective-policy artifact hash mismatch.")
    payload["policy_artifact_hash"] = stored
    if payload["schema_version"] == POLICY_SCHEMA_VERSION:
        payload = _artifact_paths(payload, path.parent.resolve(), relative=False)
    for path_key, hash_key in (
        ("base_selector_registry", "base_selector_registry_sha256"),
        ("predictor_manifest", "predictor_manifest_sha256"),
    ):
        referenced = Path(payload[path_key])
        if not referenced.is_file() or file_sha256(referenced) != payload[hash_key]:
            raise ValueError(f"Selective-policy referenced artifact mismatch: {path_key}.")
    calibration_path = payload.get("uncertainty_calibration_artifact")
    if calibration_path and file_sha256(Path(calibration_path)) != payload.get("uncertainty_calibration_sha256"):
        raise ValueError("Selective uncertainty calibration artifact hash mismatch.")
    return cast("dict[str, Any]", payload)


def _artifact_paths(payload: dict[str, Any], root: Path, *, relative: bool) -> dict[str, Any]:
    """Keep bundle-local paths out of semantic identity while retaining v1 reads."""
    keys = {"path", "base_selector_registry", "predictor_manifest", "uncertainty_calibration_artifact"}
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            result[key] = _artifact_paths(value, root, relative=relative)
        elif key in keys and isinstance(value, str) and value:
            artifact = Path(value)
            if relative:
                result[key] = (
                    str(artifact.relative_to(root))
                    if artifact.is_absolute() and artifact.is_relative_to(root)
                    else value
                )
            else:
                result[key] = str(root / artifact)
        else:
            result[key] = value
    return result


def artifact_schema() -> dict[str, Any]:
    """Return the stable deployment artifact schema."""
    return {
        "schema_version": POLICY_SCHEMA_VERSION,
        "action_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "interaction_frequency": 5,
        "required_hashes": [
            "policy_artifact_hash",
            "base_selector_registry_sha256",
            "predictor_manifest_sha256",
            "predictor_model_hashes",
            "normalizer_hash",
            "feature_schema_hash",
            "train_partition_hash",
            "gate_tune_task_hash",
            "uncertainty_calibration_task_hash",
        ],
    }


__all__ = ["POLICY_SCHEMA_VERSION", "artifact_schema", "atomic_json", "finalize_policy_bundle", "load_policy_bundle"]
