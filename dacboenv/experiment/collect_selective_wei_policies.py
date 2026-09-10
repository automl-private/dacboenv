"""Collect completed selective-WEI bundles into CARP-S policy configs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.selective_wei.artifacts import atomic_json, load_policy_bundle


def collect(config: DictConfig) -> dict[str, Any]:
    """Discover complete runs, reject identity collisions, then write YAMLs."""
    run_root = Path(str(config.run_root)).resolve()
    output = Path(str(config.output_root)).resolve()
    candidates: list[tuple[str, Path, dict[str, Any], Path]] = []
    for complete_path in sorted(run_root.rglob("selective_training_complete.json")):
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        if complete.get("status") != "complete":
            continue
        bundle_path = complete_path.parent / "policy_bundle.json"
        if not bundle_path.is_file():
            bundle_path = Path(complete["policy_bundle"])
        bundle = load_policy_bundle(bundle_path, expected_hash=complete["policy_artifact_hash"])
        model_slug = "-".join(str(value)[:8] for value in bundle["predictor_model_hashes"][:3])
        policy_id = (
            f"{bundle['policy_id']}-{bundle['gate_id'].lower()}-h{bundle['horizon']}-"
            f"m{model_slug}-p{bundle['policy_artifact_hash'][:12]}"
        )
        destination = output / "policies" / f"{policy_id}.yaml"
        candidates.append((policy_id, destination, bundle, bundle_path))
    if not candidates:
        raise FileNotFoundError(f"No completed selective runs beneath {run_root}.")
    ids = [item[0] for item in candidates]
    paths = [item[1] for item in candidates]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Selective policy discovery produced colliding identities.")
    rows = []
    for policy_id, destination, bundle, bundle_path in candidates:
        payload = {
            "policy_id": policy_id,
            "optimizer_id": policy_id,
            "selective_policy_bundle": bundle,
            "optimizer": {
                "policy_class": {
                    "_target_": "dacboenv.selective_wei.policy.SelectiveWEIPolicy",
                    "_partial_": True,
                },
                "policy_kwargs": {
                    "policy_bundle": str(bundle_path),
                    "policy_bundle_hash": bundle["policy_artifact_hash"],
                    "decision_log": None,
                },
            },
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        rendered = OmegaConf.to_yaml(OmegaConf.create(payload), sort_keys=False)
        if destination.exists() and destination.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Refusing conflicting selective policy config {destination}.")
        destination.write_text(rendered, encoding="utf-8")
        rows.append(
            {
                "policy_id": policy_id,
                "policy_config": str(destination),
                "policy_bundle": str(bundle_path),
                "policy_artifact_hash": bundle["policy_artifact_hash"],
                "gate_id": bundle["gate_id"],
                "base_selector_registry": bundle["base_selector_registry"],
                "base_selector_registry_hash": bundle["base_selector_registry_hash"],
                "predictor_model_hashes": bundle["predictor_model_hashes"],
                "predictor_manifest": bundle["predictor_manifest"],
                "base_comparator_registries": bundle["base_comparator_registries"],
                "projection_controls": bundle["projection_controls"],
                "dev_task_ids": bundle["dev_task_ids"],
                "horizon": bundle["horizon"],
                "interaction_frequency": bundle["interaction_frequency"],
                "base_target_semantics": bundle.get("base_target_semantics", "fixed_action_local_gain"),
                "initial_feature_settings": bundle.get("initial_feature_settings"),
                "calibration_base_semantic_hash": bundle.get("calibration_base_semantic_hash"),
            }
        )
    result = {
        "schema_version": "dacbo-selective-policy-inventory-v1",
        "policies": rows,
        "holdout_accessed": False,
    }
    result["inventory_hash"] = canonical_sha256(result)
    atomic_json(output / "selective_policy_inventory.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="selective_policy_export")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Collect one explicit selective run root."""
    print(json.dumps(collect(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
