"""Export completed initial-base fits to the existing CARP-S dev planner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra

from dacboenv.selective_wei.initial_base_model import load_initial_base
from dacboenv.selective_wei.initial_collection import read_verified, write_verified

if TYPE_CHECKING:
    from omegaconf import DictConfig


def collect(config: DictConfig) -> dict[str, Any]:
    """Verify all candidates and unique semantic IDs before writing inventory."""
    rows = []
    for path in sorted(Path(str(config.run_root)).resolve().rglob("training_complete.json")):
        completion = read_verified(path)
        if completion.get("status") != "complete":
            raise ValueError(f"Incomplete initial-base fit: {path}.")
        bundle_path = path.parent / "base_bundle.json"
        model = load_initial_base(bundle_path)
        if completion["semantic_hash"] != model.semantic_hash:
            raise ValueError("Initial-base completion and predictor disagree.")
        identifier = (
            f"initial-base-{''.join(model.groups)}-{model.metadata['kind']}"
            f"-seed{model.metadata['seed']}-{model.semantic_hash[:12]}"
        )
        rows.append(
            {
                "policy_id": identifier,
                "deployment_kind": "initial_base_only",
                "base_target_semantics": "full_static_terminal_loss",
                "base_selector_registry": str(bundle_path),
                "base_selector_registry_hash": model.semantic_hash,
                "calibration_base_semantic_hash": model.semantic_hash,
                "initial_feature_settings": model.metadata["feature_settings"],
                "base_comparator_registries": {},
                "dev_task_ids": completion["dev_task_ids"],
                "interaction_frequency": 5,
            }
        )
    identities = [row["policy_id"] for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("Initial-base discovery is empty or contains duplicate model identities.")
    payload = {"schema_version": "initial-base-policy-inventory-v1", "policies": rows, "holdout_accessed": False}
    destination = Path(str(config.output_root)).resolve() / "initial_base_policy_inventory.json"
    if destination.exists() and read_verified(destination) != payload:
        raise FileExistsError("Conflicting initial-base policy inventory.")
    write_verified(destination, payload)
    return payload


@hydra.main(version_base=None, config_path="../configs", config_name="initial_base_export")
def main(config: DictConfig) -> None:
    """Export without launching any benchmark evaluation."""
    print(json.dumps(collect(config), indent=2))


if __name__ == "__main__":
    main()
