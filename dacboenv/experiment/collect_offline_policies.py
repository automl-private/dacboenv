"""Export explicit offline Q checkpoints as domain-neutral CARP-S policies."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.offline.deployment import deployment_head_for_mode
from dacboenv.offline.identity import offline_policy_id


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"Refusing to replace a different offline policy inventory: {path}.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Write one policy config atomically after inventory-wide collision checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = OmegaConf.to_yaml(OmegaConf.create(payload), sort_keys=False)
    if path.exists():
        if path.read_text(encoding="utf-8") == rendered:
            return
        raise FileExistsError(f"Refusing to overwrite an existing offline policy config: {path}.")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _select(run: Path, mode: str, explicit: str | None) -> Path:
    if mode == "final":
        path = run / "final.pt"
    elif mode == "best_branch_dev":
        path = run / "best_branch_dev.pt"
    elif mode == "explicit":
        if not explicit:
            raise ValueError("checkpoint=explicit requires explicit_checkpoint.")
        path = Path(explicit)
    else:
        raise ValueError("checkpoint must be final, best_branch_dev, or explicit.")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def export(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Discover complete runs and write one YAML bundle per checkpoint."""
    run_root = Path(str(config.run_root)).resolve()
    output = Path(str(config.output_root)).resolve()
    complete_files = sorted(run_root.rglob("training_complete.json"))
    if not complete_files:
        raise FileNotFoundError(f"No complete offline runs under {run_root}.")
    candidates: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for complete_path in complete_files:
        run = complete_path.parent
        completion = json.loads(complete_path.read_text(encoding="utf-8"))
        if completion.get("status") != "complete":
            raise RuntimeError(f"Offline training run is not complete: {run}.")
        checkpoint = _select(run, str(config.checkpoint), str(config.explicit_checkpoint or "") or None)
        expected_hash_key = {
            "final": "final_sha256",
            "best_branch_dev": "best_branch_dev_sha256",
        }.get(str(config.checkpoint))
        if expected_hash_key is not None and completion.get(expected_hash_key) != file_sha256(checkpoint):
            raise ValueError(f"Selected checkpoint does not match training completion metadata: {checkpoint}.")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        resolved = payload["resolved_config"]
        dataset_root = Path(str(resolved["offline_dataset"]["root"])).resolve()
        run_normalizer = run / "normalization_schema.json"
        normalizer = run_normalizer if run_normalizer.is_file() else dataset_root / "normalization_schema.json"
        nonfeedback_registry = run / "training_fitted_nonfeedback_registry.json"
        if file_sha256(normalizer) != payload["provenance"]["normalizer_sha256"]:
            raise ValueError("Offline run normalizer does not match its frozen training provenance.")
        trained_mode = str(resolved["offline_training"].get("algorithm_mode", resolved["offline_algorithm"]["mode"]))
        deployment_head = deployment_head_for_mode(trained_mode)
        selection = payload.get("deployment_selection")
        if not isinstance(selection, dict):
            raise ValueError(f"Checkpoint lacks deployment-selection metadata: {checkpoint}.")
        if selection.get("deployment_head") != deployment_head:
            raise ValueError("Exported deployment head differs from checkpoint-selection head.")
        if selection.get("checkpoint_selection_metric") != "dev/deployment_selected_value":
            raise ValueError("Offline checkpoint was not selected by the canonical deployment metric.")
        if not bool(selection.get("deployment_selection_eligible", False)):
            raise ValueError("Offline checkpoint has no eligible deployment-head development selection.")
        if str(config.checkpoint) == "best_branch_dev" and int(selection.get("selected_update", -1)) != int(
            payload["update"]
        ):
            raise ValueError("best_branch_dev payload update differs from its selected deployment update.")
        coefficient = float(
            resolved["offline_training"].get("cql_coefficient", resolved["offline_algorithm"]["cql_coefficient"])
        )
        model_sha256 = file_sha256(checkpoint)
        selected_update = int(payload["update"])
        policy_id = offline_policy_id(
            experiment_id=str(resolved["offline_training"]["experiment_id"]),
            algorithm_mode=trained_mode,
            cql_coefficient=coefficient,
            training_seed=int(resolved["seed"]),
            selected_update=selected_update,
            checkpoint_mode=str(config.checkpoint),
            model_sha256=model_sha256,
        )
        bundle = {
            "schema_version": "dacbo-offline-carps-policy-v2",
            "policy_id": policy_id,
            "algorithm_id": "offline_shared_q",
            "architecture": payload["model_config"],
            "model_checkpoint": str(checkpoint),
            "model_sha256": model_sha256,
            "normalizer": str(normalizer),
            "normalizer_sha256": file_sha256(normalizer),
            "nonfeedback_registry": str(nonfeedback_registry.resolve()),
            "nonfeedback_registry_sha256": file_sha256(nonfeedback_registry),
            "action_family": "wei",
            "action_space": {"type": "Discrete", "n": 5, "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0]},
            "observation_keys": ["global_state", "action_features"],
            "interaction_frequency": 5,
            "training_seed": int(resolved["seed"]),
            "experiment_id": str(resolved["offline_training"]["experiment_id"]),
            "algorithm_mode": trained_mode,
            "cql_coefficient": coefficient,
            "selected_update": selected_update,
            "checkpoint_mode": str(config.checkpoint),
            "dataset_manifest_hash": payload["provenance"]["dataset_manifest_hash"],
            "final_dataset_root": str(dataset_root),
            "training_config_hash": payload["provenance"]["resolved_config_hash"],
            "deployment_head": deployment_head,
            "checkpoint_selection_head": str(selection["checkpoint_selection_head"]),
            "checkpoint_selection_metric": str(selection["checkpoint_selection_metric"]),
            "checkpoint_selection_value": float(selection["checkpoint_selection_value"]),
        }
        bundle["bundle_hash"] = canonical_sha256(bundle)
        yaml = {
            "policy_id": policy_id,
            "optimizer_id": policy_id,
            "offline_policy_bundle": bundle,
            "optimizer": {
                "policy_class": {
                    "_target_": "dacboenv.policy.offline_q.OfflineQPolicy",
                    "_partial_": True,
                },
                "policy_kwargs": {
                    "checkpoint": str(checkpoint),
                    "normalizer": str(normalizer),
                    "checkpoint_sha256": bundle["model_sha256"],
                    "normalizer_sha256": bundle["normalizer_sha256"],
                    "deployment_head": deployment_head,
                    "interaction_frequency": 5,
                },
            },
        }
        destination = output / "policies" / f"{policy_id}.yaml"
        candidates.append(({**bundle, "policy_config": str(destination)}, yaml, destination))
    policy_ids = [str(bundle["policy_id"]) for bundle, _yaml, _destination in candidates]
    destinations = [destination for _bundle, _yaml, destination in candidates]
    if len(policy_ids) != len(set(policy_ids)):
        raise ValueError("Offline policy discovery produced duplicate collision-proof policy IDs.")
    if len(destinations) != len(set(destinations)):
        raise ValueError("Offline policy discovery produced duplicate YAML destinations.")
    for _bundle, yaml, destination in candidates:
        _atomic_yaml(destination, yaml)
    inventory = [bundle for bundle, _yaml, _destination in candidates]
    result = {
        "schema_version": "dacbo-offline-policy-inventory-v2",
        "checkpoint_mode": str(config.checkpoint),
        "policies": inventory,
    }
    result["inventory_hash"] = canonical_sha256(result)
    _atomic_json(output / "offline_policy_inventory.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs/offline_policy_export", config_name="base")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Export all completed runs beneath an explicit root."""
    print(json.dumps(export(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
