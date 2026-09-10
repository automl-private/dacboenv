"""Prepare and consolidate base-history or selective-history branch campaigns."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.protocol import sealed_final_test_task_ids
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.offline.schema import MIDRUN_BRANCH_SCHEMA_VERSION, ensure_no_object_arrays, validate_branch_arrays
from dacboenv.selective_wei.artifacts import atomic_json, load_policy_bundle
from dacboenv.selective_wei.initial_base_model import load_initial_base

if TYPE_CHECKING:
    from omegaconf import DictConfig

SELECTIVE_BRANCH_PROTOCOL = "dacbo-selective-history-branch-v1"


def _environment_split(data_split: str) -> str:
    try:
        return {"train": "train", "dev": "validation"}[data_split]
    except KeyError as error:
        raise ValueError("Selective branch campaigns forbid holdout/test tasks.") from error


def prepare(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912 - explicit campaign/provenance checks
    """Freeze one job per task, seed, and phase without opening holdout."""
    final_root = Path(str(config.final_dataset_root)).resolve()
    output = Path(str(config.output_root)).resolve()
    manifest = json.loads((final_root / "final_offline_dataset_manifest.json").read_text(encoding="utf-8"))
    campaign = str(config.campaign)
    if campaign not in {"base", "selective"}:
        raise ValueError("campaign must be base or selective.")
    policy_artifact: dict[str, Any]
    if campaign == "base":
        initial_bundle = config.get("initial_base_bundle")
        registry = Path(str(initial_bundle or config.base_selector_registry)).resolve()
        if not registry.is_file():
            raise FileNotFoundError("Base-history campaign requires an explicit frozen registry.")
        policy_artifact = {
            "base_selector_registry": str(registry),
            "base_selector_registry_hash": "" if initial_bundle else str(config.base_selector_registry_hash),
        }
        if initial_bundle:
            declared_manifest = dict(manifest)
            expected_manifest_hash = declared_manifest.pop("manifest_hash")
            if canonical_sha256(declared_manifest) != expected_manifest_hash:
                raise ValueError("Initial-base branch task manifest hash mismatch.")
            if set(manifest["task_splits"]["train"] + manifest["task_splits"]["dev"]) & sealed_final_test_task_ids():
                raise ValueError("Protected task in initial-base branch campaign.")
            model = load_initial_base(registry)
            policy_artifact.update(
                initial_base_bundle=str(registry),
                initial_base_semantic_hash=model.semantic_hash,
                initial_feature_settings=model.metadata["feature_settings"],
                base_target_semantics="full_static_terminal_loss",
                source_revision=current_source_revision(),
            )
    else:
        bundle_path = Path(str(config.policy_bundle)).resolve()
        bundle = load_policy_bundle(bundle_path)
        policy_artifact = {
            "policy_bundle": str(bundle_path),
            "policy_artifact_hash": bundle["policy_artifact_hash"],
        }
    jobs: list[dict[str, Any]] = []
    target = str(config.get("target_semantics", "fixed_action_local_gain"))
    if target not in {"fixed_action_local_gain", "override_then_base_terminal_advantage"}:
        raise ValueError("Unknown selective branch target semantics.")
    if target == "override_then_base_terminal_advantage" and not policy_artifact.get("initial_base_bundle"):
        raise ValueError("Terminal interventions require an explicit immutable initial base.")
    selected_splits = ("train", "dev") if campaign == "base" else ("train",)
    for split in selected_splits:
        for task in manifest["task_splits"][split]:
            for seed in range(5):
                for phase in (0.25, 0.5, 0.75):
                    scientific = {
                        "protocol": (
                            "dacbo-anchored-selective-branch-v2"
                            if policy_artifact.get("initial_base_bundle")
                            else SELECTIVE_BRANCH_PROTOCOL
                        ),
                        "target_semantics": target,
                        "intervention_horizon": int(config.get("intervention_horizon", 5)),
                        "campaign": campaign,
                        "final_manifest_hash": manifest["manifest_hash"],
                        "data_context_split": split,
                        "environment_context_split": _environment_split(split),
                        "task_id": task,
                        "seed": seed,
                        "phase": phase,
                        "actions": [0, 1, 2, 3, 4],
                        "horizons": [5, 10],
                        "interaction_frequency": 5,
                        **policy_artifact,
                    }
                    index = len(jobs)
                    jobs.append(
                        {
                            "job_index": index,
                            "job_hash": canonical_sha256(scientific),
                            **scientific,
                            "output_path": str(output / "jobs" / f"{index:04d}.json"),
                        }
                    )
    result = {
        "schema_version": SELECTIVE_BRANCH_PROTOCOL,
        "campaign": campaign,
        "final_dataset_root": str(final_root),
        "final_manifest_hash": manifest["manifest_hash"],
        "holdout_accessed": False,
        "jobs": jobs,
        "job_count": len(jobs),
    }
    result["manifest_hash"] = canonical_sha256(result)
    path = output / f"selective_{campaign}_branch_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8"))["manifest_hash"] != result["manifest_hash"]:
        raise RuntimeError(f"Refusing to replace a different selective campaign at {path}.")
    atomic_json(path, result)
    return result


def status(config: DictConfig) -> dict[str, Any]:
    """Audit every expected atomic shard by job identity."""
    output = Path(str(config.output_root)).resolve()
    campaign = str(config.campaign)
    manifest = json.loads((output / f"selective_{campaign}_branch_manifest.json").read_text(encoding="utf-8"))
    counts = {"success": 0, "failed": 0, "missing": 0, "corrupt": 0}
    missing = []
    for row in manifest["jobs"]:
        path = Path(row["output_path"])
        state = "missing"
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if row.get("initial_base_bundle"):
                    digest = payload.pop("payload_hash")
                    if canonical_sha256(payload) != digest:
                        raise ValueError("Anchored branch payload hash mismatch.")
                if payload.get("status") != "success" or payload.get("job_hash") != row["job_hash"]:
                    raise ValueError("identity mismatch")
                state = "success"
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                state = "corrupt"
        elif path.with_suffix(".failed.json").is_file():
            state = "failed"
        counts[state] += 1
        if state != "success":
            missing.append(row["job_index"])
    result = {
        "schema_version": SELECTIVE_BRANCH_PROTOCOL,
        "campaign": campaign,
        "expected": len(manifest["jobs"]),
        "counts": counts,
        "complete": counts["success"] == len(manifest["jobs"]),
        "missing_indices": missing,
        "manifest_hash": manifest["manifest_hash"],
    }
    atomic_json(output / f"selective_{campaign}_branch_status.json", result)
    return result


def consolidate(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912 - validate before atomic consolidation
    """Create new versioned NPZs; never mutate the original branch dataset."""
    audit = status(config)
    if not audit["complete"]:
        raise RuntimeError(f"Selective branch campaign is incomplete: {audit['counts']}.")
    output = Path(str(config.output_root)).resolve()
    campaign = str(config.campaign)
    manifest = json.loads((output / f"selective_{campaign}_branch_manifest.json").read_text(encoding="utf-8"))
    outputs = {}
    targets = {row.get("target_semantics", "fixed_action_local_gain") for row in manifest["jobs"]}
    if len(targets) != 1:
        raise ValueError("Selective consolidation cannot mix target semantics.")
    if targets == {"override_then_base_terminal_advantage"}:
        for split in sorted({row["data_context_split"] for row in manifest["jobs"]}):
            destination = output / f"terminal_interventions_{split}_v1.json"
            rows = [
                json.loads(Path(row["output_path"]).read_text())
                for row in manifest["jobs"]
                if row["data_context_split"] == split
            ]
            atomic_json(
                destination,
                {
                    "target_semantics": "override_then_base_terminal_advantage",
                    "split": split,
                    "manifest_hash": manifest["manifest_hash"],
                    "rows": rows,
                },
            )
            outputs[split] = {"path": str(destination), "sha256": file_sha256(destination), "states": len(rows)}
        return {"target_semantics": "override_then_base_terminal_advantage", "outputs": outputs}
    for split in sorted({row["data_context_split"] for row in manifest["jobs"]}):
        records = [
            json.loads(Path(row["output_path"]).read_text(encoding="utf-8"))["branch_record"]
            for row in manifest["jobs"]
            if row["data_context_split"] == split
        ]
        arrays: dict[str, np.ndarray] = {}
        numeric = (
            "global_state",
            "action_features",
            "q5",
            "q10",
            "valid_action_mask",
            "tie_mask_q5",
            "tie_mask_q10",
            "top1_top2_gap_q5",
            "top1_top2_gap_q10",
            "domain_id",
            "scenario_id",
            "phase_bin",
            "seed",
        )
        for key in numeric:
            arrays[key] = np.asarray([record[key] for record in records])
        for key in ("global_state", "action_features"):
            arrays[key] = arrays[key].astype(np.float32)
        anchored = ["initial_anchor_json" in record for record in records]
        if any(anchored):
            if not all(anchored):
                raise ValueError("Cannot mix anchored and unanchored selective branch rows.")
            for key in ("initial_anchor_json", "initial_base_semantic_hash"):
                arrays[key] = np.asarray([record[key] for record in records], dtype=str)
            arrays["initial_base_action"] = np.asarray(
                [record["initial_base_action"] for record in records], dtype=np.int8
            )
        for key in ("q5", "q10", "top1_top2_gap_q5", "top1_top2_gap_q10"):
            arrays[key] = arrays[key].astype(np.float64)
        for key in ("valid_action_mask", "tie_mask_q5", "tie_mask_q10"):
            arrays[key] = arrays[key].astype(bool)
        for key in ("domain_id", "scenario_id", "phase_bin"):
            arrays[key] = arrays[key].astype(np.int8)
        arrays["seed"] = arrays["seed"].astype(np.int32)
        arrays["action_alpha"] = np.asarray([0, 0.25, 0.5, 0.75, 1], dtype=np.float32)
        for key in (
            "task_id",
            "source_policy_id",
            "source_state_digest",
            "source_replay_digest",
            "candidate_duplicate_groups",
            "branch_protocol_hash",
            "data_context_split",
            "environment_context_split",
        ):
            values = [str(record[key]) for record in records]
            arrays[key] = np.asarray(values, dtype=f"U{max(map(len, values))}")
        references = [
            json.dumps(record["reference_metadata"], sort_keys=True, separators=(",", ":")) for record in records
        ]
        arrays["reference_metadata_json"] = np.asarray(references, dtype=f"U{max(map(len, references))}")
        metadata = {
            "schema_version": MIDRUN_BRANCH_SCHEMA_VERSION,
            "component": "midrun_same_state_q5_q10",
            "split": split,
            "context_split": split,
            "environment_context_split": _environment_split(split),
            "branch_execution_semantics": SELECTIVE_BRANCH_PROTOCOL,
            "data_role": "selective_training_aggregation" if split == "train" else "selective_development",
            "campaign_role": f"{campaign}_history_branches",
            "manifest_hash": manifest["manifest_hash"],
            "original_dataset_modified": False,
            "target_semantics": "fixed_action_local_gain",
            "initial_anchor_required": all(anchored),
        }
        arrays["dataset_metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))
        ensure_no_object_arrays(arrays)
        validate_branch_arrays(arrays)
        destination = output / (
            f"branches_{split}.npz" if all(anchored) else f"selective_{campaign}_branches_{split}_v1.npz"
        )
        temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)  # type: ignore[arg-type]
        temporary.replace(destination)
        outputs[split] = {"path": str(destination), "sha256": file_sha256(destination), "states": len(records)}
    result = {"schema_version": SELECTIVE_BRANCH_PROTOCOL, "campaign": campaign, "outputs": outputs}
    atomic_json(output / f"selective_{campaign}_branch_consolidation.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="selective_branch_campaign")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Dispatch preparation, status, or consolidation."""
    operation = str(config.operation)
    result = {"prepare": prepare, "status": status, "consolidate": consolidate}[operation](config)
    print(json.dumps({key: value for key, value in result.items() if key != "jobs"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
