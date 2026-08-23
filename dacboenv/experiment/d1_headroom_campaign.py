"""Prepare and audit learned-policy H=5/H=10 headroom jobs for D1 f5 runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dacboenv.experiment.collect_snapshots import configured_structured_action_space
from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.evaluation_status import atomic_json
from dacboenv.experiment.nonfeedback_registry import load_registry
from dacboenv.experiment.sb3_algorithms import ALGORITHM_REGISTRY, resolve_rl_algorithm_id

SNAPSHOT_PHASES = (0.25, 0.50, 0.75)
BRANCH_HORIZONS = (5, 10)


def _training_domain(cfg: Any) -> str:
    declared = str(OmegaConf.select(cfg, "training_instances.domain", default="")).lower()
    if declared in {"yahpo", "mixed"}:
        return declared
    tasks = [str(value) for value in OmegaConf.select(cfg, "dacboenv.task_ids", default=[])]
    domains = {"yahpo" if task.startswith("yahpo/") else "bbob" if task.startswith("bbob/") else "unknown" for task in tasks}
    if domains == {"yahpo"}:
        return "yahpo"
    if domains == {"yahpo", "bbob"}:
        return "mixed"
    raise ValueError(f"Cannot determine training domain from {sorted(domains)}.")


def _gamma(cfg: Any, algorithm_id: str) -> float:
    path = "optimizer.gamma" if algorithm_id == "ppo" else "rl_algorithm.hyperparameters.gamma"
    value = OmegaConf.select(cfg, path, default=None)
    if value is None:
        raise ValueError(f"Missing gamma at {path}.")
    return float(value)


def _final_artifacts(run_root: Path, cfg: Any) -> tuple[Path, Path | None, int]:
    completion_path = run_root / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"Missing training completion marker: {completion_path}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError(f"Run is not complete: {run_root}")
    expected = int(OmegaConf.select(cfg, "experiment.total_timesteps"))
    if int(completion["num_timesteps"]) != expected or int(completion["expected_final_timesteps"]) != expected:
        raise ValueError(f"Final timestep mismatch in {run_root}.")
    recorded_model = Path(str(completion.get("model_path") or "model.zip"))
    model = recorded_model if recorded_model.is_absolute() else run_root / recorded_model
    if not model.is_file():
        # Training archives are often moved between clusters. Rebase a stale
        # absolute path to the canonical artifact inside the discovered run.
        model = run_root / recorded_model.name
    model = model.resolve()
    if not model.is_file():
        raise FileNotFoundError(model)
    normalization: Path | None = None
    if bool(OmegaConf.select(cfg, "experiment.vecnormalize", default=False)):
        raw = completion.get("normalization_path") or run_root / "vecnormalize.pkl"
        recorded_normalization = Path(str(raw))
        normalization = (
            recorded_normalization
            if recorded_normalization.is_absolute()
            else run_root / recorded_normalization
        )
        if not normalization.is_file():
            normalization = run_root / recorded_normalization.name
        normalization = normalization.resolve()
        if not normalization.is_file():
            raise FileNotFoundError(normalization)
    return model, normalization, expected


def discover_runs(training_roots: list[Path], *, include_small: bool) -> list[dict[str, Any]]:
    inventory = []
    seen: set[tuple[str, int, str]] = set()
    for training_root in training_roots:
        for config_path in sorted(training_root.resolve().rglob(".hydra/config.yaml")):
            run_root = config_path.parent.parent
            cfg = OmegaConf.load(config_path)
            try:
                family = configured_structured_action_space(cfg)
                domain = _training_domain(cfg)
                algorithm_id = resolve_rl_algorithm_id(cfg)
            except (ValueError, KeyError):
                continue
            if family != "wei" or domain not in {"yahpo", "mixed"}:
                continue
            if int(OmegaConf.select(cfg, "dacboenv.interaction_frequency", default=1)) != 5:
                continue
            optimizer_id = str(OmegaConf.select(cfg, "optimizer_id", default=run_root.name))
            if not include_small and "small" in optimizer_id.lower():
                continue
            try:
                model, normalization, training_step = _final_artifacts(run_root, cfg)
            except (FileNotFoundError, RuntimeError, ValueError):
                # Incomplete D1 runs are reported as skipped rather than treated
                # as scientific models.
                continue
            gamma = _gamma(cfg, algorithm_id)
            if gamma != 1.0:
                raise ValueError(f"D1 headroom requires gamma=1, got {gamma} in {run_root}.")
            reward_keys = {str(value) for value in OmegaConf.select(cfg, "dacboenv.reward_keys", default=[])}
            if not reward_keys.intersection({"reference_regret_improvement", "true_regret_improvement"}):
                raise ValueError(f"Unsupported reward in {run_root}: {sorted(reward_keys)}")
            seed = int(OmegaConf.select(cfg, "seed"))
            identity = (domain, seed, file_sha256(model))
            if identity in seen:
                raise ValueError(f"Duplicate D1 model identity: {identity}")
            seen.add(identity)
            spec = ALGORITHM_REGISTRY[algorithm_id]
            protocol_path = run_root / "protocol_metadata.json"
            inventory.append(
                {
                    "run_id": f"{domain}-{algorithm_id}-f5-seed{seed}-{identity[2][:10]}",
                    "run_root": str(run_root.resolve()),
                    "optimizer_id": optimizer_id,
                    "training_domain": domain,
                    "action_family": family,
                    "algorithm_id": algorithm_id,
                    "algorithm_class": spec.algorithm_class,
                    "outer_seed": seed,
                    # Compatibility name used by the older bootstrap code.
                    "outer_ppo_seed": seed,
                    "interaction_frequency": 5,
                    "checkpoint_mode": "final",
                    "training_step": training_step,
                    "model_path": str(model),
                    "model_sha256": identity[2],
                    "normalization_path": None if normalization is None else str(normalization),
                    "normalization_sha256": None if normalization is None else file_sha256(normalization),
                    "config_path": str(config_path.resolve()),
                    "config_sha256": file_sha256(config_path),
                    "observation_schema": str(OmegaConf.select(cfg, "observation_space_id", default="structured")),
                    "gamma": gamma,
                    "reward_keys": sorted(reward_keys),
                    "protocol_metadata_path": str(protocol_path.resolve()) if protocol_path.is_file() else None,
                    "protocol_metadata_sha256": file_sha256(protocol_path) if protocol_path.is_file() else None,
                }
            )
    if not inventory:
        raise RuntimeError("No complete final WEI f5 learned policies were found.")
    return sorted(inventory, key=lambda row: (row["training_domain"], row["outer_seed"], row["model_sha256"]))


def _panel(repository: Path, domain: str) -> dict[str, Any]:
    filename = "yahpo_frequent.yaml" if domain == "yahpo" else "mixed_frequent.yaml"
    cfg = OmegaConf.load(repository / "dacboenv/configs/validation_panels" / filename)
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(filename)
    if payload.get("split") != "validation" or payload.get("panel", {}).get("tier") != "frequent":
        raise ValueError(f"Not a frozen non-test frequent panel: {filename}")
    return payload


def prepare(
    repository: Path,
    training_roots: list[Path],
    output_root: Path,
    selector_registry: Path,
    reference_table: Path,
    *,
    include_small: bool,
) -> dict[str, Any]:
    repository = repository.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = discover_runs(training_roots, include_small=include_small)
    registry = load_registry(selector_registry.resolve())
    copied_registry = output_root / "nonfeedback_selector_registry.json"
    shutil.copy2(selector_registry.resolve(), copied_registry)
    copied_reference = output_root / "yahpo_best_known_references.json"
    shutil.copy2(reference_table.resolve(), copied_reference)

    jobs = []
    for run in inventory:
        panel = _panel(repository, str(run["training_domain"]))
        for task_id in panel["task_ids"]:
            for inner_seed in panel["inner_seeds"]:
                for phase in SNAPSHOT_PHASES:
                    row = {
                        "job_index": len(jobs),
                        **run,
                        "task_id": str(task_id),
                        "inner_seed": int(inner_seed),
                        "snapshot_phase": float(phase),
                        "manifest_id": str(panel["id"]),
                        "manifest_hash": str(panel["manifest_hash"]),
                        "selector_registry_hash": str(registry["registry_hash"]),
                        "branch_actions": [0, 1, 2, 3, 4],
                        "branch_horizons": list(BRANCH_HORIZONS),
                    }
                    row["job_hash"] = canonical_sha256(row)
                    jobs.append(row)
    manifest = {
        "schema_version": "d1-f5-learned-headroom-v1",
        "checkpoint_mode": "final",
        "snapshot_phases": list(SNAPSHOT_PHASES),
        "branch_horizons": list(BRANCH_HORIZONS),
        "include_small": include_small,
        "runs": inventory,
        "run_count": len(inventory),
        "jobs": jobs,
        "job_count": len(jobs),
        "selector_registry_hash": registry["registry_hash"],
        "manifest_hash": canonical_sha256(jobs),
    }
    atomic_json(output_root / "run_inventory.json", {"runs": inventory, "run_count": len(inventory)})
    atomic_json(output_root / "d1_headroom_job_manifest.json", manifest)
    protocol = {
        "schema_version": "d1-f5-learned-headroom-protocol-v1",
        "repository": str(repository),
        "training_roots": [str(path.resolve()) for path in training_roots],
        "output_root": str(output_root),
        "reference_table": str(copied_reference),
        "reference_table_sha256": file_sha256(copied_reference),
        "selector_registry": str(copied_registry),
        "selector_registry_hash": registry["registry_hash"],
        "job_manifest_hash": manifest["manifest_hash"],
        "horizons": list(BRANCH_HORIZONS),
        "context_split": "validation",
        "sealed_test_contexts_used": False,
    }
    protocol["protocol_hash"] = canonical_sha256(protocol)
    atomic_json(output_root / "protocol.json", protocol)
    return {
        "status": "prepared",
        "run_count": len(inventory),
        "job_count": len(jobs),
        "runs_by_domain": {
            domain: sum(row["training_domain"] == domain for row in inventory) for domain in ("mixed", "yahpo")
        },
        "output_root": str(output_root),
    }


def audit(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    manifest = json.loads((output_root / "d1_headroom_job_manifest.json").read_text(encoding="utf-8"))
    counts = {"success": 0, "failed": 0, "missing": 0, "corrupt": 0}
    details = []
    for row in manifest["jobs"]:
        path = output_root / "jobs" / f"{row['job_index']:05d}.json"
        state = "missing"
        message = ""
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") != "success" or payload.get("job_hash") != row["job_hash"]:
                    raise ValueError("status/job hash mismatch")
                state = "success"
            except Exception as error:  # noqa: BLE001
                state = "corrupt"
                message = f"{type(error).__name__}: {error}"
        else:
            failure = output_root / "jobs" / f"{row['job_index']:05d}.failed.json"
            if failure.is_file():
                state = "failed"
                message = str(json.loads(failure.read_text(encoding="utf-8")).get("exception_message", ""))
        counts[state] += 1
        if state != "success":
            details.append({"job_index": row["job_index"], "state": state, "message": message})
    result = {"expected": manifest["job_count"], "counts": counts, "complete": counts["success"] == manifest["job_count"], "non_success": details}
    atomic_json(output_root / "headroom_status.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--repository", type=Path, default=Path.cwd())
    prepare_parser.add_argument("--training-root", action="append", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--selector-registry", type=Path, required=True)
    prepare_parser.add_argument("--reference-table", type=Path, required=True)
    prepare_parser.add_argument("--include-small", action="store_true")
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.repository,
            args.training_root,
            args.output_root,
            args.selector_registry,
            args.reference_table,
            include_small=args.include_small,
        )
    else:
        result = audit(args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
