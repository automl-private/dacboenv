"""Prepare and execute deterministic Stage-B repair and headroom follow-up jobs."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dacboenv.experiment.audit_evaluation_matrix import audit_matrix
from dacboenv.experiment.augment_headroom_with_learned_policies import preflight_run
from dacboenv.experiment.collect_ppo import create_ppo_eval_configs
from dacboenv.experiment.collect_snapshots import configured_structured_action_space
from dacboenv.experiment.evaluation_determinism import (
    PROCESS_DETERMINISM_CONTRACT,
    canonical_sha256,
    file_sha256,
    require_process_determinism,
)
from dacboenv.experiment.evaluation_status import atomic_json
from dacboenv.experiment.import_carps_evaluation_matrix import import_carps_matrix
from dacboenv.experiment.nonfeedback_registry import build_registry, load_registry
from dacboenv.experiment.plan_evaluation_repairs import plan_repairs
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.utils.carps_optimizer import get_task_config

EXPECTED_STAGEB = {
    (domain, family, seed) for domain in ("yahpo", "mixed") for family in ("wei", "af_selection") for seed in (0, 1, 2)
}
SNAPSHOT_PHASES = (0.25, 0.5, 0.75)


@dataclass(frozen=True)
class _CommandArrayJob:
    """Pickle-safe Submitit callable for one manifest row."""

    command: tuple[str, ...]

    def __call__(self, index: int) -> int:
        environment = os.environ.copy()
        environment.update(PROCESS_DETERMINISM_CONTRACT)
        return subprocess.run(  # noqa: S603
            [*self.command, "--job-index", str(index)], env=environment, check=False
        ).returncode


def _training_domain(cfg: Any) -> str:
    declared = str(OmegaConf.select(cfg, "training_instances.domain", default="")).lower()
    if declared in {"yahpo", "mixed"}:
        return declared
    tasks = [str(task) for task in OmegaConf.select(cfg, "dacboenv.task_ids", default=[])]
    domains = {
        "yahpo" if task.startswith("yahpo/") else "bbob" if task.startswith("bbob/") else "unknown" for task in tasks
    }
    if domains == {"yahpo"}:
        return "yahpo"
    if domains == {"yahpo", "bbob"}:
        return "mixed"
    raise ValueError(f"Could not identify Stage-B training domain from tasks: {sorted(domains)!r}.")


def discover_stageb_runs(training_root: Path) -> list[dict[str, Any]]:
    """Discover exactly the predeclared twelve complete final Stage-B runs."""
    inventory = []
    for config_path in sorted(training_root.rglob(".hydra/config.yaml")):
        run_root = config_path.parent.parent
        cfg = OmegaConf.load(config_path)
        try:
            family = configured_structured_action_space(cfg)
            domain = _training_domain(cfg)
        except ValueError:
            continue
        if family not in {"wei", "af_selection"} or domain not in {"yahpo", "mixed"}:
            continue
        selected = preflight_run(run_root, "final", family)
        selected["training_domain"] = domain
        identity = (domain, family, int(selected["outer_ppo_seed"]))
        selected["run_id"] = f"{domain}-{family}-seed{identity[2]}"
        selected["identity"] = list(identity)
        inventory.append(selected)
    identities = [tuple(item["identity"]) for item in inventory]
    if len(set(identities)) != len(identities):
        raise ValueError("Stage-B root contains duplicate domain/action-family/outer-seed runs.")
    missing = sorted(EXPECTED_STAGEB - set(identities))
    unexpected = sorted(set(identities) - EXPECTED_STAGEB)
    if missing or unexpected:
        raise ValueError(f"Stage-B final-run inventory mismatch; missing={missing!r}, unexpected={unexpected!r}.")
    return sorted(inventory, key=lambda row: tuple(row["identity"]))


def _panel(repository: Path, domain: str) -> dict[str, Any]:
    name = "yahpo_frequent.yaml" if domain == "yahpo" else "mixed_frequent.yaml"
    payload = OmegaConf.to_container(
        OmegaConf.load(repository / "dacboenv" / "configs" / "validation_panels" / name), resolve=True
    )
    assert isinstance(payload, dict)
    if payload.get("split") != "validation" or payload.get("panel", {}).get("tier") != "frequent":
        raise ValueError(f"Headroom panel is not a frozen frequent validation panel: {name}")
    return payload


def build_headroom_manifest(
    inventory: list[dict[str, Any]], repository: Path, selector_registry: dict[str, Any]
) -> dict[str, Any]:
    """Create one unit per run/context/phase, with all five actions in-unit."""
    jobs = []
    budget_cache: dict[str, int] = {}
    for run in inventory:
        panel = _panel(repository, run["training_domain"])
        for raw_task_id in panel["task_ids"]:
            task_id = str(raw_task_id)
            if task_id not in budget_cache:
                task_cfg = get_task_config(task_id)
                budget_cache[task_id] = int(task_cfg.task.optimization_resources.n_trials)
            for inner_seed in panel["inner_seeds"]:
                for phase in SNAPSHOT_PHASES:
                    row = {
                        "job_index": len(jobs),
                        "run_id": run["run_id"],
                        "run_root": run["run_root"],
                        "outer_ppo_seed": run["outer_ppo_seed"],
                        "training_domain": run["training_domain"],
                        "action_family": run["action_family"],
                        "interaction_frequency": int(run["interaction_frequency"]),
                        "checkpoint_mode": "final",
                        "model_path": run["model_path"],
                        "model_sha256": run["model_sha256"],
                        "normalization_path": run["normalization_path"],
                        "normalization_sha256": run["normalization_sha256"],
                        "task_id": task_id,
                        "inner_seed": int(inner_seed),
                        "evaluation_budget": budget_cache[task_id],
                        "snapshot_phase": phase,
                        "manifest_id": str(panel["id"]),
                        "manifest_hash": str(panel["manifest_hash"]),
                        "selector_registry_hash": selector_registry["registry_hash"],
                        "branch_actions": [0, 1, 2, 3, 4],
                        "branch_horizons": [1, 5, 10],
                    }
                    row["job_hash"] = canonical_sha256(row)
                    jobs.append(row)
    # One source-history rollout plus five deterministic replay clones.  Each
    # clone replays the prefix and then performs ten new evaluations.  This is
    # a transparent upper estimate of objective calls, not a wall-time claim.
    estimated_calls = sum(round(6 * float(row["snapshot_phase"]) * int(row["evaluation_budget"])) + 50 for row in jobs)
    return {
        "schema_version": "stageb-learned-headroom-jobs-v1",
        "checkpoint_mode": "final",
        "include_control_histories": False,
        "jobs": jobs,
        "job_count": len(jobs),
        "estimated_objective_calls_upper": estimated_calls,
        "manifest_hash": canonical_sha256(jobs),
    }


def _repair_groups(repair_plan: dict[str, Any], inventory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Coalesce repair cells into CARP-S/Hydra sweeps without rerunning successes."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in repair_plan["cells"]:
        method = str(cell["method_id"])
        if method.startswith("ppo_"):
            if not str(cell["task_id"]).startswith("bbob/"):
                raise ValueError(f"Unexpected learned repair outside failed BBOB composition: {cell!r}")
            kind = "learned_bbob"
        elif method == "static-wei_alpha_discrete-action0" and cell["task_id"] == "yahpo/so/rbv2_xgboost/12/None":
            kind = "static_xgboost_alpha0"
        else:
            raise ValueError(f"No scientifically reviewed repair executor for unexpected cell: {cell!r}")
        groups.setdefault((kind, method), []).append(cell)
    result = []
    for (kind, method), cells in sorted(groups.items()):
        group: dict[str, Any] = {
            "repair_kind": kind,
            "method_id": method,
            "task_ids": sorted({str(cell["task_id"]) for cell in cells}),
            "evaluation_seeds": sorted({int(cell["evaluation_seed"]) for cell in cells}),
            "cell_count": len(cells),
        }
        if kind == "learned_bbob":
            candidates = [
                run
                for run in inventory
                if f"seed{run['outer_ppo_seed']}" in method
                and ((run["action_family"] == "wei") == ("AWEI" in method))
                and run["training_domain"] == "yahpo"
            ]
            if len(candidates) != 1:
                raise ValueError(f"Repair method {method!r} does not map to exactly one final model.")
            group["model"] = candidates[0]
        result.append(group)
    return result


def _shell_header(
    repository: Path,
    python: Path,
    followup: Path,
    metadata: dict[str, Any] | None = None,
) -> str:
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in PROCESS_DETERMINISM_CONTRACT.items())
    metadata = metadata or {}
    provenance = "\n".join(
        (
            f"# evaluation_source_revision={metadata.get('evaluation_source_revision', 'not-frozen-test-fixture')}",
            f"# protocol_hash={metadata.get('protocol_hash', 'not-frozen-test-fixture')}",
            f"# model_sha256={','.join(metadata.get('model_hashes', [])) or 'not-frozen-test-fixture'}",
        )
    )
    return f"""#!/usr/bin/env bash
set -euo pipefail
{provenance}
{exports}
repository_root={shlex.quote(str(repository))}
python_bin={shlex.quote(str(python))}
followup_root={shlex.quote(str(followup))}
cd "${{repository_root}}"
"""


def _write_generated_scripts(
    repository: Path,
    python: Path,
    followup: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    header = _shell_header(repository, python, followup, metadata)
    common = """
launcher=local
dry_run=0
partition=normal
timeout_min=360
mem_per_cpu=8G
while (( $# )); do
  case "$1" in
    --launcher) launcher="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --partition) partition="$2"; shift 2 ;;
    --timeout-min) timeout_min="$2"; shift 2 ;;
    --mem-per-cpu) mem_per_cpu="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
"""
    commands = {
        "run_eval_repairs.sh": (
            header
            + common
            + 'command=("${python_bin}" -m dacboenv.experiment.stageb_followup run-repairs --followup-root "${followup_root}" --launcher "${launcher}" --partition "${partition}" --timeout-min "${timeout_min}" --mem-per-cpu "${mem_per_cpu}")\n'
            + 'if [[ "${dry_run}" == 1 ]]; then command+=(--dry-run); fi\nexec "${command[@]}"\n'
        ),
        "run_headroom_augmentation.sh": (
            header
            + common
            + 'command=("${python_bin}" -m dacboenv.experiment.stageb_followup run-headroom --followup-root "${followup_root}" --launcher "${launcher}" --partition "${partition}" --timeout-min "${timeout_min}" --mem-per-cpu "${mem_per_cpu}")\n'
            + 'if [[ "${dry_run}" == 1 ]]; then command+=(--dry-run); fi\nexec "${command[@]}"\n'
        ),
        "merge_eval_repairs.sh": (
            header
            + f'"${{python_bin}}" -m dacboenv.experiment.merge_evaluation_repairs --original-root {shlex.quote(str(followup / "imported_status"))} --repair-root "${{followup_root}}/evaluation_repairs_v1" --expected-protocol "${{followup_root}}/expected_protocol.json" --output "${{followup_root}}/evaluation_protocol_v2_deterministic_complete"\n'
        ),
        "consolidate_headroom.sh": header
        + '"${python_bin}" -m dacboenv.experiment.stageb_followup consolidate --followup-root "${followup_root}"\n',
        "audit_followup.sh": header
        + '"${python_bin}" -m dacboenv.experiment.stageb_followup audit --followup-root "${followup_root}"\n',
    }
    for name, content in commands.items():
        path = followup / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


def prepare(training: Path, evaluation: Path, followup: Path, repository: Path) -> dict[str, Any]:
    """Perform the single user-facing preparation operation."""
    training, evaluation, followup, repository = (
        training.resolve(),
        evaluation.resolve(),
        followup.resolve(),
        repository.resolve(),
    )
    if not training.is_dir() or not evaluation.is_dir():
        raise FileNotFoundError("Training and current evaluation roots must both exist.")
    followup.mkdir(parents=True, exist_ok=True)
    inventory = discover_stageb_runs(training)
    atomic_json(followup / "run_inventory.json", {"runs": inventory, "run_count": len(inventory)})
    atomic_json(followup / "model_inventory.json", {"models": inventory, "model_count": len(inventory)})
    policy_root = followup / "config" / "policy" / "optimized"
    create_ppo_eval_configs(training, policy_root, "final", followup / "ppo_policy_inventory.json")

    expected, import_summary = import_carps_matrix(evaluation, followup, model_inventory=inventory)
    audit = audit_matrix(followup / "imported_status", followup / "expected_protocol.json")
    atomic_json(followup / "evaluation_matrix_audit.json", audit)
    repair_plan = plan_repairs(followup / "imported_status", followup / "expected_protocol.json")
    repair_plan["groups"] = _repair_groups(repair_plan, inventory)
    atomic_json(followup / "evaluation_repairs.json", repair_plan)

    registry_path = followup / "nonfeedback_selector_registry.json"
    source_registry = repository / "artifacts" / "stageb_followup" / "nonfeedback_selector_registry.json"
    if source_registry.is_file():
        registry_path.write_bytes(source_registry.read_bytes())
        registry = load_registry(registry_path)
    else:
        registry = build_registry(repository / "artifacts" / "branch_results.parquet", registry_path)
    headroom = build_headroom_manifest(inventory, repository, registry)
    atomic_json(followup / "headroom_job_manifest.json", headroom)
    reference_table = repository / "dacboenv" / "experiment" / "analysis" / "yahpo_best_known_references.json"
    protocol = {
        "schema_version": "stageb-followup-v1",
        "repository": str(repository),
        # Preserve the repository-local virtual-environment entry point.  Its
        # symlink target is cluster-installation specific and must not leak
        # into generated portable launchers.
        "python": str(repository / ".env" / "bin" / "python"),
        "training_root": str(training),
        "evaluation_root": str(evaluation),
        "followup_root": str(followup),
        "checkpoint_mode": "final",
        "evaluation_source_revision": current_source_revision(repository),
        "training_source_revisions": sorted({str(row["training_source_revision"]) for row in inventory}),
        "determinism_contract": PROCESS_DETERMINISM_CONTRACT,
        "reference_table": str(reference_table.resolve()),
        "reference_table_sha256": file_sha256(reference_table),
        "run_inventory_hash": canonical_sha256(inventory),
        "evaluation_matrix_hash": expected["matrix_hash"],
        "repair_manifest_hash": repair_plan["repair_manifest_hash"],
        "headroom_manifest_hash": headroom["manifest_hash"],
        "selector_registry_hash": registry["registry_hash"],
    }
    protocol["protocol_hash"] = canonical_sha256(protocol)
    atomic_json(followup / "followup_config.json", protocol)
    atomic_json(followup / "protocol_hash.json", {"protocol_hash": protocol["protocol_hash"]})
    python = repository / ".env" / "bin" / "python"
    _write_generated_scripts(
        repository,
        python,
        followup,
        {**protocol, "model_hashes": [str(row["model_sha256"]) for row in inventory]},
    )
    return {
        "status": "prepared",
        "run_count": len(inventory),
        "expected_evaluation_cells": expected["cells"].__len__(),
        "repair_cells": repair_plan["repair_cell_count"],
        "repair_groups": len(repair_plan["groups"]),
        "headroom_jobs": headroom["job_count"],
        "import_summary": import_summary,
        "followup_root": str(followup),
    }


def _run_repairs(args: argparse.Namespace) -> int:
    require_process_determinism()
    root = args.followup_root.resolve()
    plan = json.loads((root / "evaluation_repairs.json").read_text(encoding="utf-8"))
    base_command = [
        str(Path(json.loads((root / "followup_config.json").read_text())["python"])),
        "-m",
        "dacboenv.experiment.run_stageb_repair_group",
        "--manifest",
        str(root / "evaluation_repairs.json"),
        "--output-root",
        str(root / "evaluation_repairs_v1"),
    ]
    print(
        json.dumps(
            {"repair_cells": plan["repair_cell_count"], "launcher": args.launcher, "command": base_command},
            indent=2,
        )
    )
    if args.dry_run:
        return 0
    if args.launcher == "local":
        for index in range(plan["repair_cell_count"]):
            subprocess.run([*base_command, "--job-index", str(index)], check=True)  # noqa: S603
        return 0
    return _submit_array(base_command, plan["repair_cell_count"], root / ".submitit_repairs", args)


def _run_headroom(args: argparse.Namespace) -> int:
    require_process_determinism()
    root = args.followup_root.resolve()
    manifest = json.loads((root / "headroom_job_manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "followup_config.json").read_text(encoding="utf-8"))
    command = [
        protocol["python"],
        "-m",
        "dacboenv.experiment.run_stageb_headroom_job",
        "--manifest",
        str(root / "headroom_job_manifest.json"),
        "--output-root",
        str(root / "headroom_jobs"),
        "--reference-table",
        protocol["reference_table"],
    ]
    print(json.dumps({"job_count": manifest["job_count"], "launcher": args.launcher, "command": command}, indent=2))
    if args.dry_run:
        return 0
    if args.launcher == "local":
        for index in range(manifest["job_count"]):
            subprocess.run([*command, "--job-index", str(index)], check=True)  # noqa: S603
        return 0
    return _submit_array(command, manifest["job_count"], root / ".submitit_headroom", args)


def _submit_array(command: list[str], count: int, folder: Path, args: argparse.Namespace) -> int:
    """Submit one repository-standard Submitit array without per-action jobs."""
    import submitit  # noqa: PLC0415

    executor = submitit.AutoExecutor(folder=str(folder))
    executor.update_parameters(
        slurm_partition=args.partition,
        timeout_min=args.timeout_min,
        slurm_mem_per_cpu=args.mem_per_cpu,
        cpus_per_task=1,
        name="dacbo-stageb-followup",
    )

    jobs = executor.map_array(_CommandArrayJob(tuple(command)), range(count))
    ids = [job.job_id for job in jobs]
    atomic_json(args.followup_root / "submitted_jobs.json", {"launcher": "slurm", "job_ids": ids})
    print(json.dumps({"submitted_jobs": len(ids), "job_ids": ids}, indent=2))
    return 0


def _audit_followup(root: Path) -> int:
    manifest = json.loads((root / "headroom_job_manifest.json").read_text(encoding="utf-8"))
    valid = 0
    for row in manifest["jobs"]:
        path = root / "headroom_jobs" / f"{row['job_index']:05d}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            valid += int(payload.get("job_hash") == row["job_hash"] and payload.get("status") == "success")
    result = {"expected_headroom_jobs": manifest["job_count"], "valid_headroom_jobs": valid}
    atomic_json(root / "followup_audit.json", result)
    print(json.dumps(result, indent=2))
    return 0 if valid == manifest["job_count"] else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch preparation, launch, consolidation, and audit subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("training_root", type=Path)
    prepare_parser.add_argument("evaluation_root", type=Path)
    prepare_parser.add_argument("followup_root", type=Path)
    prepare_parser.add_argument("--repository", type=Path, default=Path.cwd())
    for name in ("run-repairs", "run-headroom"):
        child = sub.add_parser(name)
        child.add_argument("--followup-root", type=Path, required=True)
        child.add_argument("--launcher", choices=("local", "slurm"), default="local")
        child.add_argument("--partition", default="normal")
        child.add_argument("--timeout-min", type=int, default=360)
        child.add_argument("--mem-per-cpu", default="8G")
        child.add_argument("--dry-run", action="store_true")
    audit_parser = sub.add_parser("audit")
    audit_parser.add_argument("--followup-root", type=Path, required=True)
    consolidate_parser = sub.add_parser("consolidate")
    consolidate_parser.add_argument("--followup-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.training_root, args.evaluation_root, args.followup_root, args.repository)
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "run-repairs":
        return _run_repairs(args)
    if args.command == "run-headroom":
        return _run_headroom(args)
    if args.command == "consolidate":
        from dacboenv.experiment.consolidate_stageb_headroom import consolidate  # noqa: PLC0415

        consolidate(args.followup_root)
        return 0
    return _audit_followup(args.followup_root)


if __name__ == "__main__":
    raise SystemExit(main())
