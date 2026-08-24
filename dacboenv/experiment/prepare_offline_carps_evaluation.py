"""Prepare an exact dev-task CARP-S launcher for exported offline policies."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any

import carps
import hydra
from omegaconf import DictConfig, OmegaConf

from dacboenv.utils.carps_optimizer import get_task_config


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.chmod(0o750)
    temporary.replace(path)


def _task_configs(task_ids: set[str], generated_config_root: Path) -> dict[str, list[str]]:
    """Resolve packaged tasks and materialize installed non-test YAHPO configs."""
    root = Path(carps.__file__).resolve().parent / "configs" / "task"
    result: dict[str, list[str]] = {"BBOB": [], "YAHPO/SO": []}
    found = set()
    for group, names in result.items():
        for path in sorted((root / group).glob("*.yaml")):
            config = OmegaConf.load(path)
            task_id = str(config.task.name)
            if task_id in task_ids:
                names.append(path.stem)
                found.add(task_id)
    missing = sorted(task_ids - found)
    for task_id in missing:
        if not task_id.startswith("yahpo/so/"):
            raise FileNotFoundError(f"CARP-S has no packaged config for offline dev task {task_id!r}.")
        _namespace, _kind, scenario, instance, _budget = task_id.split("/")
        name = f"offline_{scenario}_{instance}"
        destination = generated_config_root / "task" / "YAHPO" / "SO" / f"{name}.yaml"
        destination.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(get_task_config(task_id), destination)
        result["YAHPO/SO"].append(name)
        found.add(task_id)
    if found != task_ids:
        raise RuntimeError("Offline dev task config generation did not cover the frozen task set.")
    return result


def prepare(config: DictConfig) -> dict[str, Any]:
    """Write a self-contained shell launcher without opening holdout tasks."""
    inventory_path = Path(str(config.policy_inventory)).resolve()
    output_root = Path(str(config.output_root)).resolve()
    repository = Path(__file__).resolve().parents[2]
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    roots = {str(row["final_dataset_root"]) for row in inventory["policies"]}
    if len(roots) != 1:
        raise ValueError("All evaluated offline policies must use one finalized dataset root.")
    final_root = Path(roots.pop())
    final_manifest = json.loads((final_root / "final_offline_dataset_manifest.json").read_text(encoding="utf-8"))
    tasks = set(final_manifest["task_splits"]["dev"])
    if tasks.intersection(final_manifest["task_splits"]["holdout"]):
        raise RuntimeError("Offline dev and holdout tasks overlap.")
    generated_config_root = output_root / "config"
    groups = _task_configs(tasks, generated_config_root)
    reference = repository / "dacboenv/experiment/analysis/yahpo_best_known_references.json"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repository))}",
        "export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
        'dry_run="${1:-}"',
        '[[ -z "${dry_run}" || "${dry_run}" == "--dry-run" ]] || { echo "Usage: $0 [--dry-run]" >&2; exit 2; }',
        "declare -a launcher_pids=() launcher_labels=()",
        "launch() {",
        '  local label="$1"; shift',
        '  if [[ "${dry_run}" == "--dry-run" ]]; then',
        '    printf "[%s] " "${label}"',
        "    printf '%q ' \"$@\"",
        "    printf '\\n'",
        "    return",
        "  fi",
        '  "$@" &',
        '  launcher_pids+=("$!")',
        '  launcher_labels+=("${label}")',
        "}",
    ]
    launcher_labels: list[str] = []

    def append_command(label: str, group: str, configs: list[str], policy_overrides: list[str]) -> None:
        if not configs:
            return
        group_slug = group.lower().replace("/", "-")
        launcher_label = f"{label}--{group_slug}"
        command = [
            "uv",
            "run",
            "--frozen",
            "python",
            "-m",
            "carps.run",
            "-m",
            f"hydra.searchpath=[file://{generated_config_root},pkg://dacboenv.configs]",
            f"+task/{group}={','.join(configs)}",
            "+eval=base",
            "+env=base",
            "+env/opt=base",
            "+env/action=wei_alpha_discrete",
            "+env/interaction_freq=f5",
            "+env/obs=structured",
            "+env/reward=reference_regret_improvement",
            "+env/reference_provider=composite",
            "+cluster=cpu_noctua",
            "seed=range(0,5)",
            "dacboenv.context_split=validation",
            "dacboenv.evaluation_mode=true",
            "dacboenv.terminate_after_reference_performance_reached=false",
            f"dacboenv.reference_provider.providers.yahpo.source={reference}",
            f"baserundir={output_root}/runs",
            f"hydra.sweep.dir={output_root}/hydra_sweeps/{label}/{group_slug}",
            "hydra.sweep.subdir=${hydra.job.num}",
            "hydra.job.chdir=false",
            *policy_overrides,
        ]
        lines.append("launch " + shlex.quote(launcher_label) + " " + " ".join(shlex.quote(item) for item in command))
        launcher_labels.append(launcher_label)

    for policy in inventory["policies"]:
        overrides = [
            "+policy=offline_q",
            f"policy_id={policy['policy_id']}",
            f"optimizer_id={policy['policy_id']}",
            f"optimizer.policy_kwargs.checkpoint={policy['model_checkpoint']}",
            f"optimizer.policy_kwargs.normalizer={policy['normalizer']}",
            f"optimizer.policy_kwargs.checkpoint_sha256={policy['model_sha256']}",
            f"optimizer.policy_kwargs.normalizer_sha256={policy['normalizer_sha256']}",
            f"optimizer.policy_kwargs.deployment_head={policy['deployment_head']}",
        ]
        for group, names in groups.items():
            append_command(str(policy["policy_id"]), group, names, overrides)
    registry_hashes = {str(policy["nonfeedback_registry_sha256"]) for policy in inventory["policies"]}
    if len(registry_hashes) != 1:
        raise ValueError("Offline policies disagree on the frozen training-fitted nonfeedback registry.")
    registry = inventory["policies"][0]["nonfeedback_registry"]
    registry_hash = inventory["policies"][0]["nonfeedback_registry_sha256"]
    baselines = [(f"static-alpha-{index}", [f"+policy/static/discrete_action=action_{index}"]) for index in range(5)]
    baselines.extend(
        [
            ("uniform-random", ["+policy=random"]),
            ("native-default-smac", ["+policy=defaultaction"]),
        ]
    )
    baselines.append(
        (
            "training-fitted-context-static",
            [
                "+policy=context_static_offline",
                f"optimizer.policy_kwargs.registry={registry}",
                f"optimizer.policy_kwargs.registry_sha256={registry_hash}",
            ],
        )
    )
    for label, overrides in baselines:
        for group, names in groups.items():
            append_command(label, group, names, [*overrides, f"optimizer_id={label}"])
    lines.extend(
        [
            'if [[ "${dry_run}" == "--dry-run" ]]; then exit 0; fi',
            "launcher_status=0",
            'for index in "${!launcher_pids[@]}"; do',
            '  if ! wait "${launcher_pids[$index]}"; then',
            '    echo "CARP-S launcher failed: ${launcher_labels[$index]}" >&2',
            "    launcher_status=1",
            "  fi",
            "done",
            'exit "${launcher_status}"',
        ]
    )
    path = output_root / "run_carps_dev_evaluation.sh"
    _atomic_text(path, "\n".join(lines) + "\n")
    result = {
        "schema_version": "dacbo-offline-carps-dev-plan-v2",
        "policy_inventory": str(inventory_path),
        "task_split_hash": final_manifest["task_split_hash"],
        "dev_tasks": sorted(tasks),
        "holdout_accessed": False,
        "launcher": str(path),
        "launcher_labels": launcher_labels,
        "scientific_result_root": str(output_root / "runs"),
        "hydra_sweep_root": str(output_root / "hydra_sweeps"),
        "generated_task_config_root": str(generated_config_root),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "offline_eval_plan.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


@hydra.main(version_base=None, config_path="../configs/offline_eval", config_name="base")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Prepare the dev-only evaluation launcher."""
    print(json.dumps(prepare(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
