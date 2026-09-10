"""Prepare disjoint Hydra/CARP-S development launchers for selective WEI."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra

from dacboenv.experiment.prepare_offline_carps_evaluation import _task_configs
from dacboenv.selective_wei.artifacts import atomic_json

if TYPE_CHECKING:
    from omegaconf import DictConfig


def _atomic_script(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.chmod(0o750)
    temporary.replace(path)


def prepare(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Generate paired dev commands and never read holdout tasks."""
    inventory_path = Path(str(config.policy_inventory)).resolve()
    output = Path(str(config.output_root)).resolve()
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    policies = inventory["policies"]
    task_sets = {tuple(row["dev_task_ids"]) for row in policies}
    if len(task_sets) != 1:
        raise ValueError("Selective policies disagree on the frozen development task panel.")
    tasks = set(next(iter(task_sets)))
    generated = output / "config"
    groups = _task_configs(tasks, generated)
    repository = Path(__file__).resolve().parents[2]
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
        '    printf "[%s] " "${label}"; printf "%q " "$@"; printf "\\n"; return',
        "  fi",
        '  "$@" &',
        '  launcher_pids+=("$!")',
        '  launcher_labels+=("${label}")',
        "}",
    ]
    labels: list[str] = []

    def add(label: str, group: str, names: list[str], overrides: list[str]) -> None:
        if not names:
            return
        slug = group.lower().replace("/", "-")
        launcher_label = f"{label}--{slug}"
        command = [
            "uv",
            "run",
            "--frozen",
            "python",
            "-m",
            "carps.run",
            "-m",
            f"hydra.searchpath=[file://{generated},pkg://dacboenv.configs]",
            f"+task/{group}={','.join(names)}",
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
            f"baserundir={output}/runs",
            f"hydra.sweep.dir={output}/hydra_sweeps/{label}/{slug}",
            "hydra.sweep.subdir=${hydra.job.num}",
            "hydra.job.chdir=false",
            *overrides,
        ]
        lines.append("launch " + shlex.quote(launcher_label) + " " + " ".join(shlex.quote(item) for item in command))
        labels.append(launcher_label)

    for row in policies:
        policy_id = str(row["policy_id"])
        overrides = (
            [
                "+policy=selective_wei",
                f"policy_id={policy_id}",
                f"optimizer_id={policy_id}",
                f"optimizer.policy_kwargs.policy_bundle={row['policy_bundle']}",
                f"optimizer.policy_kwargs.policy_bundle_hash={row['policy_artifact_hash']}",
                f"optimizer.policy_kwargs.decision_log={output}/runs/decision_logs/{policy_id}/${{task.name}}/${{seed}}.jsonl",
            ]
            if row.get("deployment_kind") != "initial_base_only"
            else [
                "+policy=initial_base",
                f"policy_id={policy_id}",
                f"optimizer_id={policy_id}",
                f"optimizer.policy_kwargs.base_bundle={row['base_selector_registry']}",
            ]
        )
        settings = row.get("initial_feature_settings")
        if settings:
            overrides.extend(
                f"+dacboenv.initial_context_settings.{key}={str(value).lower() if isinstance(value, bool) else value}"
                for key, value in settings.items()
            )
        for group, names in groups.items():
            add(policy_id, group, names, overrides)
        if (
            row.get("base_target_semantics") == "full_static_terminal_loss"
            and row.get("deployment_kind") != "initial_base_only"
        ):
            base_id = f"initial-base-{row['calibration_base_semantic_hash'][:12]}"
            if not any(label.startswith(f"{base_id}--") for label in labels):
                base_overrides = [
                    "+policy=initial_base",
                    f"policy_id={base_id}",
                    f"optimizer_id={base_id}",
                    f"optimizer.policy_kwargs.base_bundle={row['base_selector_registry']}",
                    *(override for override in overrides if override.startswith("+dacboenv.initial_context_settings.")),
                ]
                for group, names in groups.items():
                    add(base_id, group, names, base_overrides)

    first = policies[0]
    comparator_hashes = {
        json.dumps(
            {key: value["registry_hash"] for key, value in row["base_comparator_registries"].items()}, sort_keys=True
        )
        for row in policies
    }
    if len(comparator_hashes) != 1:
        raise ValueError("Selective fits disagree on training-fitted base comparator registries.")
    comparators = first["base_comparator_registries"]
    for base_id in ("B0_global_static", "B1_context_mean", "B2_context_lcb", "B5_context_phase_ablation"):
        if base_id not in comparators:
            continue
        artifact = comparators[base_id]
        overrides = [
            "+policy=selective_base_only",
            f"policy_id={base_id}",
            f"optimizer_id={base_id}",
            f"optimizer.policy_kwargs.registry={artifact['path']}",
            f"optimizer.policy_kwargs.registry_hash={artifact['registry_hash']}",
        ]
        for group, names in groups.items():
            add(base_id, group, names, overrides)

    baselines: list[tuple[str, list[str]]] = [
        (f"static-alpha-{index}", [f"+policy/static/discrete_action=action_{index}"]) for index in range(5)
    ]
    baselines.extend(
        [
            ("uniform-random", ["+policy=random"]),
            ("native-default-smac", ["+policy=defaultaction"]),
            (
                "unconstrained-branch-argmax",
                [
                    "+policy=unconstrained_branch_q",
                    f"optimizer.policy_kwargs.predictor_manifest={first['predictor_manifest']}",
                    "optimizer.policy_kwargs.horizon=5",
                ],
            ),
            (
                "modal-projection",
                [f"+policy/static/discrete_action=action_{first['projection_controls']['modal_action']}"],
            ),
            (
                "marginal-projection",
                [
                    "+policy=marginal_random",
                    "optimizer.policy_kwargs.probabilities="
                    + json.dumps(first["projection_controls"]["marginal_action_probabilities"], separators=(",", ":")),
                ],
            ),
        ]
        if first.get("predictor_manifest")
        else [
            ("uniform-random", ["+policy=random"]),
            ("native-default-smac", ["+policy=defaultaction"]),
        ]
    )
    for label, overrides in baselines:
        for group, names in groups.items():
            add(label, group, names, [*overrides, f"optimizer_id={label}"])
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
    script = output / "run_selective_carps_dev.sh"
    _atomic_script(script, lines)
    result = {
        "schema_version": "dacbo-selective-carps-dev-plan-v1",
        "policy_inventory": str(inventory_path),
        "dev_tasks": sorted(tasks),
        "holdout_accessed": False,
        "scientific_result_root": str(output / "runs"),
        "hydra_sweep_root": str(output / "hydra_sweeps"),
        "launcher": str(script),
        "launcher_labels": labels,
        "unavailable_baselines": {
            "sawei": (
                "The repository SAWEI policy requires UBR observations and a continuous alpha action; "
                "it is not compatible with the fixed Discrete(5), structured-observation comparison."
            )
        },
    }
    atomic_json(output / "selective_eval_plan.json", result)
    return result


@hydra.main(version_base=None, config_path="../configs", config_name="selective_carps_eval")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Prepare one explicit development inventory."""
    print(json.dumps(prepare(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
