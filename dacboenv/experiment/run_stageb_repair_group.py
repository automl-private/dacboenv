"""Execute one failed/missing Stage-B broad-evaluation cell through CARP-S."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dacboenv.experiment.evaluation_determinism import canonical_sha256, require_process_determinism
from dacboenv.experiment.evaluation_status import atomic_json, episode_status, evaluation_cell_hash


def _scalar_cost(value: Any) -> float:
    if isinstance(value, list):
        if len(value) != 1:
            raise ValueError(f"Expected one scalar CARP-S cost, got {value!r}.")
        value = value[0]
    return float(value)


def _repair_initial_design_fingerprint(run_directory: Path, initial_design_size: int) -> str:
    """Hash the exact CARP-S configuration/cost prefix emitted by FileLogger."""
    trial_log = run_directory / "trial_logs.jsonl"
    if not trial_log.is_file():
        raise FileNotFoundError(f"Successful CARP-S repair did not write {trial_log}.")
    records = [json.loads(line) for line in trial_log.read_text(encoding="utf-8").splitlines() if line]
    if len(records) < initial_design_size:
        raise RuntimeError(
            f"Repair emitted only {len(records)} trials; cannot verify {initial_design_size}-point initial design."
        )
    prefix = [
        {
            "config": record["trial_info"]["config"],
            "cost": _scalar_cost(record["trial_value"]["cost"]),
        }
        for record in records[:initial_design_size]
    ]
    return canonical_sha256(prefix)


def _task_override(task_id: str) -> str:
    parts = task_id.split("/")
    if parts[0] == "bbob" and len(parts) == 4:  # noqa: PLR2004
        return f"+task/BBOB=cfg_{parts[1]}_{parts[2]}_{parts[3]}"
    if parts[:2] == ["yahpo", "so"] and len(parts) == 5:  # noqa: PLR2004
        return f"+task/YAHPO/SO=cfg_{parts[2]}_{parts[3]}"
    raise ValueError(f"Unsupported repair task ID: {task_id!r}")


def _learned_policy_override(method: str, inventory: dict[str, Any]) -> tuple[str, str, str, int]:
    candidates = [
        item
        for item in inventory["policies"]
        if f"seed{item['seed']}" in method
        and ((item["action_family"] == "wei") == ("AWEI" in method))
        and (("mixed" in str(item["task_id"]).lower()) == ("Imixed" in method))
    ]
    if len(candidates) != 1:
        raise ValueError(f"Repair method {method!r} does not map to one generated domain-neutral policy.")
    item = candidates[0]
    action = "wei_alpha_discrete" if item["action_family"] == "wei" else "af_selection_discrete"
    observation = "structured" if item["action_family"] == "wei" else "structured_af_selection"
    policy_path = Path(item["config_path"])
    # .../policy/optimized/GROUP/TASK/seedN.yaml
    group = f"{policy_path.parents[1].name}/{policy_path.parent.name}"
    return f"+policy/optimized/{group}={policy_path.stem}", action, observation, int(item["frequency"])


def command_for_cell(cell: dict[str, Any], followup_root: Path, output_root: Path) -> list[str]:
    """Build one explicit single-context CARP-S command."""
    config = json.loads((followup_root / "followup_config.json").read_text(encoding="utf-8"))
    repository = Path(config["repository"])
    method = str(cell["method_id"])
    if method.startswith("ppo_"):
        inventory = json.loads((followup_root / "ppo_policy_inventory.json").read_text(encoding="utf-8"))
        policy, action, observation, frequency = _learned_policy_override(method, inventory)
    elif method == "static-wei_alpha_discrete-action0":
        policy, action, observation, frequency = (
            "+policy/static/wei_discrete=level_000",
            "wei_alpha_discrete",
            "structured",
            1,
        )
    else:
        raise ValueError(f"No approved Stage-B repair executor for method {method!r}.")
    result = output_root / "runs" / evaluation_cell_hash(cell)
    return [
        config["python"],
        "-m",
        "carps.run",
        _task_override(str(cell["task_id"])),
        "+eval=base",
        "+env=base",
        "+env/opt=base",
        f"+env/action={action}",
        f"+env/interaction_freq=f{frequency}",
        f"+env/obs={observation}",
        "+env/reward=reference_regret_improvement",
        "+env/reference_provider=composite",
        policy,
        f"seed={int(cell['evaluation_seed'])}",
        f"baserundir={result}",
        f"hydra.run.dir={result}",
        "dacboenv.evaluation_mode=true",
        "dacboenv.context_split=validation",
        "dacboenv.terminate_after_reference_performance_reached=false",
        "+cluster=cpu_noctua",
        f"hydra.searchpath=[file://{followup_root / 'config'},file://{repository / 'dacboenv/configs'},pkg://carps/configs]",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one indexed CARP-S repair cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--job-index", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    require_process_determinism()
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    cell = payload["cells"][args.job_index]
    followup = args.manifest.resolve().parent
    command = command_for_cell(cell, followup, args.output_root.resolve())
    if args.dry_run:
        print(json.dumps({"cell": cell, "command": command}, indent=2))
        return 0
    cell_hash = evaluation_cell_hash(cell)
    carps_run_directory = args.output_root.resolve() / "runs" / cell_hash
    directory = args.output_root.resolve() / "cells" / cell_hash
    result_path = directory / "repair_result.json"
    status_path = directory / "episode.status.json"
    context_hash = canonical_sha256(
        {"task_id": cell["task_id"], "inner_seed": cell["evaluation_seed"], "budget": cell["evaluation_budget"]}
    )
    with episode_status(status_path, cell=cell, context_hash=context_hash, result_path=result_path) as status:
        environment = os.environ.copy()
        protocol = json.loads((followup / "followup_config.json").read_text(encoding="utf-8"))
        environment["DACBO_YAHPO_REFERENCE_TABLE"] = protocol["reference_table"]
        completed = subprocess.run(  # noqa: S603
            command, cwd=protocol["repository"], env=environment, check=False
        )
        if completed.returncode != 0:
            raise RuntimeError(f"CARP-S repair cell exited with status {completed.returncode}.")
        initial_hash = _repair_initial_design_fingerprint(
            carps_run_directory,
            int(cell["initial_design_size"]),
        )
        if initial_hash != cell["initial_design_hash_expected"]:
            raise RuntimeError(
                "Repair initial design differs from the paired broad-evaluation context: "
                f"expected {cell['initial_design_hash_expected']}, observed {initial_hash}."
            )
        status["objective_evaluations_completed"] = int(cell["evaluation_budget"])
        status["initial_design_hash"] = initial_hash
        atomic_json(
            result_path,
            {
                "status": "success",
                "cell": cell,
                "command": command,
                "carps_result_root": str(carps_run_directory),
                "initial_design_hash": initial_hash,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
