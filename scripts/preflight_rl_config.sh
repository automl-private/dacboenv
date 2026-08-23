#!/usr/bin/env bash
# Safe Hydra preflight for DACBOEnv RL configs.
#
# Hydra's `--cfg job --resolve` is invalid for this repository because the
# composed tree intentionally contains object-valued resolvers such as
# `${get_alpharulenet_configspace:...}`.  This script asks Hydra for the
# unresolved job config, then resolves only scalar/list leaves relevant to the
# requested training run.
#
# Usage:
#   bash scripts/preflight_rl_config.sh TRAINING_CONFIG [HYDRA_OVERRIDE ...]
#
# Example:
#   bash scripts/preflight_rl_config.sh mixed_wei_double_dqn_f5_d1_small

set -euo pipefail

if (( $# < 1 )); then
    echo "Usage: $0 TRAINING_CONFIG [HYDRA_OVERRIDE ...]" >&2
    exit 2
fi

training_config="$1"
shift

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

python_bin=""
for candidate in \
    "${repo_root}/.venv/bin/python" \
    "${repo_root}/.env/bin/python"
do
    if [[ -x "${candidate}" ]]; then
        python_bin="${candidate}"
        break
    fi
done
if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python)"
fi

tmp_cfg="$(mktemp "${TMPDIR:-/tmp}/dacbo-preflight.XXXXXX.yaml")"
trap 'rm -f "${tmp_cfg}"' EXIT

# Deliberately omit --resolve. Global resolution tries to store a
# ConfigSpace.ConfigurationSpace object in an OmegaConf primitive node.
"${python_bin}" -m dacboenv.experiment.rl \
    --cfg job \
    "+training=${training_config}" \
    "$@" > "${tmp_cfg}"

"${python_bin}" - "${tmp_cfg}" "${training_config}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import dacboenv  # noqa: F401  # register OmegaConf resolvers
from omegaconf import DictConfig, ListConfig, OmegaConf

cfg_path = Path(sys.argv[1])
training_config = sys.argv[2]
cfg = OmegaConf.load(cfg_path)
if not isinstance(cfg, DictConfig):
    raise TypeError("Hydra preflight expected a mapping config.")

def select(path: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, path, default=default)
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    return value

summary = {
    "training_config": training_config,
    "rl_algorithm_id": str(select("rl_algorithm_id", "ppo")),
    "optimizer_id": str(select("optimizer_id", "unavailable")),
    "observation_space_id": str(select("observation_space_id", "unavailable")),
    "action_space_id": str(select("action_space_id", "unavailable")),
    "interaction_frequency": int(select("dacboenv.interaction_frequency")),
    "n_workers": int(select("experiment.n_workers")),
    "bo_evaluation_budget": int(select("experiment.bo_evaluation_budget")),
    "total_timesteps": int(select("experiment.total_timesteps")),
    "checkpoint_bo_evaluations": int(select("experiment.checkpoint_bo_evaluations")),
    "checkpoint_freq": int(select("experiment.checkpoint_freq")),
    "validation_enabled": bool(select("experiment.validation.enabled", False)),
    "full_validation_enabled": bool(select("experiment.validation.full_enabled", False)),
    "final_evaluation_enabled": bool(select("experiment.final_evaluation.enabled", False)),
    "training_manifest_id": str(select("training_instances.id", "unavailable")),
    "training_manifest_hash": str(select("training_instances.manifest_hash", "unavailable")),
    "training_task_count": len(select("dacboenv.task_ids", [])),
    "gamma": float(select("rl_algorithm.hyperparameters.gamma", select("optimizer.gamma", 1.0))),
    "buffer_size": int(select("rl_algorithm.hyperparameters.buffer_size", 0)),
    "batch_size": int(select("rl_algorithm.hyperparameters.batch_size", select("optimizer.batch_size", 0))),
    "learning_starts": int(select("rl_algorithm.hyperparameters.learning_starts", 0)),
    "gradient_steps": int(select("rl_algorithm.hyperparameters.gradient_steps", 0)),
    "target_update_interval": int(select("rl_algorithm.hyperparameters.target_update_interval", 0)),
}

errors: list[str] = []
f = summary["interaction_frequency"]
n_workers = summary["n_workers"]
bo_budget = summary["bo_evaluation_budget"]
total = summary["total_timesteps"]

if f <= 0:
    errors.append("interaction_frequency must be positive")
if bo_budget % f:
    errors.append(f"bo_evaluation_budget={bo_budget} is not divisible by interaction_frequency={f}")
if total != bo_budget // f:
    errors.append(
        f"total_timesteps={total} but expected bo_evaluation_budget/frequency={bo_budget // f}"
    )
if total % n_workers:
    errors.append(f"total_timesteps={total} is not divisible by n_workers={n_workers}")
if summary["checkpoint_freq"] != total:
    errors.append(
        "D1 no-validation configs must save exactly one numbered checkpoint at the final timestep"
    )
if summary["validation_enabled"] or summary["full_validation_enabled"]:
    errors.append("D1 config unexpectedly enables validation")
if summary["final_evaluation_enabled"]:
    errors.append("D1 config unexpectedly enables final evaluation")
if summary["rl_algorithm_id"] in {"dqn", "double_dqn"}:
    if summary["learning_starts"] <= 0:
        errors.append("learning_starts must be positive")
    if summary["gradient_steps"] <= 0:
        errors.append("gradient_steps must be positive")
    if summary["target_update_interval"] <= 0:
        errors.append("target_update_interval must be positive")

summary["vector_steps"] = total // n_workers
summary["nominal_optimizer_updates"] = max(
    summary["vector_steps"] - (summary["learning_starts"] // n_workers),
    0,
) * summary["gradient_steps"]
summary["status"] = "pass" if not errors else "fail"
summary["errors"] = errors

print(json.dumps(summary, indent=2, sort_keys=True))
if errors:
    raise SystemExit(1)
PY
