#!/usr/bin/env bash
# Create algorithm-neutral CARP-S policy configs for the exact final checkpoint
# of every completed PPO/DQN/Double-DQN run below one directory.
#
# Usage:
#   bash scripts/otus/create_final_sb3_eval_configs.sh RUN_ROOT
#   bash scripts/otus/create_final_sb3_eval_configs.sh RUN_ROOT BUNDLE_ROOT
#   bash scripts/otus/create_final_sb3_eval_configs.sh RUN_ROOT BUNDLE_ROOT --overwrite
#
# The default bundle is the sibling path <RUN_ROOT>_carps_final_eval.
# No benchmark evaluation is launched by this script.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  create_final_sb3_eval_configs.sh RUN_ROOT [BUNDLE_ROOT] [--overwrite]

Arguments:
  RUN_ROOT      Directory containing completed Hydra RL runs.
  BUNDLE_ROOT   Output bundle. Default: <RUN_ROOT>_carps_final_eval.

Options:
  --overwrite   Replace only the generated config/inventory tree.

The script selects checkpoint mode `final`. The canonical selector verifies
`training_complete.json`, requires the exact configured final timestep, and
loads the root `model.zip` (plus root `vecnormalize.pkl` when enabled).
Validation history is neither required nor consulted.
EOF
}

if (( $# < 1 || $# > 3 )); then
    usage >&2
    exit 2
fi

run_root_input="$1"
shift

bundle_root_input=""
overwrite=0
while (( $# > 0 )); do
    case "$1" in
        --overwrite)
            overwrite=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "${bundle_root_input}" ]]; then
                echo "Only one BUNDLE_ROOT may be supplied." >&2
                exit 2
            fi
            bundle_root_input="$1"
            ;;
    esac
    shift
done

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if [[ ! -d "${run_root_input}" ]]; then
    echo "RUN_ROOT does not exist: ${run_root_input}" >&2
    exit 1
fi
run_root="$(cd "${run_root_input}" && pwd -P)"

if [[ -n "${bundle_root_input}" ]]; then
    mkdir -p "${bundle_root_input}"
    bundle_root="$(cd "${bundle_root_input}" && pwd -P)"
else
    bundle_root="${run_root%/}_carps_final_eval"
    mkdir -p "${bundle_root}"
    bundle_root="$(cd "${bundle_root}" && pwd -P)"
fi

config_root="${bundle_root}/config"
policy_root="${config_root}/policy/optimized"
inventory_json="${config_root}/policy_inventory.json"
launch_json="${config_root}/policy_launch_inventory.json"
launch_tsv="${config_root}/policy_launch_inventory.tsv"
bundle_json="${bundle_root}/bundle.json"
reference_dir="${config_root}/reference"
reference_source="${repo_root}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
reference_copy="${reference_dir}/yahpo_best_known_references.json"

if [[ -e "${inventory_json}" && "${overwrite}" != "1" ]]; then
    echo "Generated inventory already exists: ${inventory_json}" >&2
    echo "Use --overwrite or choose another BUNDLE_ROOT." >&2
    exit 1
fi

if [[ "${overwrite}" == "1" ]]; then
    rm -rf "${config_root}"
fi
mkdir -p "${policy_root}" "${reference_dir}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required on Otus but was not found in PATH." >&2
    exit 1
fi

# The positional arguments map exactly to:
#   rundir, configs_path, model_selection, inventory_path
#
# `final` is intentionally validation-independent: it is resolved from
# training_complete.json and the root model.zip by checkpoint_selection.py.
uv run --frozen python -m dacboenv.experiment.collect_trained_policies \
    "${run_root}" \
    "${policy_root}" \
    final \
    "${inventory_json}"

if [[ ! -s "${inventory_json}" ]]; then
    echo "Policy collector did not create: ${inventory_json}" >&2
    exit 1
fi

if [[ -s "${reference_source}" ]]; then
    cp -f "${reference_source}" "${reference_copy}"
else
    echo "Warning: YAHPO reference table is absent; BBOB launch remains possible." >&2
    rm -f "${reference_copy}"
fi

uv run --frozen python - \
    "${repo_root}" \
    "${run_root}" \
    "${bundle_root}" \
    "${config_root}" \
    "${inventory_json}" \
    "${launch_json}" \
    "${launch_tsv}" \
    "${bundle_json}" \
    "${reference_copy}" <<'PY'
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

from omegaconf import OmegaConf

(
    repo_root_raw,
    requested_run_root_raw,
    bundle_root_raw,
    config_root_raw,
    inventory_raw,
    launch_json_raw,
    launch_tsv_raw,
    bundle_json_raw,
    reference_raw,
) = sys.argv[1:]

repo_root = Path(repo_root_raw).resolve()
requested_run_root = Path(requested_run_root_raw).resolve()
bundle_root = Path(bundle_root_raw).resolve()
config_root = Path(config_root_raw).resolve()
inventory_path = Path(inventory_raw).resolve()
launch_json_path = Path(launch_json_raw).resolve()
launch_tsv_path = Path(launch_tsv_raw).resolve()
bundle_json_path = Path(bundle_json_raw).resolve()
reference_path = Path(reference_raw).resolve()

observation_configs = {
    "structured": "structured",
    "structured-quantile": "structured_quantile",
    "structured-af-selection": "structured_af_selection",
    "structured-gp-summary-v1": "structured_gp_summary",
    "structured-gp-summary-change-v1": "structured_gp_summary_change",
    "structured-gp-raw64-v1": "structured_gp_raw",
}
action_configs = {
    "wei": "wei_alpha_discrete",
    "af_selection": "af_selection_discrete",
    "lcb_quantile": "lcb_quantile_discrete",
    "ucb_quantile": "ucb_quantile_discrete",
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def find_run_root(model_path: Path) -> Path:
    for parent in model_path.parents:
        if (parent / ".hydra" / "config.yaml").is_file():
            return parent
    raise FileNotFoundError(
        f"Could not find the Hydra run owning model {model_path}"
    )

def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value or "policy"

payload = json.loads(inventory_path.read_text(encoding="utf-8"))
policies = payload.get("policies")
if not isinstance(policies, list) or not policies:
    raise RuntimeError(f"No trained policies were exported from {requested_run_root}")

rows: list[dict[str, object]] = []
seen_slugs: set[str] = set()

for item in policies:
    if str(item.get("checkpoint_mode")) != "final":
        raise RuntimeError(
            f"Expected checkpoint_mode=final, got {item.get('checkpoint_mode')!r}"
        )

    config_path = Path(str(item["config_path"])).resolve()
    model_path = Path(str(item["model_path"])).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    run_root = find_run_root(model_path)
    completion_path = run_root / "training_complete.json"
    if not completion_path.is_file():
        raise FileNotFoundError(
            f"Completed-training marker is missing for {run_root}"
        )
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("complete") is not True:
        raise RuntimeError(f"Run is not marked complete: {run_root}")

    training_step = int(item["training_step"])
    expected_step = int(completion["expected_final_timesteps"])
    completed_step = int(completion["num_timesteps"])
    if training_step != expected_step or completed_step != expected_step:
        raise RuntimeError(
            "Final-step mismatch for "
            f"{run_root}: exported={training_step}, "
            f"completed={completed_step}, expected={expected_step}"
        )

    cfg = OmegaConf.load(config_path)
    bundle = cfg.policy_bundle

    algorithm_id = str(bundle.algorithm_id)
    observation_schema = str(bundle.observation_schema)
    action_family = str(bundle.action_family)
    frequency = int(bundle.interaction_frequency)
    seed = int(item["seed"])

    try:
        observation_config = observation_configs[observation_schema]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported observation schema {observation_schema!r} "
            f"in {config_path}"
        ) from exc
    try:
        action_config = action_configs[action_family]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported action family {action_family!r} in {config_path}"
        ) from exc

    relative_config = config_path.relative_to(config_root)
    policy_group = relative_config.parent.as_posix()
    policy_name = relative_config.stem

    model_sha = str(item["model_sha256"])
    if sha256(model_path) != model_sha:
        raise RuntimeError(f"Model hash changed after export: {model_path}")

    policy_slug = slugify(
        f"{algorithm_id}__{observation_schema}__{action_family}"
        f"__seed{seed}__{model_sha[:10]}"
    )
    if policy_slug in seen_slugs:
        raise RuntimeError(f"Duplicate generated policy slug: {policy_slug}")
    seen_slugs.add(policy_slug)

    rows.append(
        {
            "policy_slug": policy_slug,
            "algorithm_id": algorithm_id,
            "outer_seed": seed,
            "checkpoint_mode": "final",
            "training_step": training_step,
            "action_family": action_family,
            "frequency": frequency,
            "observation_schema": observation_schema,
            "action_config": action_config,
            "observation_config": observation_config,
            "policy_group": policy_group,
            "policy_name": policy_name,
            "config_path": str(config_path),
            "model_path": str(model_path),
            "model_sha256": model_sha,
            "run_root": str(run_root),
            "policy_id": str(item["policy_id"]),
        }
    )

rows.sort(
    key=lambda row: (
        str(row["algorithm_id"]),
        str(row["observation_schema"]),
        int(row["outer_seed"]),
        str(row["model_sha256"]),
    )
)

launch_json_path.write_text(
    json.dumps({"schema_version": 1, "policies": rows}, indent=2, sort_keys=True)
    + "\n",
    encoding="utf-8",
)

fieldnames = [
    "policy_slug",
    "algorithm_id",
    "outer_seed",
    "checkpoint_mode",
    "training_step",
    "action_family",
    "frequency",
    "observation_schema",
    "action_config",
    "observation_config",
    "policy_group",
    "policy_name",
    "config_path",
    "model_path",
    "model_sha256",
    "run_root",
    "policy_id",
]
with launch_tsv_path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows)

reference_payload: dict[str, object] | None = None
if reference_path.is_file():
    reference_payload = {
        "path": str(reference_path),
        "sha256": sha256(reference_path),
    }

bundle_json_path.write_text(
    json.dumps(
        {
            "schema_version": 1,
            "checkpoint_mode": "final",
            "repository_root_at_creation": str(repo_root),
            "requested_run_root": str(requested_run_root),
            "bundle_root": str(bundle_root),
            "config_root": str(config_root),
            "policy_inventory": str(inventory_path),
            "launch_inventory": str(launch_json_path),
            "launch_inventory_tsv": str(launch_tsv_path),
            "n_policies": len(rows),
            "yahpo_reference": reference_payload,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [[ ! -s "${launch_tsv}" || ! -s "${bundle_json}" ]]; then
    echo "Failed to create launch inventory." >&2
    exit 1
fi

n_policies="$(
    uv run --frozen python -c \
        'import json,sys; print(len(json.load(open(sys.argv[1]))["policies"]))' \
        "${launch_json}"
)"

echo
echo "Created final-checkpoint CARP-S evaluation bundle:"
echo "  ${bundle_root}"
echo "Policies exported: ${n_policies}"
echo "Inventory:"
echo "  ${launch_json}"
echo
echo "List policies:"
echo "  bash scripts/otus/otus_eval_final_sb3.sh ${bundle_root} list"
echo
echo "Dry-run BBOB and YAHPO launchers:"
echo "  bash scripts/otus/otus_eval_final_sb3.sh ${bundle_root} both --dry-run"
echo "Dry-run OptBench launcher:"
echo "  bash scripts/otus/otus_eval_final_sb3.sh ${bundle_root} optbench --dry-run"
echo
echo "Submit on Otus:"
echo "  bash scripts/otus/otus_eval_final_sb3.sh ${bundle_root} optbench"
