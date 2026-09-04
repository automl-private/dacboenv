#!/usr/bin/env bash
# Deterministic paired static-WEI CARP-S trajectory collection on Otus.
#
# This script launches ordinary CARP-S evaluations. It does NOT use
# OfflineDatasetOptimizer and therefore does not create offline_episode.npz.
#
# Usage:
#   bash eval_all_static_deterministic.sh OUTPUT_ROOT [MODE] [--dry-run]
#
# Modes:
#   all             BBOB d={2,4,8,16}, f=1..24, plus all packaged YAHPO/SO.
#   bbob            BBOB only.
#   yahpo-so        All packaged YAHPO/SO tasks.
#   yahpo-focused   3 LCBench + 4 XGBoost + ranger 16,42 + CARP-S ranger 1040.
#
# Use a fresh OUTPUT_ROOT. Do not merge the corrected rerun with an earlier
# collection whose initial designs were not paired.

set -euo pipefail
set -f

usage() {
    cat <<'EOF'
Usage:
  eval_all_static_deterministic.sh OUTPUT_ROOT [MODE] [--dry-run]

MODE:
  all             BBOB plus all packaged YAHPO/SO tasks (default)
  bbob            BBOB only
  yahpo-so        all packaged YAHPO/SO tasks
  yahpo-focused   requested 10-task YAHPO set

The launcher fixes Python hash randomization and native thread counts before
starting any Python or Submitit process. For one task and CARP-S seed, the five
static policies therefore enter SMAC through the same deterministic process
contract. Pairing must still be verified from the resulting initial-design
configuration hashes before scientific use.
EOF
}

if (( $# < 1 )); then
    usage >&2
    exit 2
fi

output_root_input="$1"
shift
mode="all"
if (( $# > 0 )) && [[ "$1" != -* ]]; then
    mode="$1"
    shift
fi

dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "${mode}" in
    all|bbob|yahpo-so|yahpo-focused) ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but was not found in PATH." >&2
    exit 1
fi

mkdir -p -- "${output_root_input}"
output_root="$(cd "${output_root_input}" && pwd -P)"
result_root="${output_root}/runs"
hydra_root="${output_root}/hydra_sweeps"
mkdir -p "${result_root}" "${hydra_root}"

reference_table="${repo_root}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
if [[ ! -s "${reference_table}" ]]; then
    echo "Missing YAHPO reference table: ${reference_table}" >&2
    exit 1
fi
export DACBO_YAHPO_REFERENCE_TABLE="${reference_table}"

# Source the repository helper when present, then enforce the exact variables
# locally as well. PYTHONHASHSEED must be set before each independent Python
# interpreter starts.
determinism_helper="${repo_root}/scripts/evaluation_determinism_env.sh"
if [[ -f "${determinism_helper}" ]]; then
    # shellcheck source=/dev/null
    source "${determinism_helper}"
fi
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

seed_spec="range(0,10)"
actions=(action_0 action_1 action_2 action_3 action_4)

bbob_tasks=()
for dimension in 2 4 8 16; do
    for function_id in {1..24}; do
        bbob_tasks+=("cfg_${dimension}_${function_id}_0")
    done
done
printf -v bbob_task_csv '%s,' "${bbob_tasks[@]}"
bbob_task_csv="${bbob_task_csv%,}"

focused_yahpo_so=(
    cfg_lcbench_167168
    cfg_lcbench_189873
    cfg_lcbench_189906
    cfg_rbv2_xgboost_12
    cfg_rbv2_xgboost_1501
    cfg_rbv2_xgboost_16
    cfg_rbv2_xgboost_40499
    cfg_rbv2_ranger_16
    cfg_rbv2_ranger_42
)
printf -v focused_yahpo_so_csv '%s,' "${focused_yahpo_so[@]}"
focused_yahpo_so_csv="${focused_yahpo_so_csv%,}"

# Discover the exact task group in the installed CARP-S version rather than
# assuming whether its directory is named BlackBox, blackbox, or otherwise.
discover_ranger1040_group() {
    uv run --frozen python - <<'PY_TASK_GROUP'
from pathlib import Path

import carps

task_root = Path(carps.__file__).resolve().parent / "configs" / "task"
matches = sorted(task_root.rglob("cfg_rbv2_ranger_1040.yaml"))
if len(matches) != 1:
    raise SystemExit(
        "Expected exactly one installed CARP-S cfg_rbv2_ranger_1040.yaml, "
        f"found {len(matches)}: {matches}"
    )
print(matches[0].parent.relative_to(task_root).as_posix())
PY_TASK_GROUP
}

launcher_pids=()
launcher_labels=()

launch_suite() {
    local action="$1"
    local suite_id="$2"
    local task_override="$3"
    local optimizer_id="StaticWEI-${action}"

    local -a command=(
        uv run --frozen python -m carps.run -m
        "hydra.searchpath=[pkg://dacboenv/configs]"
        "${task_override}"
        "+eval=base"
        "+env=base"
        "+env/opt=base"
        "+env/reward=reference_regret_improvement"
        "+env/obs=structured"
        "+env/reference_provider=composite"
        "+env/action=wei_alpha_discrete"
        "+policy/static/discrete_action=${action}"
        "+cluster=cpu_noctua"
        "optimizer_id=${optimizer_id}"
        "policy_id=${optimizer_id}"
        "seed=${seed_spec}"
        "baserundir=${result_root}"
        "hydra.sweeper.max_batch_size=5000"
        "dacboenv.context_split=validation"
        "dacboenv.instance_selector_class._target_=dacboenv.env.instance.RoundRobinInstanceSelector"
        "dacboenv.instance_selector_class._partial_=true"
        "+dacboenv.instance_selector_class.offset=0"
        "dacboenv.evaluation_mode=true"
        "dacboenv.terminate_after_reference_performance_reached=false"
        "dacboenv.yahpo_training_budget_multiplier=1.0"
        "hydra.job.chdir=false"
    )

    echo
    echo "Action=${action}; suite=${suite_id}"
    if (( dry_run == 1 )); then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}" &
        launcher_pids+=("$!")
        launcher_labels+=("${suite_id}/${action}")
    fi
}

ranger1040_group=""
if [[ "${mode}" == yahpo-focused ]]; then
    ranger1040_group="$(discover_ranger1040_group)"
    echo "Installed ranger-1040 task group: ${ranger1040_group}"
fi

for action in "${actions[@]}"; do
    if [[ "${mode}" == all || "${mode}" == bbob ]]; then
        launch_suite \
            "${action}" \
            bbob \
            "+task/BBOB=${bbob_task_csv}"
    fi

    if [[ "${mode}" == all || "${mode}" == yahpo-so ]]; then
        launch_suite \
            "${action}" \
            yahpo_so \
            "+task/YAHPO/SO=glob(*)"
    fi

    if [[ "${mode}" == yahpo-focused ]]; then
        launch_suite \
            "${action}" \
            yahpo_focused_so \
            "+task/YAHPO/SO=${focused_yahpo_so_csv}"

        launch_suite \
            "${action}" \
            yahpo_focused_ranger1040 \
            "+task/${ranger1040_group}=cfg_rbv2_ranger_1040"
    fi
done

if (( dry_run == 1 )); then
    echo
    echo "Dry-run complete; no jobs were submitted."
    exit 0
fi

launcher_failure=0
for index in "${!launcher_pids[@]}"; do
    pid="${launcher_pids[$index]}"
    label="${launcher_labels[$index]}"
    if wait "${pid}"; then
        echo "Hydra launcher completed: ${label}"
    else
        status=$?
        echo "Hydra launcher failed (${status}): ${label}" >&2
        launcher_failure=1
    fi
done

if (( launcher_failure != 0 )); then
    exit 1
fi

cat <<EOF

All Hydra launchers expanded successfully.

Scientific CARP-S output:
  ${result_root}

Hydra/Submitit bookkeeping:
  ${hydra_root}

After all Slurm jobs finish:
  uv run --frozen python -m carps.analysis.gather_data "${result_root}"
  uv run --frozen python -m carps.utils.check_missing "${result_root}" || true

Do not infer pairing merely from equal integer seeds. Before using the data,
verify that the ordered initial-design configuration hashes are identical
across action_0,...,action_4 for every (task, seed).
EOF
