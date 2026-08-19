#!/usr/bin/env bash
# Launch CARP-S evaluations of exported final PPO/DQN/Double-DQN policies on
# Otus. The Hydra `cpu_noctua` config owns all Slurm resource settings.
#
# Examples:
#   bash scripts/otus/otus_eval_final_sb3.sh BUNDLE_ROOT list
#   bash scripts/otus/otus_eval_final_sb3.sh BUNDLE_ROOT bbob
#   bash scripts/otus/otus_eval_final_sb3.sh BUNDLE_ROOT yahpo
#   bash scripts/otus/otus_eval_final_sb3.sh BUNDLE_ROOT both
#   bash scripts/otus/otus_eval_final_sb3.sh BUNDLE_ROOT gather

set -euo pipefail
set -f

usage() {
    cat <<'EOF'
Usage:
  otus_eval_final_sb3.sh BUNDLE_ROOT MODE [OPTIONS]

Modes:
  list      Show exported final policies.
  bbob     Evaluate all selected policies on BBOB functions 1..24.
  yahpo    Evaluate all selected policies on YAHPO/SO tasks.
  both     Submit BBOB and YAHPO evaluations.
  gather   Gather CARP-S data and report missing runs.

Options:
  --output PATH             Result root. Default: BUNDLE_ROOT/results
  --seeds SPEC              Hydra seed sweep. Default: range(0,6)
  --bbob-dims CSV           Default: 2,8
  --bbob-functions CSV      Default: 1,2,...,24
  --bbob-instance N         Native CARP-S BBOB instance. Default: 0
  --yahpo-tasks SPEC        Default: glob(*)
  --yahpo-reference PATH    Default: bundled reference table
  --policy-filter REGEX     Match generated policy slug. Default: .*
  --dry-run                 Print Hydra commands without submitting.
  --overwrite               Permit populated policy result roots.
  --serial-launchers        Wait for each Hydra launcher before starting next.
  -h, --help                Show this help.

The launcher always composes `+cluster=cpu_noctua`; no Slurm resource flags or
cluster environment variables are required.
EOF
}

if (( $# < 2 )); then
    usage >&2
    exit 2
fi

bundle_root_input="$1"
mode="$2"
shift 2

if [[ ! -d "${bundle_root_input}" ]]; then
    echo "BUNDLE_ROOT does not exist: ${bundle_root_input}" >&2
    exit 1
fi
bundle_root="$(cd "${bundle_root_input}" && pwd -P)"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

bundle_json="${bundle_root}/bundle.json"
config_root="${bundle_root}/config"
launch_tsv="${config_root}/policy_launch_inventory.tsv"
default_reference="${config_root}/reference/yahpo_best_known_references.json"

if [[ ! -s "${bundle_json}" || ! -s "${launch_tsv}" ]]; then
    echo "This is not a complete evaluation bundle: ${bundle_root}" >&2
    echo "Run create_final_sb3_eval_configs.sh first." >&2
    exit 1
fi

output_root="${bundle_root}/results"
seed_spec="range(0,6)"
bbob_dims="2,8"
bbob_functions="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24"
bbob_instance="0"
yahpo_tasks="glob(*)"
yahpo_reference="${default_reference}"
policy_filter=".*"
dry_run=0
overwrite=0
parallel_launchers=1

while (( $# > 0 )); do
    case "$1" in
        --output)
            output_root="$2"
            shift
            ;;
        --seeds)
            seed_spec="$2"
            shift
            ;;
        --bbob-dims)
            bbob_dims="$2"
            shift
            ;;
        --bbob-functions)
            bbob_functions="$2"
            shift
            ;;
        --bbob-instance)
            bbob_instance="$2"
            shift
            ;;
        --yahpo-tasks)
            yahpo_tasks="$2"
            shift
            ;;
        --yahpo-reference)
            yahpo_reference="$2"
            shift
            ;;
        --policy-filter)
            policy_filter="$2"
            shift
            ;;
        --dry-run)
            dry_run=1
            ;;
        --overwrite)
            overwrite=1
            ;;
        --serial-launchers)
            parallel_launchers=0
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
    list|bbob|yahpo|both|gather) ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

case "${bbob_instance}" in
    0|1|2) ;;
    *)
        echo "--bbob-instance must be 0, 1, or 2." >&2
        exit 2
        ;;
esac

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required on Otus but was not found in PATH." >&2
    exit 1
fi

mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd -P)"

export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ "${mode}" == "yahpo" || "${mode}" == "both" ]]; then
    if [[ ! -s "${yahpo_reference}" ]]; then
        echo "YAHPO reference table does not exist: ${yahpo_reference}" >&2
        exit 1
    fi
fi
if [[ -s "${yahpo_reference}" ]]; then
    yahpo_reference="$(cd "$(dirname "${yahpo_reference}")" && pwd -P)/$(basename "${yahpo_reference}")"
    export DACBO_YAHPO_REFERENCE_TABLE="${yahpo_reference}"
fi

if [[ "${mode}" == "list" ]]; then
    if command -v column >/dev/null 2>&1; then
        column -t -s $'\t' "${launch_tsv}"
    else
        cat "${launch_tsv}"
    fi
    exit 0
fi

if [[ "${mode}" == "gather" ]]; then
    uv run --frozen python -m carps.analysis.gather_data "${output_root}"
    uv run --frozen python -m carps.utils.check_missing "${output_root}" || true
    exit 0
fi

build_bbob_tasks() {
    local result=""
    local dim function_id

    IFS=',' read -r -a dims <<< "${bbob_dims}"
    IFS=',' read -r -a functions <<< "${bbob_functions}"

    for dim in "${dims[@]}"; do
        if [[ ! "${dim}" =~ ^(2|4|8|16|32)$ ]]; then
            echo "Unsupported native CARP-S BBOB dimension: ${dim}" >&2
            exit 2
        fi
        for function_id in "${functions[@]}"; do
            if [[ ! "${function_id}" =~ ^([1-9]|1[0-9]|2[0-4])$ ]]; then
                echo "Invalid BBOB function ID: ${function_id}" >&2
                exit 2
            fi
            result+="${result:+,}cfg_${dim}_${function_id}_${bbob_instance}"
        done
    done
    printf '%s' "${result}"
}

bbob_tasks="$(build_bbob_tasks)"

launcher_pids=()
launcher_labels=()
selected_policies=0

launch_policy_suite() {
    local suite_id="$1"
    local task_override="$2"
    local policy_slug="$3"
    local action_config="$4"
    local frequency="$5"
    local observation_config="$6"
    local policy_group="$7"
    local policy_name="$8"
    local algorithm_id="$9"
    local outer_seed="${10}"

    local suite_root="${output_root}/${suite_id}"
    local policy_result_root="${suite_root}/${policy_slug}"

    if [[ "${dry_run}" == "0" && "${overwrite}" == "0" ]] \
        && [[ -d "${policy_result_root}" ]] \
        && [[ -n "$(find "${policy_result_root}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Refusing to overwrite populated result root:" >&2
        echo "  ${policy_result_root}" >&2
        echo "Use --overwrite or choose another --output root." >&2
        exit 1
    fi

    local -a command=(
        uv run --frozen python -m carps.run -m
        "hydra.searchpath=[file://${config_root},pkg://dacboenv/configs]"
        "${task_override}"
        "+eval=base"
        "+env=base"
        "+env/opt=base"
        "+env/action=${action_config}"
        "+env/interaction_freq=f${frequency}"
        "+env/obs=${observation_config}"
        "+env/reward=reference_regret_improvement"
        "+env/reference_provider=composite"
        "+${policy_group}=${policy_name}"
        "+cluster=cpu_noctua"
        "optimizer_id=${policy_slug}"
        "policy_id=${policy_slug}"
        "seed=${seed_spec}"
        "baserundir=${suite_root}"
        "hydra.sweep.dir=${policy_result_root}"
        "hydra.sweep.subdir=\${benchmark_id}/\${task_id}/\${seed}"
        "hydra.sweeper.max_batch_size=500"
        "dacboenv.evaluation_mode=false"
        "dacboenv.terminate_after_reference_performance_reached=false"
        "dacboenv.context_split=test"
        "dacboenv.yahpo_training_budget_multiplier=1.0"
    )

    echo
    echo "Policy: ${policy_slug}"
    echo "  algorithm=${algorithm_id} outer_seed=${outer_seed}"
    echo "  suite=${suite_id} observation=${observation_config}"
    if [[ "${dry_run}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    elif [[ "${parallel_launchers}" == "1" ]]; then
        "${command[@]}" &
        launcher_pids+=("$!")
        launcher_labels+=("${policy_slug}/${suite_id}")
        echo "  Hydra launcher PID: ${launcher_pids[-1]}"
    else
        "${command[@]}"
    fi
}

while IFS=$'\t' read -r \
    policy_slug \
    algorithm_id \
    outer_seed \
    checkpoint_mode \
    training_step \
    action_family \
    frequency \
    observation_schema \
    action_config \
    observation_config \
    policy_group \
    policy_name \
    config_path \
    model_path \
    model_sha256 \
    run_root \
    policy_id
do
    if [[ "${policy_slug}" == "policy_slug" ]]; then
        continue
    fi
    if [[ ! "${policy_slug}" =~ ${policy_filter} ]]; then
        continue
    fi
    if [[ "${checkpoint_mode}" != "final" ]]; then
        echo "Refusing non-final bundle entry: ${policy_slug}" >&2
        exit 1
    fi
    if [[ ! -s "${config_path}" || ! -s "${model_path}" ]]; then
        echo "Missing policy config or model for ${policy_slug}" >&2
        exit 1
    fi

    selected_policies=$((selected_policies + 1))

    if [[ "${mode}" == "bbob" || "${mode}" == "both" ]]; then
        launch_policy_suite \
            bbob \
            "+task/BBOB=${bbob_tasks}" \
            "${policy_slug}" \
            "${action_config}" \
            "${frequency}" \
            "${observation_config}" \
            "${policy_group}" \
            "${policy_name}" \
            "${algorithm_id}" \
            "${outer_seed}"
    fi

    if [[ "${mode}" == "yahpo" || "${mode}" == "both" ]]; then
        launch_policy_suite \
            yahpo \
            "+task/YAHPO/SO=${yahpo_tasks}" \
            "${policy_slug}" \
            "${action_config}" \
            "${frequency}" \
            "${observation_config}" \
            "${policy_group}" \
            "${policy_name}" \
            "${algorithm_id}" \
            "${outer_seed}"
    fi
done < "${launch_tsv}"

if (( selected_policies == 0 )); then
    echo "No policies matched --policy-filter '${policy_filter}'." >&2
    exit 1
fi

launcher_failures=0
for index in "${!launcher_pids[@]}"; do
    pid="${launcher_pids[${index}]}"
    label="${launcher_labels[${index}]}"
    if wait "${pid}"; then
        echo "Hydra launcher completed: ${label}"
    else
        status=$?
        echo "Hydra launcher failed (${status}): ${label}" >&2
        launcher_failures=1
    fi
done

if (( launcher_failures != 0 )); then
    exit 1
fi

echo
if [[ "${dry_run}" == "1" ]]; then
    echo "Dry-run complete."
else
    echo "All Hydra launchers completed or submitted successfully."
    echo "Result root: ${output_root}"
    echo "Gather after Slurm jobs finish:"
    echo "  bash scripts/otus/otus_eval_final_sb3.sh ${bundle_root} gather --output ${output_root}"
fi
