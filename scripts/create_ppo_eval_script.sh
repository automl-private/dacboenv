#!/usr/bin/env bash

# Discover every completed PPO run below one root and create a concrete CARP-S
# evaluation launcher. Final checkpoints are selected at the exact configured
# terminal timestep; step-zero artifacts are never used.

set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/create_ppo_eval_script.sh PPO_RESULTS_DIR \
    [--output DIR] [--checkpoint MODE] [--explicit-model PATH] \
    [--explicit-normalizer PATH] [--overwrite]

Arguments:
  PPO_RESULTS_DIR   Any directory containing completed Hydra PPO runs. Every
                    descendant complete run is discovered.

Options:
  --checkpoint MODE         final (default), full_best, frequent_best, or
                            explicit. Deprecated aliases: best=full_best,
                            last=final.
  --output DIR              Evaluation bundle root. Default:
                            `<PPO_RESULTS_DIR>_carps_eval`.
  --explicit-model PATH     Required with --checkpoint explicit.
  --explicit-normalizer PATH  Matching optional normalization state.
  --overwrite               Replace an existing generated run script.

Deprecated environment compatibility:
  DACBO_POLICY_MODEL, DACBO_EVAL_RESULTS_DIR, DACBO_EVAL_RUN_SCRIPT,
  DACBO_OVERWRITE_RUN_SCRIPT. Command-line arguments take precedence.
  DACBO_PYTHON              Python executable. Default: .venv/bin/python when
                            present, otherwise python from PATH.

The generated script accepts `training`, `bbob_2d_8d`, or `both` (default),
plus `ppo` (default), `baselines`, `all`, or a comma-separated method subset.
Its header documents the independent CARP-S seed, BBOB instance, Slurm, and
dry-run controls. Every discovered PPO outer seed is model identity only and is
evaluated with independently configured CARP-S seeds.
EOF
}

if (( $# < 1 )); then
    usage >&2
    exit 2
fi

repository_root="$(cd "${script_directory}/.." && pwd -P)"
cd "${repository_root}"

ppo_results_input="$1"
shift
checkpoint="${DACBO_POLICY_MODEL:-final}"
eval_results_input="${DACBO_EVAL_RESULTS_DIR:-}"
explicit_model=""
explicit_normalizer=""
overwrite="${DACBO_OVERWRITE_RUN_SCRIPT:-0}"
while (( $# )); do
    case "$1" in
        --checkpoint) checkpoint="${2:?--checkpoint requires a value}"; shift 2 ;;
        --output) eval_results_input="${2:?--output requires a value}"; shift 2 ;;
        --explicit-model) explicit_model="${2:?--explicit-model requires a value}"; shift 2 ;;
        --explicit-normalizer) explicit_normalizer="${2:?--explicit-normalizer requires a value}"; shift 2 ;;
        --overwrite) overwrite=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
if [[ ! -d "${ppo_results_input}" ]]; then
    echo "PPO results directory does not exist: ${ppo_results_input}" >&2
    exit 1
fi
ppo_results="$(cd "${ppo_results_input}" && pwd -P)"

eval_results_input="${eval_results_input:-${ppo_results}_carps_eval}"
mkdir -p "${eval_results_input}"
eval_results="$(cd "${eval_results_input}" && pwd -P)"
generated_config_root="${eval_results}/config"
generated_policy_root="${generated_config_root}/policy/optimized"
mkdir -p "${generated_policy_root}"

run_script="${DACBO_EVAL_RUN_SCRIPT:-${eval_results}/run_carps_eval.sh}"
mkdir -p "$(dirname "${run_script}")"
run_script_directory="$(cd "$(dirname "${run_script}")" && pwd -P)"
run_script="${run_script_directory}/$(basename "${run_script}")"
if [[ -e "${run_script}" && "${overwrite}" != "1" ]]; then
    echo "Run script already exists: ${run_script}" >&2
    echo "Set DACBO_OVERWRITE_RUN_SCRIPT=1 to replace it." >&2
    exit 1
fi

if [[ -n "${DACBO_PYTHON:-}" ]]; then
    python_bin="${DACBO_PYTHON}"
elif [[ -x "${repository_root}/.venv/bin/python" ]]; then
    python_bin="${repository_root}/.venv/bin/python"
else
    python_bin="$(command -v python)"
fi
if [[ "${python_bin}" != */* ]]; then
    python_bin="$(command -v "${python_bin}")"
fi
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable is not executable: ${python_bin}" >&2
    exit 1
fi
python_bin="$(cd "$(dirname "${python_bin}")" && pwd -P)/$(basename "${python_bin}")"

policy_model="${checkpoint}"
case "${policy_model}" in
    final|full_best|frequent_best|explicit|best|last) ;;
    *)
        echo "Unknown DACBO_POLICY_MODEL=${policy_model}." >&2
        echo "Expected final, full_best, frequent_best, explicit, best, or last." >&2
        exit 2
        ;;
esac
optimizer_group="PPO-Structured-MLP"
inventory_path="${generated_config_root}/ppo_inventory.json"

# Create domain-neutral CARP-S policy bundles. Evaluation environments are
# composed later from each evaluation task, never copied from training.
collector_args=(
    "${ppo_results}"
    --configs-path="${generated_policy_root}"
    --model-selection="${policy_model}"
    --inventory-path="${inventory_path}"
)
if [[ -n "${explicit_model}" ]]; then collector_args+=(--explicit-model="${explicit_model}"); fi
if [[ -n "${explicit_normalizer}" ]]; then collector_args+=(--explicit-normalizer="${explicit_normalizer}"); fi
"${python_bin}" -m dacboenv.experiment.collect_ppo "${collector_args[@]}"

inventory_tsv="${generated_config_root}/ppo_inventory.tsv"
"${python_bin}" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print("{}|{}|{}|{}|{}".format(r["frequency"],r["seed"],r["task_id"],r["model_path"],r["action_family"])) for r in p["policies"]]' \
    "${inventory_path}" > "${inventory_tsv}"
mapfile -t selected_entries < "${inventory_tsv}"
if (( ${#selected_entries[@]} == 0 )); then
    echo "No completed PPO runs with trained frequent-validation checkpoints were found below ${ppo_results}." >&2
    exit 1
fi

for entry in "${selected_entries[@]}"; do
    IFS="|" read -r frequency policy_seed task_id model_path action_family <<< "${entry}"
    policy_config="${generated_policy_root}/${optimizer_group}/${task_id}/seed${policy_seed}.yaml"
    if [[ ! -s "${policy_config}" ]]; then
        echo "Collector did not create the expected CARP-S policy config: ${policy_config}" >&2
        exit 1
    fi
    configured_model="$(
        "${python_bin}" -c \
            'import sys; from omegaconf import OmegaConf; print(OmegaConf.load(sys.argv[1]).optimizer.policy_kwargs.model)' \
            "${policy_config}"
    )"
    if [[ "${configured_model}" != "${model_path}" ]]; then
        echo "Generated policy config points at an unexpected model:" >&2
        echo "  expected: ${model_path}" >&2
        echo "  found:    ${configured_model}" >&2
        exit 1
    fi
done

{
    cat <<EOF
#!/usr/bin/env bash

# Generated by scripts/create_ppo_eval_script.sh.
# Protocol namespace: evaluation_protocol_v2_deterministic
# PPO results: ${ppo_results}
# CARP-S results: ${eval_results}
# PPO runs discovered: ${#selected_entries[@]}
# PPO checkpoint mode: ${policy_model}
#
# Usage:
#   bash ${run_script} [training|bbob_2d_8d|both] [METHODS]
#   METHODS: ppo (default), baselines, all, or a comma-separated subset of
#            ppo,random,static,smac
#   smac uses the structured DACBO/SMAC stack with +policy=defaultaction;
#   NoOpPolicy leaves the configured acquisition function unchanged.
#
# Independent evaluation axes:
#   DACBO_EVAL_SEEDS=0,1,2,3,4,5,6,7,8,9
#                                      CARP-S/SMAC seeds (default shown)
#   DACBO_INNER_SEEDS=...              backward-compatible alias
#   DACBO_BBOB_INSTANCE_ID=0           instance for the broad 2D/8D suite
#
# Execution controls:
#   DACBO_EVAL_METHODS=...           METHODS when the second argument is omitted
#   DACBO_DRY_RUN=1                print commands without running/submitting
#   DACBO_EVAL_LAUNCHER=slurm      "slurm" (default) or "local"
#   DACBO_PARALLEL_LAUNCHERS=1      start all Hydra launchers concurrently
#                                  (default: 1 for Slurm, 0 for local)
#   DACBO_OVERWRITE_EVAL_RESULTS=1 allow running into populated optimizer roots
#   DACBO_SLURM_PARTITION=normal
#   DACBO_SLURM_TIMEOUT_MIN=360
#   DACBO_SLURM_MEM_PER_CPU=8G
#   Slurm starts every selected method/suite Hydra launcher before waiting.
#
# The exact training tasks are dimensions 2 and 4, functions
# 3,6,8,13,17,21, BBOB instance 0. Ten default evaluation seeds include the
# original four training seeds (0..3) plus 4..9. The broad suite is all
# functions 1..24 at dimensions 2 and 8. All are selected through CARP-S's
# native +task/BBOB=cfg_* config group.

set -euo pipefail
set -f

export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

repository_root=$(printf '%q' "${repository_root}")
python_bin=$(printf '%q' "${python_bin}")
eval_root=$(printf '%q' "${eval_results}")
generated_config_root=$(printf '%q' "${generated_config_root}")
evaluation_reward_config=reference_regret_improvement

policy_entries=(
EOF
    for entry in "${selected_entries[@]}"; do
        printf '    %q\n' "${entry}"
    done
    cat <<'EOF'
)

if (( $# > 2 )); then
    echo "Usage: bash $0 [training|bbob_2d_8d|both] [METHODS]" >&2
    exit 2
fi

suite="${1:-both}"
case "${suite}" in
    training|bbob_2d_8d|both) ;;
    *)
        echo "Unknown suite: ${suite}; expected training, bbob_2d_8d, or both." >&2
        exit 2
        ;;
esac

method_selection="${2:-${DACBO_EVAL_METHODS:-ppo}}"
run_ppo=0
run_random=0
run_static=0
run_smac=0
case "${method_selection}" in
    baselines)
        run_random=1
        run_static=1
        run_smac=1
        ;;
    all)
        run_ppo=1
        run_random=1
        run_static=1
        run_smac=1
        ;;
    *)
        IFS="," read -r -a requested_methods <<< "${method_selection}"
        if (( ${#requested_methods[@]} == 0 )); then
            echo "At least one evaluation method is required." >&2
            exit 2
        fi
        declare -A seen_methods=()
        for requested_method in "${requested_methods[@]}"; do
            if [[ -n "${seen_methods[${requested_method}]+x}" ]]; then
                echo "Duplicate evaluation method: ${requested_method}." >&2
                exit 2
            fi
            seen_methods["${requested_method}"]=1
            case "${requested_method}" in
                ppo) run_ppo=1 ;;
                random) run_random=1 ;;
                static) run_static=1 ;;
                smac) run_smac=1 ;;
                *)
                    echo "Unknown evaluation method: ${requested_method}." >&2
                    echo "Expected ppo, random, static, smac, baselines, or all." >&2
                    exit 2
                    ;;
            esac
        done
        ;;
esac

eval_seeds="${DACBO_EVAL_SEEDS:-${DACBO_INNER_SEEDS:-0,1,2,3,4,5,6,7,8,9}}"
broad_instance_id="${DACBO_BBOB_INSTANCE_ID:-0}"
launcher="${DACBO_EVAL_LAUNCHER:-slurm}"
dry_run="${DACBO_DRY_RUN:-0}"
overwrite_eval_results="${DACBO_OVERWRITE_EVAL_RESULTS:-0}"
if [[ -n "${DACBO_PARALLEL_LAUNCHERS:-}" ]]; then
    parallel_launchers="${DACBO_PARALLEL_LAUNCHERS}"
elif [[ "${launcher}" == "slurm" ]]; then
    parallel_launchers=1
else
    parallel_launchers=0
fi

case "${launcher}" in
    slurm|local) ;;
    *)
        echo "Unknown DACBO_EVAL_LAUNCHER=${launcher}; expected slurm or local." >&2
        exit 2
        ;;
esac
case "${dry_run}" in
    0|1) ;;
    *)
        echo "DACBO_DRY_RUN must be 0 or 1; got: ${dry_run}" >&2
        exit 2
        ;;
esac
case "${overwrite_eval_results}" in
    0|1) ;;
    *)
        echo "DACBO_OVERWRITE_EVAL_RESULTS must be 0 or 1; got: ${overwrite_eval_results}" >&2
        exit 2
        ;;
esac
case "${parallel_launchers}" in
    0|1) ;;
    *)
        echo "DACBO_PARALLEL_LAUNCHERS must be 0 or 1; got: ${parallel_launchers}" >&2
        exit 2
        ;;
esac

if [[ ! "${eval_seeds}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "Evaluation seeds must be comma-separated non-negative integers; got: ${eval_seeds}" >&2
    exit 2
fi
IFS="," read -r -a eval_seed_values <<< "${eval_seeds}"
n_eval_seeds="${#eval_seed_values[@]}"
declare -A seen_eval_seeds=()
for eval_seed in "${eval_seed_values[@]}"; do
    if [[ ! "${eval_seed}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "Evaluation seeds must use canonical integers without leading zeros; got: ${eval_seed}" >&2
        exit 2
    fi
    if [[ -n "${seen_eval_seeds[${eval_seed}]+x}" ]]; then
        echo "Duplicate evaluation seed: ${eval_seed}." >&2
        exit 2
    fi
    seen_eval_seeds["${eval_seed}"]=1
done
case "${broad_instance_id}" in
    0|1|2) ;;
    *)
        echo "DACBO_BBOB_INSTANCE_ID must be 0, 1, or 2; CARP-S packages those instances." >&2
        exit 2
        ;;
esac

n_policies="${#policy_entries[@]}"
baseline_frequencies=(1 5 10)
static_policy_entries=(
    "level_000|0.00"
    "level_025|0.25"
    "level_050|0.50"
    "level_075|0.75"
    "level_100|1.00"
)

n_optimizer_configs=0
selected_optimizer_ids=()
if [[ "${run_ppo}" == "1" ]]; then
    n_optimizer_configs=$((n_optimizer_configs + n_policies))
    for entry in "${policy_entries[@]}"; do
        IFS="|" read -r frequency policy_seed task_id model_path action_family <<< "${entry}"
        selected_optimizer_ids+=("PPO-Structured-MLP--${task_id}--seed${policy_seed}")
    done
fi
if [[ "${run_random}" == "1" ]]; then
    n_optimizer_configs=$((n_optimizer_configs + ${#baseline_frequencies[@]}))
    for frequency in "${baseline_frequencies[@]}"; do
        selected_optimizer_ids+=("Random-f${frequency}")
    done
fi
if [[ "${run_static}" == "1" ]]; then
    n_optimizer_configs=$((n_optimizer_configs + ${#baseline_frequencies[@]} * ${#static_policy_entries[@]}))
    for frequency in "${baseline_frequencies[@]}"; do
        for static_entry in "${static_policy_entries[@]}"; do
            IFS="|" read -r static_choice static_alpha <<< "${static_entry}"
            selected_optimizer_ids+=("StaticAlpha-${static_alpha}-f${frequency}")
        done
    done
fi
if [[ "${run_smac}" == "1" ]]; then
    n_optimizer_configs=$((n_optimizer_configs + 1))
    selected_optimizer_ids+=("DefaultPolicy")
    if ! "${python_bin}" -c \
        'import dacboenv.optimizer as m; assert getattr(m, "SUPPORTS_NOOP_POLICY_ACTION", False), f"{m.__file__} lacks NoOp action support"'
    then
        echo "The selected Python runtime cannot safely evaluate +policy=defaultaction." >&2
        echo "Update dacboenv/optimizer.py before submitting DefaultPolicy jobs." >&2
        exit 1
    fi
fi

if [[ "${dry_run}" == "0" && "${overwrite_eval_results}" == "0" ]]; then
    case "${suite}" in
        training) suite_ids_to_check=("training") ;;
        bbob_2d_8d) suite_ids_to_check=("bbob_2d_8d") ;;
        both) suite_ids_to_check=("training" "bbob_2d_8d") ;;
    esac
    for suite_id_to_check in "${suite_ids_to_check[@]}"; do
        for optimizer_id_to_check in "${selected_optimizer_ids[@]}"; do
            optimizer_root_to_check="${eval_root}/${suite_id_to_check}/${optimizer_id_to_check}"
            if [[ -d "${optimizer_root_to_check}" ]] \
                && [[ -n "$(find "${optimizer_root_to_check}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
                echo "Refusing to overwrite populated CARP-S result root: ${optimizer_root_to_check}" >&2
                echo "Select other methods, use a new evaluation directory, or set" >&2
                echo "DACBO_OVERWRITE_EVAL_RESULTS=1." >&2
                exit 1
            fi
        done
    done
fi

training_jobs=$((n_optimizer_configs * n_eval_seeds * 12))
broad_jobs=$((n_optimizer_configs * n_eval_seeds * 48))
case "${suite}" in
    training) projected_jobs="${training_jobs}" ;;
    bbob_2d_8d) projected_jobs="${broad_jobs}" ;;
    both) projected_jobs=$((training_jobs + broad_jobs)) ;;
esac

echo "CARP-S suite: ${suite}"
echo "Evaluation methods: ${method_selection}"
echo "Optimizer configurations: ${n_optimizer_configs}"
echo "Evaluation seeds: ${eval_seeds}"
echo "Projected CARP-S jobs: ${projected_jobs}"
echo "Result root: ${eval_root}"
echo "Parallel Hydra launchers: ${parallel_launchers}"

cd "${repository_root}"

launcher_pids=()
launcher_labels=()

launch_carps_sweep() {
    local launch_label="$1"
    local suite_id="$2"
    local optimizer_id="$3"
    local task_configs="$4"
    shift 4

    local suite_root="${eval_root}/${suite_id}"
    local -a command=(
        "${python_bin}"
        -m carps.run
        --multirun
        "hydra.searchpath=[file://${generated_config_root},pkg://dacboenv/configs]"
        "+task/BBOB=${task_configs}"
        "$@"
        "seed=${eval_seeds}"
        "baserundir=${suite_root}"
        "hydra.sweep.dir=${suite_root}/${optimizer_id}"
        "hydra.sweep.subdir=\${benchmark_id}/\${task_id}/\${seed}"
        "hydra.sweeper.max_batch_size=500"
    )

    if [[ "${launcher}" == "slurm" ]]; then
        command+=(
            "hydra/launcher=submitit_slurm"
            "hydra.launcher.partition=${DACBO_SLURM_PARTITION:-normal}"
            "hydra.launcher.cpus_per_task=1"
            "hydra.launcher.mem_per_cpu=${DACBO_SLURM_MEM_PER_CPU:-8G}"
            "hydra.launcher.timeout_min=${DACBO_SLURM_TIMEOUT_MIN:-360}"
            "hydra.launcher.name=dacbo-carps-eval"
        )
    fi

    echo
    echo "CARP-S launcher: ${launch_label}; suite ${suite_id}; optimizer ${optimizer_id}"
    if [[ "${dry_run}" == "1" ]]; then
        printf "%q " "${command[@]}"
        printf "\n"
    elif [[ "${parallel_launchers}" == "1" ]]; then
        "${command[@]}" &
        local launcher_pid=$!
        launcher_pids+=("${launcher_pid}")
        launcher_labels+=("${launch_label}/${suite_id}")
        echo "Started Hydra launcher PID ${launcher_pid}: ${launch_label}/${suite_id}"
    else
        "${command[@]}"
    fi
}

run_ppo_configs() {
    local suite_id="$1"
    local task_configs="$2"
    for entry in "${policy_entries[@]}"; do
        local frequency
        local policy_seed
        local task_id
        local model_path
        local action_family
        IFS="|" read -r frequency policy_seed task_id model_path action_family <<< "${entry}"

        if [[ ! -s "${model_path}" ]]; then
            echo "Selected model is missing or empty: ${model_path}" >&2
            exit 1
        fi

        local policy_group="PPO-Structured-MLP/${task_id}"
        local optimizer_id="PPO-Structured-MLP--${task_id}--seed${policy_seed}"
        local action_config
        local observation_config
        case "${action_family}" in
            wei) action_config="wei_alpha_discrete"; observation_config="structured" ;;
            af_selection) action_config="af_selection_discrete"; observation_config="structured_af_selection" ;;
            lcb_quantile) action_config="lcb_quantile_discrete"; observation_config="structured_quantile" ;;
            ucb_quantile) action_config="ucb_quantile_discrete"; observation_config="structured_quantile" ;;
            *) echo "Unsupported learned-policy action family: ${action_family}" >&2; exit 1 ;;
        esac
        launch_carps_sweep \
            "ppo-f${frequency}-seed${policy_seed}" \
            "${suite_id}" \
            "${optimizer_id}" \
            "${task_configs}" \
            "+eval=base" \
            "+env=base" \
            "+env/opt=base" \
            "+env/action=${action_config}" \
            "+env/interaction_freq=f${frequency}" \
            "+env/obs=${observation_config}" \
            "+env/reward=${evaluation_reward_config}" \
            "+policy/optimized/${policy_group}=seed${policy_seed}"
    done
}

run_random_configs() {
    local suite_id="$1"
    local task_configs="$2"
    for frequency in "${baseline_frequencies[@]}"; do
        local optimizer_id="Random-f${frequency}"
        launch_carps_sweep \
            "random-f${frequency}" \
            "${suite_id}" \
            "${optimizer_id}" \
            "${task_configs}" \
            "+eval=base" \
            "+env=base" \
            "+env/opt=base" \
            "+env/action=wei_alpha_discrete" \
            "+env/interaction_freq=f${frequency}" \
            "+env/obs=structured" \
            "+env/reward=${evaluation_reward_config}" \
            "+policy=random" \
            "optimizer_id=${optimizer_id}" \
            "policy_id=${optimizer_id}" \
            "dacboenv.evaluation_mode=false" \
            "dacboenv.terminate_after_reference_performance_reached=false"
    done
}

run_static_configs() {
    local suite_id="$1"
    local task_configs="$2"
    for frequency in "${baseline_frequencies[@]}"; do
        for static_entry in "${static_policy_entries[@]}"; do
            local static_choice
            local static_alpha
            IFS="|" read -r static_choice static_alpha <<< "${static_entry}"
            local optimizer_id="StaticAlpha-${static_alpha}-f${frequency}"
            launch_carps_sweep \
                "static-alpha${static_alpha}-f${frequency}" \
                "${suite_id}" \
                "${optimizer_id}" \
                "${task_configs}" \
                "+eval=base" \
                "+env=base" \
                "+env/opt=base" \
                "+env/action=wei_alpha_discrete" \
                "+env/interaction_freq=f${frequency}" \
                "+env/obs=structured" \
                "+env/reward=${evaluation_reward_config}" \
                "+policy/static/wei_discrete=${static_choice}" \
                "optimizer_id=${optimizer_id}" \
                "policy_id=${optimizer_id}" \
                "dacboenv.evaluation_mode=false" \
                "dacboenv.terminate_after_reference_performance_reached=false"
        done
    done
}

run_smac_config() {
    local suite_id="$1"
    local task_configs="$2"
    launch_carps_sweep \
        "smac-noop" \
        "${suite_id}" \
        "DefaultPolicy" \
        "${task_configs}" \
        "+eval=base" \
        "+env=base" \
        "+env/opt=base" \
        "+env/action=wei_alpha_discrete" \
        "+env/interaction_freq=f1" \
        "+env/obs=structured" \
        "+env/reward=${evaluation_reward_config}" \
        "+policy=defaultaction" \
        "dacboenv.evaluation_mode=false" \
        "dacboenv.terminate_after_reference_performance_reached=false"
}

run_suite() {
    local suite_id="$1"
    local task_configs="$2"
    if [[ "${run_ppo}" == "1" ]]; then
        run_ppo_configs "${suite_id}" "${task_configs}"
    fi
    if [[ "${run_random}" == "1" ]]; then
        run_random_configs "${suite_id}" "${task_configs}"
    fi
    if [[ "${run_static}" == "1" ]]; then
        run_static_configs "${suite_id}" "${task_configs}"
    fi
    if [[ "${run_smac}" == "1" ]]; then
        run_smac_config "${suite_id}" "${task_configs}"
    fi
}

training_task_configs="cfg_2_3_0,cfg_2_6_0,cfg_2_8_0,cfg_2_13_0,cfg_2_17_0,cfg_2_21_0,cfg_4_3_0,cfg_4_6_0,cfg_4_8_0,cfg_4_13_0,cfg_4_17_0,cfg_4_21_0"
broad_task_config_values=()
for dimension in 2 8; do
    for function_id in {1..24}; do
        broad_task_config_values+=("cfg_${dimension}_${function_id}_${broad_instance_id}")
    done
done
printf -v broad_task_configs "%s," "${broad_task_config_values[@]}"
broad_task_configs="${broad_task_configs%,}"

if [[ "${suite}" == "training" || "${suite}" == "both" ]]; then
    run_suite "training" "${training_task_configs}"
fi

if [[ "${suite}" == "bbob_2d_8d" || "${suite}" == "both" ]]; then
    run_suite "bbob_2d_8d" "${broad_task_configs}"
fi

launcher_failures=0
for launcher_index in "${!launcher_pids[@]}"; do
    launcher_pid="${launcher_pids[${launcher_index}]}"
    launcher_label="${launcher_labels[${launcher_index}]}"
    if wait "${launcher_pid}"; then
        echo "Hydra launcher completed: ${launcher_label}"
    else
        launcher_status=$?
        echo "Hydra launcher failed (${launcher_status}): ${launcher_label}" >&2
        launcher_failures=1
    fi
done
if (( launcher_failures != 0 )); then
    exit 1
fi
EOF
} > "${run_script}"

chmod u+x "${run_script}"

echo
echo "Created CARP-S evaluation launcher:"
echo "  ${run_script}"
echo
echo "It contains ${#selected_entries[@]} learned policies selected by checkpoint mode ${policy_model}."
echo "Review learned policies plus all baselines without submitting:"
echo "  DACBO_DRY_RUN=1 bash ${run_script} both all"
echo "Submit learned policies plus all baselines to Otus Slurm:"
echo "  bash ${run_script} both all"
