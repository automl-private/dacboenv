#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"

export HYDRA_FULL_ERROR=1

repository_root="$(cd -- "${script_directory}/.." && pwd -P)"
python_bin="${DACBO_PYTHON:-${repository_root}/.env/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment is missing or not executable: ${python_bin}" >&2
    exit 2
fi
cd -- "${repository_root}"

evaluation_set="${DACBO_EVALUATION_SET:-bbob_validation}"
manifest_path="${repository_root}/dacboenv/configs/instance_sets/${evaluation_set}.yaml"
baseline_run_dir="${DACBO_BASELINE_RUN_DIR:-${repository_root}/runs_structured_baselines_carps}"
dry_run="${DACBO_BASELINE_DRY_RUN:-0}"

if [[ ! -f "${manifest_path}" ]]; then
    echo "Unknown evaluation manifest: ${manifest_path}" >&2
    exit 2
fi
if [[ "${evaluation_set}" == *test* && "${DACBO_ALLOW_SEALED_TEST:-0}" != "1" ]]; then
    echo "Refusing sealed test manifest '${evaluation_set}'." >&2
    echo "Set DACBO_ALLOW_SEALED_TEST=1 only for an authorized final report." >&2
    exit 2
fi
if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
    echo "DACBO_BASELINE_DRY_RUN must be 0 or 1, got: ${dry_run}" >&2
    exit 2
fi

launch_carps() {
    local label="$1"
    shift
    local -a command=(
        "${python_bin}"
        -m carps.run
        --multirun
        "hydra.searchpath=[pkg://dacboenv/configs]"
        "+task/BBOB=glob(cfg_2_*)"
        "seed=${evaluation_seeds}"
        "+eval=base"
        "$@"
        "dacboenv.context_split=${context_split}"
        "dacboenv.instance_selector_class._target_=dacboenv.env.instance.RoundRobinInstanceSelector"
        "dacboenv.instance_selector_class._partial_=true"
        "+dacboenv.instance_selector_class.offset=0"
        "dacboenv.evaluation_mode=false"
        "dacboenv.terminate_after_reference_performance_reached=false"
        "baserundir=${baseline_run_dir}"
    )

    if [[ "${dry_run}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
}

echo "CARP-S baseline protocol: deterministic-v2"
echo "Manifest: ${evaluation_set} (${manifest_hash})"
echo "Contexts: ${task_configs} x seeds ${evaluation_seeds}"
echo "Result root: ${baseline_run_dir}"

action_families=(wei_alpha_discrete lcb_quantile_discrete ucb_quantile_discrete af_selection_discrete)
observation_families=(structured structured_quantile structured_quantile structured_af_selection)

for index in "${!action_families[@]}"; do
    action_family="${action_families[index]}"
    observation_family="${observation_families[index]}"

    launch_carps "random-${action_family}" \
        +env=base \
        +env/opt=base \
        "+env/action=${action_family}" \
        +env/interaction_freq=f1 \
        "+env/obs=${observation_family}" \
        +env/reward=true_regret_improvement \
        +policy=random

    launch_carps "static-${action_family}" \
        +env=base \
        +env/opt=base \
        "+env/action=${action_family}" \
        +env/interaction_freq=f1,f5,f10 \
        "+env/obs=${observation_family}" \
        +env/reward=true_regret_improvement \
        +policy/static/discrete_action=action_0,action_1,action_2,action_3,action_4
done

# The native no-op controller preserves the configured SMAC acquisition and
# consumes the same CARP-S task/seed contexts as every action-based baseline.
launch_carps default-smac \
    +env=base \
    +env/opt=base \
    +env/action=wei_alpha_discrete \
    +env/interaction_freq=f1 \
    +env/obs=structured \
    +env/reward=true_regret_improvement \
    +policy=defaultaction
