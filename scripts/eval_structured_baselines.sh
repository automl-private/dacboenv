#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"

export HYDRA_FULL_ERROR=1

repository_root="$(cd -- "${script_directory}/.." && pwd -P)"
python_bin="${DACBO_PYTHON:-${repository_root}/.venv/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python environment is missing or not executable: ${python_bin}" >&2
    exit 2
fi
cd -- "${repository_root}"

evaluation_set="${DACBO_EVALUATION_SET:-bbob_validation}"
manifest_path="${repository_root}/dacboenv/configs/instance_sets/${evaluation_set}.yaml"
baseline_run_dir="${DACBO_BASELINE_RUN_DIR:-${repository_root}/runs_structured_baselines_carps_new}"
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
        "seed=range(0,6)"
        "+eval=base"
        "+cluster=cpu_noctua"
        "+env/reward=reference_free_improvement"
        "$@"
        "dacboenv.context_split=validation"
        "dacboenv.instance_selector_class._target_=dacboenv.env.instance.RoundRobinInstanceSelector"
        "dacboenv.instance_selector_class._partial_=true"
        "+dacboenv.instance_selector_class.offset=0"
        "dacboenv.evaluation_mode=true"
        "dacboenv.terminate_after_reference_performance_reached=false"
        "++optimizer_id=${label}"
        "baserundir=${baseline_run_dir}"
    )

    if [[ "${dry_run}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}" &
    fi
}

action_families=(wei_alpha_discrete) # lcb_quantile_discrete ucb_quantile_discrete af_selection_discrete)
observation_families=(structured structured_quantile structured_quantile structured_af_selection)
tasks=('+task/BBOB=glob(cfg_2_*_0)' '+task/BBOB=glob(cfg_8_*_0)' '+task/YAHPO/SO=glob(*)')

for task in "${tasks[@]}"; do
    for index in "${!action_families[@]}"; do
        action_family="${action_families[index]}"
        observation_family="${observation_families[index]}"

        launch_carps "random-${action_family}" \
            "$task" \
            +env=base \
            +env/opt=base \
            "+env/action=${action_family}" \
            +env/interaction_freq=f1 \
            "+env/obs=${observation_family}" \
            +policy=random

        for action in 0 1 2 3 4; do
            launch_carps "static-${action_family}-action${action}" \
                "$task" \
                +env=base \
                +env/opt=base \
                "+env/action=${action_family}" \
                "+env/interaction_freq=f1" \
                "+env/obs=${observation_family}" \
                "+policy/static/discrete_action=action_${action}"
        done
    done

    # The native no-op controller preserves the configured SMAC acquisition and
    # consumes the same CARP-S task/seed contexts as every action-based baseline.
    launch_carps default-smac \
        "$task" \
        +env=base \
        +env/opt=base \
        +env/action=wei_alpha_discrete \
        +env/interaction_freq=f1 \
        +env/obs=structured \
        +policy=defaultaction
done

wait
