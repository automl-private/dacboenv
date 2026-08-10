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
baseline_run_dir="${DACBO_BASELINE_RUN_DIR:-${repository_root}/runs_structured_baselines}"

sealed_override=()
if [[ "${evaluation_set}" == *test* ]]; then
    if [[ "${DACBO_ALLOW_SEALED_TEST:-0}" != "1" ]]; then
        echo "Refusing sealed test manifest '${evaluation_set}'." >&2
        echo "Set DACBO_ALLOW_SEALED_TEST=1 only for an authorized final report." >&2
        exit 2
    fi
    sealed_override+=("experiment.allow_sealed_test=true")
fi

launcher_args=("baserundir=${baseline_run_dir}")
if [[ "${DACBO_BASELINE_JOBS:-1}" -gt 1 ]]; then
    launcher_args+=(
        "hydra/launcher=joblib"
        "hydra.launcher.n_jobs=${DACBO_BASELINE_JOBS}"
    )
fi

controller_tasks=(
    dacboenv_structured_reference_free
    dacboenv_structured_lcb_quantile
    dacboenv_structured_ucb_quantile
    dacboenv_structured_af_selection
)

for controller_task in "${controller_tasks[@]}"; do
    # Uniform random actions: identical contexts, ten independent policy RNGs.
    "${python_bin}" -m dacboenv.experiment.baseline \
        +baseline=structured_random \
        "task=${controller_task}" \
        +env/reward=true_regret_improvement \
        "+env/interaction_freq=f1,f5,f10" \
        "instance_sets@evaluation_instances=${evaluation_set}" \
        "${sealed_override[@]}" \
        "seed=range(0,10)" \
        "+cluster=cpu_noctua" \
        "${launcher_args[@]}" \
        --multirun &

    # Every exact static action for the same controller and context manifest.
    "${python_bin}" -m dacboenv.experiment.baseline \
        +baseline=structured_static_action \
        "task=${controller_task}" \
        +env/reward=true_regret_improvement \
        "+env/interaction_freq=f1,f5,f10" \
        "instance_sets@evaluation_instances=${evaluation_set}" \
        "${sealed_override[@]}" \
        "policy/static/discrete_action=action_0,action_1,action_2,action_3,action_4" \
        "seed=0" \
        "+cluster=cpu_noctua" \
        "${launcher_args[@]}" \
        --multirun &
done

# Exact native BlackBoxFacade defaults, evaluated once per paired context.
"${python_bin}" -m dacboenv.experiment.default_smac \
    +baseline=default_smac \
    "instance_sets@evaluation_instances=${evaluation_set}" \
    "${sealed_override[@]}" \
    +cluster=cpu_noctua \
    "${launcher_args[@]}" \
    --multirun

wait
