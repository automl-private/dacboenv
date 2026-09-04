#!/usr/bin/env bash
# Submit exact-reference OptBench Double-DQN f=5 runs through the shared
# algorithm-neutral scripts/opt_ppo.sh worker on Otus.
#
# Usage:
#   bash scripts/launch_optbench_ddqn_f5.sh MODE [RUN_ROOT] [OVERRIDE ...]
#
# MODE:
#   primary   61,440 BO evaluations, 12,288 transitions, 2,880 nominal updates
#   short     30,720 BO evaluations,  6,144 transitions, 1,344 nominal updates
#   both      submit both configurations
#
# Example (run from the Otus DACBOEnv checkout):
#   bash scripts/launch_optbench_ddqn_f5.sh primary /ABS/RUN_ROOT

set -euo pipefail
set -f

usage() {
    sed -n '5,15p' "$0"
}

if (( $# < 1 )); then
    usage >&2
    exit 2
fi

mode="$1"
shift
case "${mode}" in
    primary|short|both) ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"
mkdir -p slurmlogs/ppo

if (( $# > 0 )) && [[ "$1" != +* && "$1" != ~* && "$1" != *=* ]]; then
    run_root_input="$1"
    shift
else
    run_root_input="${repo_root}/runs_optbench_ddqn_f5"
fi
mkdir -p "${run_root_input}"
run_root="$(cd "${run_root_input}" && pwd -P)"

python_bin="${repo_root}/.venv/bin/python"
[[ -x "${python_bin}" ]] || {
    echo "Otus environment is missing: ${python_bin}" >&2
    exit 1
}
"${python_bin}" -c 'import optbench; print("OptBench:", optbench.__file__)'
echo "Validating installed OptBench global minima against optbench_train..."
"${python_bin}" -m dacboenv.experiment.optbench_inventory

configs=()
case "${mode}" in
    primary)
        configs+=(optbench_wei_double_dqn_f5_d1)
        ;;
    short)
        configs+=(optbench_wei_double_dqn_f5_d1_short)
        ;;
    both)
        configs+=(
            optbench_wei_double_dqn_f5_d1
            optbench_wei_double_dqn_f5_d1_short
        )
        ;;
esac

for config in "${configs[@]}"; do
    echo "Submitting ${config}: outer seeds 0,1,2; workers=12; root=${run_root}"
    sbatch \
        --array=0-2 \
        --job-name="${config}" \
        --cpus-per-task=13 \
        --mem=64G \
        --export="ALL,DACBO_RUN_DIR=${run_root},DACBO_N_WORKERS=12" \
        scripts/opt_ppo.sh \
        "+training=${config}" \
        "$@"
done
