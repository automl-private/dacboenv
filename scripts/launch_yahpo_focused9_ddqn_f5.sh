#!/usr/bin/env bash
# Submit focused nine-task YAHPO Double-DQN f=5 runs through the existing
# algorithm-neutral scripts/opt_ppo.sh worker on TSUBAME.
#
# Usage:
#   bash scripts/launch_yahpo_focused9_ddqn_f5.sh MODE [RUN_ROOT] [OVERRIDE ...]
#
# MODE:
#   primary   61,440 BO evaluations, 12,288 transitions, 2,880 nominal updates
#   short     30,720 BO evaluations,  6,144 transitions, 1,344 nominal updates
#   both      submit both configurations
#
# Examples:
#   bash scripts/launch_yahpo_focused9_ddqn_f5.sh primary \
#     /scratch/.../runs_yahpo_focused9
#
#   bash scripts/launch_yahpo_focused9_ddqn_f5.sh both \
#     /scratch/.../runs_yahpo_focused9 \
#     rl_algorithm.hyperparameters.verbose=0

set -euo pipefail
set -f

usage() {
    sed -n '1,28p' "$0"
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
    run_root_input="${repo_root}/runs_yahpo_focused9_ddqn_f5"
fi
mkdir -p "${run_root_input}"
run_root="$(cd "${run_root_input}" && pwd -P)"

configs=()
case "${mode}" in
    primary)
        configs+=(yahpo_wei_double_dqn_f5_d1_smaller)
        ;;
    short)
        configs+=(yahpo_wei_double_dqn_f5_d1_smaller_short)
        ;;
    both)
        configs+=(
            yahpo_wei_double_dqn_f5_d1_smaller
            yahpo_wei_double_dqn_f5_d1_smaller_short
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
