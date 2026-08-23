#!/usr/bin/env bash
# Submit the reduced-inventory D1 frequency-5 screen.
#
# Usage:
#   bash scripts/launch_d1_f5_small.sh [all|yahpo|mixed] [RUN_ROOT] [HYDRA_OVERRIDE ...]
#
# The screen uses two outer seeds (0 and 1), half the D1 BO-evaluation budget,
# and a reduced but six-scenario-complete YAHPO inventory.  It is exploratory
# and must not replace the full-inventory D1 result.

set -euo pipefail

mode="${1:-all}"
case "${mode}" in
    all|yahpo|mixed)
        shift
        ;;
    -h|--help)
        sed -n '1,16p' "$0"
        exit 0
        ;;
    *)
        echo "Unknown mode: ${mode}" >&2
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
    run_root_input="${repo_root}/runs_d1_ddqn_f5_small"
fi
mkdir -p "${run_root_input}"
run_root="$(cd "${run_root_input}" && pwd -P)"

jobs=()
case "${mode}" in
    all)
        jobs+=(
            'yahpo_wei_double_dqn_f5_d1_small|12|13|64G'
            'mixed_wei_double_dqn_f5_d1_small|24|25|96G'
        )
        ;;
    yahpo)
        jobs+=( 'yahpo_wei_double_dqn_f5_d1_small|12|13|64G' )
        ;;
    mixed)
        jobs+=( 'mixed_wei_double_dqn_f5_d1_small|24|25|96G' )
        ;;
esac

for entry in "${jobs[@]}"; do
    IFS='|' read -r config workers cpus memory <<< "${entry}"

    echo "Preflight ${config}"
    bash scripts/preflight_rl_config.sh "${config}" "$@"

    echo "Submit ${config}: outer seeds 0-1; workers=${workers}; output=${run_root}"
    sbatch \
        --array='0-2' \
        --job-name="${config}" \
        --cpus-per-task="${cpus}" \
        --mem="${memory}" \
        --export="ALL,DACBO_RUN_DIR=${run_root},DACBO_N_WORKERS=${workers}" \
        scripts/opt_ppo.sh \
        "+training=${config}" \
        "$@"
done
