#!/usr/bin/env bash
# Submit the D1 fixed-frequency-5 Double-DQN pilot through the existing
# algorithm-neutral scripts/opt_ppo.sh worker.
#
# Usage:
#   bash scripts/launch_d1_f5.sh [all|yahpo|mixed] [RUN_ROOT] [HYDRA_OVERRIDE ...]
#
# Examples:
#   bash scripts/launch_d1_f5.sh all
#   bash scripts/launch_d1_f5.sh mixed /absolute/path/to/runs_d1_f5
#   bash scripts/launch_d1_f5.sh yahpo runs_d1_f5 rl_algorithm.hyperparameters.verbose=0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  launch_d1_f5.sh [all|yahpo|mixed] [RUN_ROOT] [HYDRA_OVERRIDE ...]

Defaults:
  mode:      all
  RUN_ROOT:  <repository>/runs_d1_ddqn_f5
  seeds:     0, 1, 2 through a Slurm array

The script uses scripts/opt_ppo.sh. Despite its historical name, that worker
must call the algorithm-neutral `python -m dacboenv.experiment.rl` entry point.
No validation jobs are launched by the D1 configs.
EOF
}

mode="${1:-all}"
case "${mode}" in
    all|yahpo|mixed)
        shift || true
        ;;
    -h|--help)
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
    run_root_input="${repo_root}/runs_d1_ddqn_f5"
fi
mkdir -p "${run_root_input}"
run_root="$(cd "${run_root_input}" && pwd -P)"

# CONFIG|WORKERS|CPUS|MEMORY
jobs=()
case "${mode}" in
    all)
        jobs+=(
            'yahpo_wei_double_dqn_f5_d1|12|13|64G'
            'mixed_wei_double_dqn_f5_d1|24|25|96G'
        )
        ;;
    yahpo)
        jobs+=( 'yahpo_wei_double_dqn_f5_d1|12|13|64G' )
        ;;
    mixed)
        jobs+=( 'mixed_wei_double_dqn_f5_d1|24|25|96G' )
        ;;
esac

for entry in "${jobs[@]}"; do
    IFS='|' read -r config workers cpus memory <<< "${entry}"
    echo "Submit ${config}: seeds 0-2, workers=${workers}, output=${run_root}"
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
