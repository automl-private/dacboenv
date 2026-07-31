#!/usr/bin/env bash

set -euo pipefail

export HYDRA_FULL_ERROR=1
mkdir -p slurmlogs/ppo

usage() {
    cat <<'EOF'
Usage:
  bash scripts/launch_ppo.sh [MODE] [HYDRA_OVERRIDE ...]

MODE defaults to `wei` and accepts:
  wei                 existing five-level WEI-alpha controller
  lcb_quantile        exploratory lower posterior-quantile controller
  ucb_quantile        uncertainty-averse upper posterior-quantile controller
  posterior_quantile  alias for the recommended lcb_quantile mode
  af_selection        posterior mean / PI / EI / LCB / max-variance selector
  all                 all four controller families

Every family submits f1, f5, and f10 arrays. `DACBO_RUN_DIR` can force one
shared result root; otherwise each family uses its own collision-free root.
Additional arguments are forwarded to every PPO worker as Hydra overrides.
EOF
}

mode="${DACBO_PPO_MODE:-wei}"
if (( $# > 0 )); then
    case "$1" in
        wei|lcb_quantile|ucb_quantile|posterior_quantile|af_selection|all)
            mode="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        +*|~*|*=*)
            # No explicit mode: retain the default and forward this override.
            ;;
        *)
            mode="$1"
            shift
            ;;
    esac
fi

case "${mode}" in
    wei)
        training_families=("structured_ppo|runs_structured")
        ;;
    lcb_quantile)
        training_families=("lcb_quantile_ppo|runs_lcb_quantile")
        ;;
    ucb_quantile)
        training_families=("ucb_quantile_ppo|runs_ucb_quantile")
        ;;
    posterior_quantile)
        training_families=("ucb_quantile_ppo|runs_post_quantile")
        ;;
    af_selection)
        training_families=("af_selection_ppo|runs_af_selection")
        ;;
    all)
        training_families=(
            "structured_ppo|runs_structured"
            "lcb_quantile_ppo|runs_lcb_quantile"
            "ucb_quantile_ppo|runs_ucb_quantile"
            "af_selection_ppo|runs_af_selection"
        )
        ;;
    *)
        echo "Unknown DACBO_PPO_MODE=${mode}." >&2
        usage >&2
        exit 2
        ;;
esac

for family_entry in "${training_families[@]}"; do
    IFS="|" read -r training_prefix default_run_directory <<< "${family_entry}"
    run_directory="${DACBO_RUN_DIR:-${default_run_directory}}"
    for frequency in 1 5 10; do
        training_config="${training_prefix}_f${frequency}"
        echo "Launch Stable-Baselines3 PPO: ${training_config} -> ${run_directory}"
        sbatch \
            --export="ALL,DACBO_RUN_DIR=${run_directory}" \
            scripts/opt_ppo.sh \
            "+training=${training_config}" \
            "$@"
    done
done
