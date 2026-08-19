#!/usr/bin/env bash

set -euo pipefail

export HYDRA_FULL_ERROR=1
mkdir -p slurmlogs/ppo

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/launch_ppo.sh [MODE] [HYDRA_OVERRIDE ...]

Existing PPO modes:
  wei                 five-level WEI-alpha controller (f1/f5/f10)
  lcb_quantile        lower posterior-quantile controller (f1/f5/f10)
  ucb_quantile        upper posterior-quantile controller (f1/f5/f10)
  posterior_quantile  alias for lcb_quantile
  af_selection        posterior mean / PI / EI / LCB / max variance (f1/f5/f10)
  all                 all four existing controller families

Stage-C modes (three outer seeds, mixed BBOB/YAHPO, interaction frequency 1):
  stage_c             DDQN structured + PPO GP + DDQN GP
  stage_c_ddqn        the two DDQN cells only
  stage_c_gp          PPO GP + DDQN GP only

DACBO_RUN_DIR can override the default result root. Additional arguments are
forwarded unchanged as Hydra overrides to scripts/opt_ppo.sh.
USAGE
}

mode="${DACBO_PPO_MODE:-wei}"
if (( $# > 0 )); then
    case "$1" in
        wei|lcb_quantile|ucb_quantile|posterior_quantile|af_selection|all|stage_c|stage_c_ddqn|stage_c_gp)
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

# Each entry is TRAINING_CONFIG|DEFAULT_RUN_ROOT|ARRAY_RANGE.
case "${mode}" in
    wei)
        training_jobs=(
            "structured_ppo_f1|runs_structured|0-4"
            "structured_ppo_f5|runs_structured|0-4"
            "structured_ppo_f10|runs_structured|0-4"
        )
        ;;
    lcb_quantile)
        training_jobs=(
            "lcb_quantile_ppo_f1|runs_lcb_quantile|0-4"
            "lcb_quantile_ppo_f5|runs_lcb_quantile|0-4"
            "lcb_quantile_ppo_f10|runs_lcb_quantile|0-4"
        )
        ;;
    ucb_quantile)
        training_jobs=(
            "ucb_quantile_ppo_f1|runs_ucb_quantile|0-4"
            "ucb_quantile_ppo_f5|runs_ucb_quantile|0-4"
            "ucb_quantile_ppo_f10|runs_ucb_quantile|0-4"
        )
        ;;
    posterior_quantile)
        training_jobs=(
            "lcb_quantile_ppo_f1|runs_post_quantile|0-4"
            "lcb_quantile_ppo_f5|runs_post_quantile|0-4"
            "lcb_quantile_ppo_f10|runs_post_quantile|0-4"
        )
        ;;
    af_selection)
        training_jobs=(
            "af_selection_ppo_f1|runs_af_selection|0-4"
            "af_selection_ppo_f5|runs_af_selection|0-4"
            "af_selection_ppo_f10|runs_af_selection|0-4"
        )
        ;;
    all)
        training_jobs=(
            "structured_ppo_f1|runs_structured|0-4"
            "structured_ppo_f5|runs_structured|0-4"
            "structured_ppo_f10|runs_structured|0-4"
            "lcb_quantile_ppo_f1|runs_lcb_quantile|0-4"
            "lcb_quantile_ppo_f5|runs_lcb_quantile|0-4"
            "lcb_quantile_ppo_f10|runs_lcb_quantile|0-4"
            "ucb_quantile_ppo_f1|runs_ucb_quantile|0-4"
            "ucb_quantile_ppo_f5|runs_ucb_quantile|0-4"
            "ucb_quantile_ppo_f10|runs_ucb_quantile|0-4"
            "af_selection_ppo_f1|runs_af_selection|0-4"
            "af_selection_ppo_f5|runs_af_selection|0-4"
            "af_selection_ppo_f10|runs_af_selection|0-4"
        )
        ;;
    stage_c)
        training_jobs=(
            "mixed_wei_double_dqn_stage_c|runs_stage_c_gp_ddqn|0-2"
            "mixed_wei_ppo_gp_stage_c|runs_stage_c_gp_ddqn|0-2"
            "mixed_wei_double_dqn_gp_stage_c|runs_stage_c_gp_ddqn|0-2"
        )
        ;;
    stage_c_ddqn)
        training_jobs=(
            "mixed_wei_double_dqn_stage_c|runs_stage_c_gp_ddqn|0-2"
            "mixed_wei_double_dqn_gp_stage_c|runs_stage_c_gp_ddqn|0-2"
        )
        ;;
    stage_c_gp)
        training_jobs=(
            "mixed_wei_ppo_gp_stage_c|runs_stage_c_gp_ddqn|0-2"
            "mixed_wei_double_dqn_gp_stage_c|runs_stage_c_gp_ddqn|0-2"
        )
        ;;
    *)
        echo "Unknown DACBO_PPO_MODE=${mode}." >&2
        usage >&2
        exit 2
        ;;
esac

for job_entry in "${training_jobs[@]}"; do
    IFS="|" read -r training_config default_run_directory array_range <<< "${job_entry}"
    run_directory="${DACBO_RUN_DIR:-${default_run_directory}}"
    echo "Launch ${training_config} seeds ${array_range} -> ${run_directory}"
    sbatch \
        --array="${array_range}" \
        --job-name="${training_config}" \
        --export="ALL,DACBO_RUN_DIR=${run_directory}" \
        scripts/opt_ppo.sh \
        "+training=${training_config}" \
        "$@"
done
