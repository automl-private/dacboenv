#!/usr/bin/env bash
#SBATCH -t 48:00:00
#SBATCH -J "ppo4dacbo"
#SBATCH --cpus-per-task=33
#SBATCH --mem=64G
#SBATCH -p normal
#SBATCH --array=0-9
#SBATCH --output=slurmlogs/ppo/slurm-%A_%a.out
#SBATCH --error=slurmlogs/ppo/slurm-%A_%a.err

set -euo pipefail

N_WORKERS="${DACBO_N_WORKERS:-32}"
BASERUNDIR="${DACBO_RUN_DIR:-runs_structured}"
RUN_SEED="${SLURM_ARRAY_TASK_ID:-0}"

export HYDRA_FULL_ERROR=1

python -m dacboenv.experiment.ppo \
    "$@" \
    "experiment.n_workers=${N_WORKERS}" \
    "seed=${RUN_SEED}" \
    "baserundir=${BASERUNDIR}"
