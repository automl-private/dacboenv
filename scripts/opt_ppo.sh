#!/usr/bin/env bash
#SBATCH -t 48:00:00
#SBATCH -J "ppo4dacbo"
#SBATCH --cpus-per-task=33
#SBATCH --mem=128G
#SBATCH -p normal
#SBATCH --array=0-2
#SBATCH --output=slurmlogs/ppo/slurm-%A_%a.out
#SBATCH --error=slurmlogs/ppo/slurm-%A_%a.err

source ~/.bashrc
cd /scratch/hpc-prf-intexml/tklenke/repos/dacboenv_new
source .venv/bin/activate

N_WORKERS="${DACBO_N_WORKERS:-32}"
BASERUNDIR="${DACBO_RUN_DIR:-runs_stagec}"
RUN_SEED="${SLURM_ARRAY_TASK_ID:-0}"

# Reproducible scientific process contract. The shared RL runner derives the
# policy/network and BO-worker streams from RUN_SEED.
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

exec python -m dacboenv.experiment.rl \
    "$@" \
    "experiment.n_workers=${N_WORKERS}" \
    "seed=${RUN_SEED}" \
    "baserundir=${BASERUNDIR}"
