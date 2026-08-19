#!/usr/bin/env bash
#SBATCH -t 48:00:00
#SBATCH -J "ppostageb"
#SBATCH --cpus-per-task=33
#SBATCH --mem=128G
#SBATCH -p normal
#SBATCH --array=0-2
#SBATCH --output=slurmlogs/ppo/slurm-%A_%a.out
#SBATCH --error=slurmlogs/ppo/slurm-%A_%a.err

source ~/.bashrc
cd /scratch/hpc-prf-intexml/tklenke/repos/dacboenv_new
source .venv/bin/activate

RUN_SEED="${SLURM_ARRAY_TASK_ID:-0}"

export HYDRA_FULL_ERROR=1

export DACBO_YAHPO_REFERENCE_TABLE=/scratch/hpc-prf-intexml/tklenke/repos/dacboenv_new/dacboenv/experiment/analysis/yahpo_best_known_references.json

PYTHONHASHSEED=0 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
python -m dacboenv.experiment.ppo \
    "$@" \
    "seed=${RUN_SEED}" \
