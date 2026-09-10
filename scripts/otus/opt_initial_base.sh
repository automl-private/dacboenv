#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=dacbo-initial-base
#SBATCH --output=slurmlogs/initial-base/%A_%a.out
#SBATCH --error=slurmlogs/initial-base/%A_%a.err
set -euo pipefail
repository="$1"; output="$2"; shift 2; cd "$repository"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec uv run --frozen python -m dacboenv.experiment.initial_base_campaign operation=run \
  "output_root=$output" "job_index=${SLURM_ARRAY_TASK_ID}" "$@"
