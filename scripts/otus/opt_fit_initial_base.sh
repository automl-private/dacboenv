#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=dacbo-fit-initial
#SBATCH --output=slurmlogs/initial-base/%A_%a.out
#SBATCH --error=slurmlogs/initial-base/%A_%a.err
set -euo pipefail
repository="$1"; data="$2"; output="$3"; shift 3; cd "$repository"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
variants=(base_metadata base_initial_data base_initial_gp base_initial_gp_ubr)
index="${SLURM_ARRAY_TASK_ID}"; variant="${variants[$((index/3))]}"; seed="$((index%3))"
exec uv run --frozen python -m dacboenv.experiment.fit_initial_base "dataset_root=$data" \
  "output_root=$output/$variant/seed$seed" "features=$variant" "seed=$seed" "$@"
