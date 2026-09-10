#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=dacbo-selective-branch
#SBATCH --output=slurmlogs/selective-branch/%A_%a.out
#SBATCH --error=slurmlogs/selective-branch/%A_%a.err
set -euo pipefail
if (( $# != 3 )); then echo "Usage: $0 REPOSITORY MANIFEST REFERENCE_TABLE" >&2; exit 2; fi
cd "$1"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec uv run --frozen python -m dacboenv.experiment.run_selective_branch_job \
  "manifest=$2" "job_index=${SLURM_ARRAY_TASK_ID}" "reference_table=$3"
