#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=dacbo-branch-f5
#SBATCH --output=slurmlogs/offline-branch/%A_%a.out
#SBATCH --error=slurmlogs/offline-branch/%A_%a.err
set -euo pipefail
if (( $# != 3 )); then echo "Usage: $0 REPOSITORY MANIFEST REFERENCE_TABLE" >&2; exit 2; fi
repository="$1" manifest="$2" reference_table="$3"
cd "${repository}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec uv run --frozen python -m dacboenv.experiment.run_offline_branch_job \
  "manifest=${manifest}" "job_index=${SLURM_ARRAY_TASK_ID}" "reference_table=${reference_table}"
