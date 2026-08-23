#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --job-name=dacbo-offline-f5
#SBATCH --output=slurmlogs/offline-training-v2/%A_%a.out
#SBATCH --error=slurmlogs/offline-training-v2/%A_%a.err

set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 REPOSITORY MANIFEST" >&2
  exit 2
fi
repository="$1"
manifest="$2"
cd "${repository}"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || python_bin="${repository}/.env/bin/python"
[[ -x "${python_bin}" ]] || { echo "No repository Python environment found." >&2; exit 1; }

exec "${python_bin}" -m dacboenv.experiment.offline_training_v2 worker \
  --manifest "${manifest}" \
  --job-index "${SLURM_ARRAY_TASK_ID}"
