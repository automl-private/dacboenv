#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=dacbo-offline-q
#SBATCH --output=slurmlogs/offline-q/%A_%a.out
#SBATCH --error=slurmlogs/offline-q/%A_%a.err
set -euo pipefail
if (( $# != 4 )); then echo "Usage: $0 REPOSITORY MANIFEST FINAL_ROOT BRANCH_ROOT" >&2; exit 2; fi
repository="$1" manifest="$2" final_root="$3" branch_root="$4"
cd "${repository}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
IFS=$'\t' read -r cell seed coefficient updates output_root < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
exec uv run --frozen python -m dacboenv.experiment.offline_train \
  "+offline_training=${cell}" "offline_dataset.root=${final_root}" \
  "offline_dataset.branch_train=${branch_root}/branches_train.npz" \
  "offline_dataset.branch_dev=${branch_root}/branches_dev.npz" \
  "offline_training.cql_coefficient=${coefficient}" "offline_training.maximum_updates=${updates}" \
  "seed=${seed}" "output_root=${output_root}"
