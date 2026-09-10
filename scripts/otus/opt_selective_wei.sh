#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=dacbo-selective-wei
#SBATCH --output=slurmlogs/selective-wei/%A_%a.out
#SBATCH --error=slurmlogs/selective-wei/%A_%a.err
set -euo pipefail
if (( $# < 3 )); then echo "Usage: $0 REPOSITORY MANIFEST BRANCH_ROOT [Hydra overrides...]" >&2; exit 2; fi
repository="$1" manifest="$2" branch_root="$3"
shift 3
cd "${repository}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
IFS=$'\t' read -r policy seed predictor output_root < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}")
exec uv run --frozen python -m dacboenv.experiment.fit_selective_wei \
  "+selective_policy=${policy}" "selective_predictor=${predictor}" \
  "selective_data.branch_root=${branch_root}" "selective_output.root=${output_root}" "seed=${seed}" "$@"
