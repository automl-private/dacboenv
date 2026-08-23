#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --job-name=d1-f5-headroom
#SBATCH --output=slurmlogs/d1-headroom/%A_%a.out
#SBATCH --error=slurmlogs/d1-headroom/%A_%a.err

set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: $0 REPOSITORY MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi
repository="$1"
manifest="$2"
output_root="$3"
cd "${repository}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || python_bin="${repository}/.env/bin/python"
[[ -x "${python_bin}" ]] || { echo "No repository Python environment found." >&2; exit 1; }
exec "${python_bin}" -m dacboenv.experiment.run_d1_headroom_job \
  --manifest "${manifest}" \
  --output-root "${output_root}/jobs" \
  --reference-table "${output_root}/yahpo_best_known_references.json" \
  --job-index "${SLURM_ARRAY_TASK_ID}"
