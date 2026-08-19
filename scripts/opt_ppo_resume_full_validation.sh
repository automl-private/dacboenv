#!/usr/bin/env bash
#SBATCH -t 48:00:00
#SBATCH -J "ppo-fullval-resume"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p normal
#SBATCH --output=slurmlogs/ppo-resume/slurm-%A_%a.out
#SBATCH --error=slurmlogs/ppo-resume/slurm-%A_%a.err

set -euo pipefail

if (( $# < 2 || $# > 3 )); then
    echo "Usage: sbatch ... scripts/opt_ppo_resume_full_validation.sh MANIFEST [delete|archive|keep] REPOSITORY_ROOT" >&2
    exit 2
fi

manifest="$1"
partial_policy="${2:-delete}"
repository_root="${3:-${DACBO_REPOSITORY_ROOT:-}}"
case "${partial_policy}" in
    delete|archive|keep) ;;
    *) echo "Unknown partial policy: ${partial_policy}" >&2; exit 2 ;;
esac
if [[ -z "${repository_root}" || ! -d "${repository_root}" ]]; then
    echo "Repository root is missing or invalid: ${repository_root}" >&2
    exit 2
fi
repository_root="$(cd "${repository_root}" && pwd -P)"
manifest="$(cd "$(dirname "${manifest}")" && pwd -P)/$(basename "${manifest}")"

python_bin="${repository_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    echo "Python executable is missing: ${python_bin}" >&2
    exit 1
fi

cd "${repository_root}"
mkdir -p slurmlogs/ppo-resume

export HYDRA_FULL_ERROR=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

array_index="${SLURM_ARRAY_TASK_ID:-0}"
run_root="$(sed -n "$((array_index + 1))p" "${manifest}")"
if [[ -z "${run_root}" ]]; then
    echo "No run root at manifest index ${array_index}: ${manifest}" >&2
    exit 1
fi

echo "Repository: ${repository_root}"
echo "Resume full validation: ${run_root}"
"${python_bin}" -m dacboenv.experiment.resume_full_validation \
    "${run_root}" \
    --partial-policy "${partial_policy}" \
    --quiet
