#!/usr/bin/env bash
#$ -cwd
#$ -S /bin/bash
#$ -l cpu_40=1
#$ -l h_rt=24:00:00
#$ -N ppo4dacbo
#$ -t 1-10
#$ -r n

# TSUBAME4/Altair Grid Engine worker for structured SB3 PPO.
# Submit through launch_ppo_tsubame.sh so qsub receives `-g tga-i` and an
# existing log directory. UGE array tasks 1-10 map to experiment seeds 0-9.

set -euo pipefail

if [[ -n "${DACBO_REPO_ROOT:-}" ]]; then
    repo_candidate="${DACBO_REPO_ROOT}"
elif [[ "${SGE_TASK_ID:-undefined}" != "undefined" && -n "${SGE_O_WORKDIR:-}" ]]; then
    repo_candidate="${SGE_O_WORKDIR}"
else
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
    repo_candidate="${script_dir}/.."
fi
REPO_ROOT="$(cd -- "${repo_candidate}" && pwd -P)"

if [[ ! -f "${REPO_ROOT}/pyproject.toml" || ! -d "${REPO_ROOT}/dacboenv" ]]; then
    echo "Resolved path is not the dacboenv repository: ${REPO_ROOT}" >&2
    exit 2
fi

PYTHON_BIN="${DACBO_PYTHON:-${REPO_ROOT}/.env/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment is missing or not executable: ${PYTHON_BIN}" >&2
    echo "Install the project environment before submitting the PPO matrix." >&2
    exit 2
fi

N_WORKERS="${DACBO_N_WORKERS:-32}"
if [[ ! "${N_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "DACBO_N_WORKERS must be a positive integer, got: ${N_WORKERS}" >&2
    exit 2
fi
if (( N_WORKERS > 39 )); then
    echo "DACBO_N_WORKERS=${N_WORKERS} exceeds cpu_40 capacity after reserving one PPO parent core." >&2
    exit 2
fi

if [[ "${SGE_TASK_ID:-undefined}" == "undefined" ]]; then
    RUN_SEED="${DACBO_RUN_SEED:-0}"
else
    if [[ ! "${SGE_TASK_ID}" =~ ^[0-9]+$ ]]; then
        echo "Invalid SGE_TASK_ID: ${SGE_TASK_ID}" >&2
        exit 2
    fi
    task_id=$((10#${SGE_TASK_ID}))
    if (( task_id < 1 || task_id > 10 )); then
        echo "SGE_TASK_ID must be in 1-10, got: ${SGE_TASK_ID}" >&2
        exit 2
    fi
    RUN_SEED=$((task_id - 1))
fi
if [[ ! "${RUN_SEED}" =~ ^[0-9]+$ ]]; then
    echo "Resolved experiment seed must be a non-negative integer, got: ${RUN_SEED}" >&2
    exit 2
fi

BASERUNDIR="${DACBO_RUN_DIR:-${REPO_ROOT}/runs_structured}"

# Each SB3 subprocess represents one environment worker. Keep numerical
# libraries single-threaded so 32 workers fit within the 40-core allocation.
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-${RUN_SEED}}"

cd -- "${REPO_ROOT}"
echo "PPO job ${JOB_ID:-local}.${SGE_TASK_ID:-0}: seed=${RUN_SEED}, workers=${N_WORKERS}"
echo "Repository: ${REPO_ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Runs: ${BASERUNDIR}"

exec "${PYTHON_BIN}" -m dacboenv.experiment.ppo \
    "$@" \
    "experiment.n_workers=${N_WORKERS}" \
    "seed=${RUN_SEED}" \
    "baserundir=${BASERUNDIR}"

