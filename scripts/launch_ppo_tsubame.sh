#!/usr/bin/env bash

# Submit the structured PPO matrix to TSUBAME4's Altair Grid Engine.
# TSUBAME requires the group to be supplied to qsub (not as a #$ directive).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
JOB_SCRIPT="${SCRIPT_DIR}/opt_ppo_tsubame.sh"
TSUBAME_GROUP="${TSUBAME_GROUP:-tga-i}"
LOG_DIR="${DACBO_LOG_DIR:-${REPO_ROOT}/tsubamelogs/ppo}"

if [[ -z "${TSUBAME_GROUP}" ]]; then
    echo "TSUBAME_GROUP must not be empty." >&2
    exit 2
fi
if ! command -v qsub >/dev/null 2>&1; then
    echo "qsub is unavailable; run this launcher on a TSUBAME login node." >&2
    exit 127
fi
if [[ ! -r "${JOB_SCRIPT}" ]]; then
    echo "Missing TSUBAME worker script: ${JOB_SCRIPT}" >&2
    exit 2
fi

if [[ "${LOG_DIR}" != /* ]]; then
    LOG_DIR="${REPO_ROOT}/${LOG_DIR}"
fi
mkdir -p -- "${LOG_DIR}"
LOG_DIR="$(cd -- "${LOG_DIR}" && pwd -P)"

training_configs=(
    "structured_ppo_f1"
    "structured_ppo_f5"
    "structured_ppo_f10"
)

# Export only the repository and supported runtime overrides. Avoid qsub -V:
# an activated login-node environment can otherwise leak into compute jobs.
qsub_args=(
    -terse
    -g "${TSUBAME_GROUP}"
    -o "${LOG_DIR}"
    -e "${LOG_DIR}"
    -v "DACBO_REPO_ROOT=${REPO_ROOT}"
)
for variable_name in DACBO_N_WORKERS DACBO_RUN_DIR DACBO_PYTHON; do
    if [[ -v "${variable_name}" ]]; then
        qsub_args+=(-v "${variable_name}=${!variable_name}")
    fi
done

cd -- "${REPO_ROOT}"
for training_config in "${training_configs[@]}"; do
    submission_id="$(
        qsub \
            "${qsub_args[@]}" \
            "${JOB_SCRIPT}" \
            "+training=${training_config}" \
            "$@"
    )"
    echo "Submitted ${training_config}: ${submission_id}"
done

