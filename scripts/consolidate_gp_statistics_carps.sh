#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_directory}/.." && pwd -P)"
python_bin="${DACBO_PYTHON:-${repo_root}/.env/bin/python}"
output_root="${DACBO_GP_STATS_OUTPUT_ROOT:-${repo_root}/artifacts/gp_statistics_carps_blackbox_v1}"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd -- "${repo_root}"
exec "${python_bin}" -m dacboenv.experiment.gp_statistics_campaign consolidate \
    --inventory "${output_root}/inventory.json" \
    --output-root "${output_root}/consolidated"
