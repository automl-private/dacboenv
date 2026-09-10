#!/usr/bin/env bash
# Base-only evaluation reuses the same paired CARP-S development planner.
set -euo pipefail
if (( $# != 3 )); then echo "Usage: $0 RUN_ROOT EVAL_ROOT prepare|dry-run|submit|gather" >&2; exit 2; fi
run_root="$(realpath "$1")"; eval_root="$(realpath -m "$2")"; mode="$3"
repository="$(git rev-parse --show-toplevel)"; cd "$repository"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ -x .venv/bin/python ]] || { echo 'Otus requires .venv.' >&2; exit 2; }
case "$mode" in
  prepare)
    uv run --frozen python -m dacboenv.experiment.collect_initial_base_policies \
      "run_root=$run_root" "output_root=$eval_root/bundle"
    uv run --frozen python -m dacboenv.experiment.prepare_selective_wei_carps_eval \
      "policy_inventory=$eval_root/bundle/initial_base_policy_inventory.json" "output_root=$eval_root"
    ;;
  dry-run) exec bash "$eval_root/run_selective_carps_dev.sh" --dry-run ;;
  submit) exec bash "$eval_root/run_selective_carps_dev.sh" ;;
  gather)
    uv run --frozen python -m carps.analysis.gather_data "$eval_root/runs"
    uv run --frozen python -m carps.utils.check_missing "$eval_root/runs"
    ;;
  *) echo 'Unknown evaluation mode.' >&2; exit 2 ;;
esac
