#!/usr/bin/env bash
# Prepare, inspect, submit, or gather frozen offline-dev CARP-S evaluations.
set -euo pipefail
if (( $# != 3 )); then echo "Usage: $0 OFFLINE_RUN_ROOT EVAL_ROOT prepare|dry-run|submit|gather" >&2; exit 2; fi
run_root="$(realpath "$1")"; mkdir -p "$2"; eval_root="$(realpath "$2")"; mode="$3"
repository="$(git rev-parse --show-toplevel)"; cd "${repository}"
inventory="${eval_root}/bundle/offline_policy_inventory.json"
case "${mode}" in
  prepare)
    uv run --frozen python -m dacboenv.experiment.collect_offline_policies \
      "run_root=${run_root}" "output_root=${eval_root}/bundle" checkpoint=best_branch_dev
    uv run --frozen python -m dacboenv.experiment.prepare_offline_carps_evaluation \
      "policy_inventory=${inventory}" "output_root=${eval_root}"
    ;;
  dry-run)
    [[ -x "${eval_root}/run_carps_dev_evaluation.sh" ]] || { echo "Run prepare first." >&2; exit 1; }
    exec bash "${eval_root}/run_carps_dev_evaluation.sh" --dry-run
    ;;
  submit)
    [[ -x "${eval_root}/run_carps_dev_evaluation.sh" ]] || { echo "Run prepare first." >&2; exit 1; }
    marker="${eval_root}/submission.json"
    [[ ! -e "${marker}" ]] || { echo "Submission marker exists: ${marker}" >&2; exit 3; }
    printf '{"submitted_at":"%s"}\n' "$(date -Is)" > "${marker}"
    exec bash "${eval_root}/run_carps_dev_evaluation.sh"
    ;;
  gather)
    [[ -d "${eval_root}/runs" ]] || { echo "CARP-S result root is missing: ${eval_root}/runs" >&2; exit 1; }
    uv run --frozen python -m carps.analysis.gather_data "${eval_root}/runs"
    uv run --frozen python -m carps.utils.check_missing "${eval_root}/runs"
    ;;
  *) echo "Unknown mode ${mode}." >&2; exit 2 ;;
esac
