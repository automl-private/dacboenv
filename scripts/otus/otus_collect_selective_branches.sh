#!/usr/bin/env bash
# Prepare or manage versioned base/selective-history branch campaigns.
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 FINAL_ROOT OUTPUT_ROOT MODE [REGISTRY HASH|POLICY_BUNDLE]" >&2; exit 2
fi
final_root="$(realpath "$1")"; mkdir -p "$2"; output_root="$(realpath "$2")"; mode="$3"; shift 3
repository="$(git rev-parse --show-toplevel)"; cd "${repository}"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
reference="${repository}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
campaign="${mode##*-}"; operation="${mode%%-*}"
case "${mode}" in
  prepare-base)
    if (( $# == 1 )); then
      uv run --frozen python -m dacboenv.experiment.selective_branch_campaign operation=prepare campaign=base \
        "final_dataset_root=${final_root}" "output_root=${output_root}" "initial_base_bundle=$(realpath "$1")"
      exit 0
    fi
    (( $# == 2 )) || { echo "prepare-base requires REGISTRY REGISTRY_HASH." >&2; exit 2; }
    uv run --frozen python -m dacboenv.experiment.selective_branch_campaign operation=prepare campaign=base \
      "final_dataset_root=${final_root}" "output_root=${output_root}" \
      "base_selector_registry=$(realpath "$1")" "base_selector_registry_hash=$2"
    exit 0 ;;
  prepare-selective)
    (( $# == 1 )) || { echo "prepare-selective requires POLICY_BUNDLE." >&2; exit 2; }
    uv run --frozen python -m dacboenv.experiment.selective_branch_campaign operation=prepare campaign=selective \
      "final_dataset_root=${final_root}" "output_root=${output_root}" "policy_bundle=$(realpath "$1")" \
      "base_selector_registry=null" "base_selector_registry_hash=null"
    exit 0 ;;
  submit-base|submit-selective|resubmit-base|resubmit-selective|status-base|status-selective|consolidate-base|consolidate-selective) ;;
  *) echo "Unknown mode ${mode}." >&2; exit 2 ;;
esac
manifest="${output_root}/selective_${campaign}_branch_manifest.json"
[[ -s "${manifest}" ]] || { echo "Run prepare-${campaign} first." >&2; exit 1; }
manage() {
  uv run --frozen python -m dacboenv.experiment.selective_branch_campaign \
    "operation=$1" "campaign=${campaign}" "final_dataset_root=${final_root}" "output_root=${output_root}" \
    base_selector_registry=null base_selector_registry_hash=null policy_bundle=null
}
case "${operation}" in
  status) manage status ;;
  consolidate) manage consolidate ;;
  submit)
    count="$(.venv/bin/python -c 'import json,sys;print(json.load(open(sys.argv[1]))["job_count"])' "${manifest}")"
    .venv/bin/python -c 'import json,sys;m=json.load(open(sys.argv[1])); print("manifest={}".format(sys.argv[1])); print("rows={}".format(len(m["jobs"]))); print("campaign={}".format(m["campaign"])); print("output_root={}".format(sys.argv[2]))' "${manifest}" "${output_root}"
    mkdir -p slurmlogs/selective-branch
    # No %N suffix: Otus schedules every eligible array element.
    sbatch --array="0-$((count - 1))" scripts/otus/opt_selective_branch.sh "${repository}" "${manifest}" "${reference}"
    ;;
  resubmit)
    manage status >/dev/null
    status_path="${output_root}/selective_${campaign}_branch_status.json"
    indices="$(.venv/bin/python -c 'import json,sys;print(",".join(map(str,json.load(open(sys.argv[1]))["missing_indices"])))' "${status_path}")"
    [[ -n "${indices}" ]] || { echo "No missing jobs."; exit 0; }
    sbatch --array="${indices}" scripts/otus/opt_selective_branch.sh "${repository}" "${manifest}" "${reference}"
    ;;
esac
