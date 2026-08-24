#!/usr/bin/env bash
# Prepare, submit, audit, resume, and consolidate f5 all-action branches.
set -euo pipefail
if (( $# != 3 )); then
  echo "Usage: $0 FINAL_DATASET_ROOT OUTPUT_ROOT prepare|smoke|submit|status|resubmit-missing|consolidate" >&2
  exit 2
fi
final_root="$(realpath "$1")"
mkdir -p "$2"
output_root="$(realpath "$2")"
mode="$3"
repository="$(git rev-parse --show-toplevel)"
cd "${repository}"
[[ -x "${repository}/.venv/bin/python" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
manifest="${output_root}/offline_branch_job_manifest.json"
reference="${repository}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
manage() {
  uv run --frozen python -m dacboenv.experiment.offline_branch_campaign \
    "operation=$1" "final_dataset_root=${final_root}" "output_root=${output_root}"
}
audit_manifest() {
  "${repository}/.venv/bin/python" -c 'import json,sys; p=sys.argv[1]; m=json.load(open(p)); rows=m["jobs"]; print("manifest={}".format(p)); print("rows={}".format(len(rows))); print("splits={}".format(sorted({r["data_context_split"] for r in rows}))); print("seeds={}".format(sorted({r["seed"] for r in rows}))); print("phases={}".format(sorted({r["phase"] for r in rows}))); print("output_root={}".format(sys.argv[2]))' "${manifest}" "${output_root}"
}
case "${mode}" in
  prepare) manage prepare ;;
  status) manage status ;;
  consolidate) manage consolidate ;;
  submit)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    marker="${output_root}/submission.json"
    [[ ! -e "${marker}" ]] || { echo "Submission marker exists: ${marker}" >&2; exit 3; }
    count="$("${repository}/.venv/bin/python" -c 'import json,sys;print(json.load(open(sys.argv[1]))["job_count"])' "${manifest}")"
    audit_manifest
    mkdir -p slurmlogs/offline-branch
    # No %N suffix: Otus schedules every eligible array element.
    submission="$(sbatch --array="0-$((count - 1))" scripts/otus/opt_offline_branch.sh "${repository}" "${manifest}" "${reference}")"
    printf '{"job_count":%s,"submitted_at":"%s","scheduler_response":"%s"}\n' \
      "${count}" "$(date -Is)" "${submission}" > "${marker}"
    printf '%s\n' "${submission}"
    ;;
  smoke)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    marker="${output_root}/smoke_submission.json"
    [[ ! -e "${marker}" ]] || { echo "Smoke submission marker exists: ${marker}" >&2; exit 3; }
    selection="$("${repository}/.venv/bin/python" -c 'import json,sys; m=json.load(open(sys.argv[1])); rows=m["jobs"]; chosen=[]
for domain in ("bbob/", "yahpo/"):
 r=next(x for x in rows if x["data_context_split"]=="dev" and x["task_id"].startswith(domain)); chosen.append(r); print(json.dumps(r,sort_keys=True))
print(",".join(str(r["job_index"]) for r in chosen))' "${manifest}")"
    printf '%s\n' "${selection}"
    indices="$(printf '%s\n' "${selection}" | tail -n 1)"
    audit_manifest
    submission="$(sbatch --array="${indices}" scripts/otus/opt_offline_branch.sh "${repository}" "${manifest}" "${reference}")"
    printf '{"indices":"%s","submitted_at":"%s","scheduler_response":"%s"}\n' \
      "${indices}" "$(date -Is)" "${submission}" > "${marker}"
    printf '%s\n' "${submission}"
    ;;
  resubmit-missing)
    manage status >/dev/null
    indices="$("${repository}/.venv/bin/python" -c 'import json,sys;print(",".join(map(str,json.load(open(sys.argv[1]))["missing_indices"])))' "${output_root}/offline_branch_status.json")"
    [[ -n "${indices}" ]] || { echo "No missing/failed branch jobs."; exit 0; }
    audit_manifest
    submission="$(sbatch --array="${indices}" scripts/otus/opt_offline_branch.sh "${repository}" "${manifest}" "${reference}")"
    printf '{"indices":"%s","submitted_at":"%s","scheduler_response":"%s"}\n' \
      "${indices}" "$(date -Is)" "${submission}" > "${output_root}/resubmission_$(date +%s).json"
    printf '%s\n' "${submission}"
    ;;
  *) echo "Unknown mode ${mode}." >&2; exit 2 ;;
esac
