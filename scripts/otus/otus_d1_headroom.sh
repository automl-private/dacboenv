#!/usr/bin/env bash
# Prepare, submit, audit, and consolidate D1 learned-policy headroom on Otus.
# The Slurm array intentionally has no concurrency cap.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  otus_d1_headroom.sh OUTPUT_ROOT prepare TRAINING_ROOT [TRAINING_ROOT ...]
  otus_d1_headroom.sh OUTPUT_ROOT submit
  otus_d1_headroom.sh OUTPUT_ROOT status
  otus_d1_headroom.sh OUTPUT_ROOT resubmit-missing
  otus_d1_headroom.sh OUTPUT_ROOT consolidate

`prepare` recursively discovers complete final WEI/DDQN f5 runs. By default,
optimizer IDs containing "small" are excluded; pass only the full D1 root for
the principal analysis. The campaign uses frozen non-test frequent-validation
panels, snapshots at 25/50/75%, and branches all five actions at H=5 and H=10.
EOF
}

if (( $# < 2 )); then usage >&2; exit 2; fi
output_input="$1"
mode="$2"
shift 2
case "${mode}" in prepare|submit|status|resubmit-missing|consolidate) ;; *) usage >&2; exit 2 ;; esac

repository="$(git rev-parse --show-toplevel)"
cd "${repository}"
mkdir -p "${output_input}" slurmlogs/d1-headroom
output="$(cd "${output_input}" && pwd -P)"
manifest="${output}/d1_headroom_job_manifest.json"
registry="${repository}/artifacts/stageb_followup/nonfeedback_selector_registry.json"
reference="${repository}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || python_bin="${repository}/.env/bin/python"
[[ -x "${python_bin}" ]] || { echo "No repository Python environment found." >&2; exit 1; }
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${mode}" in
  prepare)
    (( $# >= 1 )) || { echo "prepare requires at least one TRAINING_ROOT." >&2; exit 2; }
    [[ -s "${registry}" ]] || { echo "Missing frozen selector registry: ${registry}" >&2; exit 1; }
    [[ -s "${reference}" ]] || { echo "Missing YAHPO reference table: ${reference}" >&2; exit 1; }
    command=("${python_bin}" -m dacboenv.experiment.d1_headroom_campaign prepare
      --repository "${repository}" --output-root "${output}"
      --selector-registry "${registry}" --reference-table "${reference}")
    for root in "$@"; do command+=(--training-root "${root}"); done
    "${command[@]}"
    ;;
  submit)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    count="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_count"])' "${manifest}")"
    (( count > 0 )) || { echo "Empty headroom manifest." >&2; exit 1; }
    marker="${output}/submission.json"
    [[ ! -e "${marker}" ]] || { echo "Submission marker exists: ${marker}" >&2; exit 3; }
    printf '{"job_count":%s,"submitted_at":"%s"}\n' "${count}" "$(date -Is)" > "${marker}"
    # Intentionally no %N suffix: Otus may run every eligible array element.
    sbatch --array="0-$((count - 1))" \
      scripts/otus/opt_d1_headroom.sh "${repository}" "${manifest}" "${output}"
    ;;
  status)
    "${python_bin}" -m dacboenv.experiment.d1_headroom_campaign status --output-root "${output}"
    ;;
  resubmit-missing)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    indices="$("${python_bin}" - "${output}" <<'PYI'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
manifest=json.load(open(root/'d1_headroom_job_manifest.json'))
missing=[]
for row in manifest['jobs']:
    path=root/'jobs'/f"{row['job_index']:05d}.json"
    valid=False
    if path.is_file():
        try:
            payload=json.load(open(path))
            valid=payload.get('status')=='success' and payload.get('job_hash')==row['job_hash']
        except Exception:
            valid=False
    if not valid:
        missing.append(str(row['job_index']))
print(','.join(missing))
PYI
)"
    if [[ -z "${indices}" ]]; then echo "No missing/corrupt jobs."; exit 0; fi
    sbatch --array="${indices}" \
      scripts/otus/opt_d1_headroom.sh "${repository}" "${manifest}" "${output}"
    ;;
  consolidate)
    "${python_bin}" -m dacboenv.experiment.d1_headroom_campaign status --output-root "${output}"
    "${python_bin}" -m dacboenv.experiment.consolidate_d1_headroom --output-root "${output}"
    ;;
esac
