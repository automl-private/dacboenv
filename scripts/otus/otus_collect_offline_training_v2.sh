#!/usr/bin/env bash
# Training-manifest-only f5 offline collection. No Otus concurrency cap is set.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  otus_collect_offline_training_v2.sh OUTPUT_ROOT prepare
  otus_collect_offline_training_v2.sh OUTPUT_ROOT submit
  otus_collect_offline_training_v2.sh OUTPUT_ROOT status
  otus_collect_offline_training_v2.sh OUTPUT_ROOT resubmit-missing
  otus_collect_offline_training_v2.sh OUTPUT_ROOT consolidate

The campaign uses BBOB/YAHPO training manifests, five static alpha policies,
uniform-random alpha, interaction frequency five, and seeds 0..4.  The Slurm
array has no `%N` concurrency limiter.
EOF
}

if (( $# != 2 )); then usage >&2; exit 2; fi
output_input="$1"
mode="$2"
case "${mode}" in prepare|submit|status|resubmit-missing|consolidate) ;; *) usage >&2; exit 2 ;; esac

repository="$(git rev-parse --show-toplevel)"
cd "${repository}"
mkdir -p "${output_input}" slurmlogs/offline-training-v2
output="$(cd "${output_input}" && pwd -P)"
manifest="${output}/offline_training_manifest.json"
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || python_bin="${repository}/.env/bin/python"
[[ -x "${python_bin}" ]] || { echo "No repository Python environment found." >&2; exit 1; }

export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${mode}" in
  prepare)
    "${python_bin}" -m dacboenv.experiment.offline_training_v2 prepare \
      --repository "${repository}" --output-root "${output}" --seeds 0,1,2,3,4
    ;;
  submit)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    count="$("${python_bin}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["job_count"])' "${manifest}")"
    (( count > 0 )) || { echo "Empty job manifest." >&2; exit 1; }
    marker="${output}/submission.json"
    [[ ! -e "${marker}" ]] || { echo "Submission marker exists: ${marker}" >&2; exit 3; }
    printf '{"job_count":%s,"submitted_at":"%s"}\n' "${count}" "$(date -Is)" > "${marker}"
    # Intentionally no %concurrency suffix.
    sbatch --array="0-$((count - 1))" \
      scripts/otus/opt_offline_training_v2.sh "${repository}" "${manifest}"
    ;;
  status)
    "${python_bin}" -m dacboenv.experiment.offline_training_v2 status --manifest "${manifest}"
    ;;
  resubmit-missing)
    [[ -s "${manifest}" ]] || { echo "Run prepare first." >&2; exit 1; }
    indices="$("${python_bin}" - "${manifest}" <<'PYI'
import json,sys
from pathlib import Path
manifest=json.load(open(sys.argv[1]))
missing=[]
for row in manifest['jobs']:
    path=Path(row['output_path'])
    valid=False
    if path.is_file():
        try:
            import numpy as np
            with np.load(path,allow_pickle=False) as payload:
                metadata=json.loads(str(payload['metadata_json'].item()))
                valid=metadata.get('job_hash')==row['job_hash'] and payload['rewards'].shape[0]>0
        except Exception:
            valid=False
    if not valid:
        missing.append(str(row['job_index']))
print(','.join(missing))
PYI
)"
    if [[ -z "${indices}" ]]; then echo "No missing/corrupt jobs."; exit 0; fi
    sbatch --array="${indices}" \
      scripts/otus/opt_offline_training_v2.sh "${repository}" "${manifest}"
    ;;
  consolidate)
    "${python_bin}" -m dacboenv.experiment.offline_training_v2 consolidate \
      --manifest "${manifest}" \
      --destination "${output}/offline_training_f5_v2.npz"
    ;;
esac
