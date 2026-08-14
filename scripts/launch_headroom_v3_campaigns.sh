#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
group="${TSUBAME_GROUP:-tga-i}"
root="${repo_root}/artifacts/headroom_predictability_v3"
reference="${DACBO_YAHPO_REFERENCE_TABLE:-${repo_root}/dacboenv/experiment/analysis/yahpo_best_known_references.json}"
record="${root}/cluster_submission.json"
if [[ -e "${record}" ]]; then
  echo "Refusing duplicate v3 submission: ${record} exists." >&2
  exit 3
fi
mkdir -p "${root}/history_shards" "${root}/static_shards" "${repo_root}/tsubamelogs/headroom-v3"
"${repo_root}/.env/bin/python" -m dacboenv.experiment.static_vbs_v3 \
  --build-inventory "${root}/static_inventory.json"
printf '{"status":"submitting","group":"%s"}\n' "${group}" > "${record}"
train_job="$(qsub -terse -g "${group}" -t 1-360 -o "${repo_root}/tsubamelogs/headroom-v3" -e "${repo_root}/tsubamelogs/headroom-v3" -v "DACBO_REPO_ROOT=${repo_root},DACBO_SNAPSHOTS=${repo_root}/artifacts/headroom_train_snapshots.parquet,DACBO_SPLIT=train,DACBO_OUTPUT_ROOT=${root}/history_shards,DACBO_YAHPO_REFERENCE_TABLE=${reference}" "${repo_root}/scripts/headroom_v3_history_tsubame.sh")"
validation_job="$(qsub -terse -g "${group}" -t 1-172 -o "${repo_root}/tsubamelogs/headroom-v3" -e "${repo_root}/tsubamelogs/headroom-v3" -v "DACBO_REPO_ROOT=${repo_root},DACBO_SNAPSHOTS=${repo_root}/artifacts/headroom_validation_snapshots.parquet,DACBO_SPLIT=validation,DACBO_OUTPUT_ROOT=${root}/history_shards,DACBO_YAHPO_REFERENCE_TABLE=${reference}" "${repo_root}/scripts/headroom_v3_history_tsubame.sh")"
static_job="$(qsub -terse -g "${group}" -t 1-500 -o "${repo_root}/tsubamelogs/headroom-v3" -e "${repo_root}/tsubamelogs/headroom-v3" -v "DACBO_REPO_ROOT=${repo_root},DACBO_STATIC_INVENTORY=${root}/static_inventory.json,DACBO_OUTPUT_ROOT=${root}/static_shards,DACBO_YAHPO_REFERENCE_TABLE=${reference}" "${repo_root}/scripts/headroom_v3_static_tsubame.sh")"
printf '{"status":"submitted","group":"%s","history_train":"%s","history_validation":"%s","static":"%s"}\n' "${group}" "${train_job}" "${validation_job}" "${static_job}" > "${record}"
cat "${record}"
