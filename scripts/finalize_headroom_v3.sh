#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "${repo_root}"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/dacbo-mpl-v3"
root=artifacts/headroom_predictability_v3
[[ "$(find "${root}/history_shards" -name 'train-*.json' | wc -l)" -eq 360 ]] || { echo "Training histories incomplete" >&2; exit 3; }
[[ "$(find "${root}/history_shards" -name 'validation-*.json' | wc -l)" -eq 172 ]] || { echo "Validation histories incomplete" >&2; exit 3; }
[[ "$(find "${root}/static_shards" -name '*.json' | wc -l)" -eq 500 ]] || { echo "Static panel incomplete" >&2; exit 3; }
.venv/bin/python -m dacboenv.experiment.reconstruct_histories_v3 --snapshots artifacts/headroom_train_snapshots.parquet --split train --output-root "${root}/history_shards" --consolidate
.venv/bin/python -m dacboenv.experiment.reconstruct_histories_v3 --snapshots artifacts/headroom_validation_snapshots.parquet --split validation --output-root "${root}/history_shards" --consolidate
.venv/bin/python -m dacboenv.experiment.true_history_features_v3
.venv/bin/python -m dacboenv.experiment.richer_predictability_v2 --input "${root}" --gru-seed 1103 --gru-seed 2207 --gru-seed 3301
.venv/bin/python -m dacboenv.experiment.corrected_predictability_v3
.venv/bin/python -m dacboenv.experiment.static_vbs_v3 --inventory "${root}/static_inventory.json" --output-root "${root}/static_shards" --consolidate
.venv/bin/python -m dacboenv.experiment.analyze_static_vbs_v3
.venv/bin/python -m dacboenv.experiment.render_headroom_v3
echo "V3 consolidation complete. Review reports before scientific interpretation."
