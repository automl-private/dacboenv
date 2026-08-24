#!/usr/bin/env bash
# Finalize the clean Otus f5 corpus into frozen task-disjoint views.
set -euo pipefail

if (( $# < 2 || $# > 3 )); then
  echo "Usage: $0 BEHAVIOR_NPZ OUTPUT_ROOT [LEARNED_HEADROOM_ROOT]" >&2
  exit 2
fi
repository="$(git rev-parse --show-toplevel)"
source_path="$(realpath "$1")"
mkdir -p "$2"
output_root="$(realpath "$2")"
headroom_root="${3:-}"
cd "${repository}"
[[ -x "${repository}/.venv/bin/python" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
command=(uv run --frozen python -m dacboenv.experiment.finalize_offline_training_dataset
  "behavior_npz=${source_path}" "output_root=${output_root}")
if [[ -n "${headroom_root}" ]]; then command+=("learned_headroom_root=$(realpath "${headroom_root}")"); fi
exec "${command[@]}"
