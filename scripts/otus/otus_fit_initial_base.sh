#!/usr/bin/env bash
set -euo pipefail
if (( $# < 3 )); then echo "Usage: $0 DATA_ROOT RUN_ROOT ablations|dry-run [Hydra overrides...]" >&2; exit 2; fi
data="$(realpath "$1")"; output="$(realpath -m "$2")"; mode="$3"; shift 3
repository="$(git rev-parse --show-toplevel)"; cd "$repository"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
[[ -x .venv/bin/python ]] || { echo 'Otus requires .venv.' >&2; exit 2; }
[[ "$mode" == ablations || "$mode" == dry-run ]] || { echo 'Unknown fitting mode.' >&2; exit 2; }
[[ -s "$data/initial_base_train.json" && -s "$data/initial_base_dev.json" ]] || { echo 'Consolidate the complete campaign first.' >&2; exit 2; }
echo "12 supervised jobs: four feature ablations, seeds 0/1/2, extra_trees (override model=ridge for diagnostic)."
source_revision="$(uv run --frozen python -c 'from dacboenv.experiment.source_provenance import current_source_revision; print(current_source_revision())')"
command=(sbatch --array=0-11 scripts/otus/opt_fit_initial_base.sh "$repository" "$data" "$output" "expected_source_revision=$source_revision" "$@")
printf '%q ' "${command[@]}"; printf '\n'
[[ "$mode" == dry-run ]] && exit 0
mkdir -p slurmlogs/initial-base
"${command[@]}"
