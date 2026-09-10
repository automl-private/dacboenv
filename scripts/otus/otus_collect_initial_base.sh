#!/usr/bin/env bash
# Native Slurm context jobs; every BO rollout uses the existing CARP-S adapter.
set -euo pipefail
if (( $# < 2 )); then echo "Usage: $0 OUTPUT_ROOT prepare|smoke|submit|status|resubmit-missing|consolidate|dry-run [PARTITION_MANIFEST] [Hydra overrides...]" >&2; exit 2; fi
output="$(realpath -m "$1")"; operation="$2"; shift 2
repository="$(git rev-parse --show-toplevel)"; cd "$repository"
[[ -x .venv/bin/python ]] || { echo 'Otus requires the repository .venv.' >&2; exit 2; }
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
manage=(uv run --frozen python -m dacboenv.experiment.initial_base_campaign "output_root=$output")
case "$operation" in
  prepare)
    (( $# >= 1 )) || { echo 'prepare requires the hashed final partition manifest.' >&2; exit 2; }
    partition="$(realpath "$1")"; shift
    exec "${manage[@]}" operation=prepare "partition_manifest=$partition" "$@" ;;
  status|consolidate|preflight|audit_reuse) exec "${manage[@]}" "operation=$operation" "$@" ;;
  smoke|submit|resubmit-missing|dry-run) ;;
  *) echo "Unknown operation $operation" >&2; exit 2 ;;
esac
command -v jq >/dev/null || { echo 'jq is required to inspect the frozen manifest.' >&2; exit 2; }
manifest="$output/manifest.json"; [[ -s "$manifest" ]] || { echo 'Prepare the campaign first.' >&2; exit 2; }
count="$(jq '.jobs|length' "$manifest")"
(( count > 0 )) || { echo 'No context jobs in manifest.' >&2; exit 2; }
indices="0-$((count-1))"
if [[ "$operation" == smoke ]]; then
  indices="$(jq -r '[([.jobs[]|select(.recipe.split=="dev" and (.recipe.task_id|startswith("bbob/")))][0].index),([.jobs[]|select(.recipe.split=="dev" and (.recipe.task_id|startswith("yahpo/")))][0].index)]|if any(.==null) then error("smoke needs dev BBOB and YAHPO") else join(",") end' "$manifest")"
elif [[ "$operation" == resubmit-missing ]]; then
  "${manage[@]}" operation=status "$@"
  indices="$(jq -r '.missing_indices|join(",")' "$output/status.json")"
  [[ -n "$indices" ]] || { echo 'No missing contexts.'; exit 0; }
fi
echo "Manifest: $manifest; contexts: $count; array: $indices; output: $output"
command=(sbatch "--array=$indices" scripts/otus/opt_initial_base.sh "$repository" "$output" "$@")
printf '%q ' "${command[@]}"; printf '\n'
[[ "$operation" == dry-run ]] && exit 0
mkdir -p slurmlogs/initial-base
"${command[@]}"
