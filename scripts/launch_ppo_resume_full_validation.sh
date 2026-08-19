#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/launch_ppo_resume_full_validation.sh RUNS_ROOT [OPTIONS]

Options:
  --dry-run                  Print the manifest and sbatch command only.
  --exclude OPTIMIZER:SEED   Exclude one still-running job; repeatable.
  --partition NAME           Slurm partition (default: normal).
  --time HH:MM:SS            Wall time (default: 48:00:00).
  --cpus N                   CPUs per task (default: 4).
  --mem SIZE                 Memory per task (default: 32G).
  --max-parallel N           Concurrent array tasks (default: 3).
  --partial-policy P         delete, archive, or keep (default: delete).

The launcher discovers every training-complete run below RUNS_ROOT and submits
only runs without validation/full/selection.json.  It never resumes learning.
EOF
}

if (( $# < 1 )); then usage >&2; exit 2; fi
runs_root="$1"; shift
runs_root="$(cd "${runs_root}" && pwd -P)"

dry_run=0
partition=normal
time_limit=48:00:00
cpus=4
mem=32G
max_parallel=1000
partial_policy=delete
exclusions=()
while (( $# )); do
    case "$1" in
        --dry-run) dry_run=1; shift ;;
        --exclude) exclusions+=("$2"); shift 2 ;;
        --partition) partition="$2"; shift 2 ;;
        --time) time_limit="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --mem) mem="$2"; shift 2 ;;
        --max-parallel) max_parallel="$2"; shift 2 ;;
        --partial-policy) partial_policy="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "${partial_policy}" in delete|archive|keep) ;; *) echo "Invalid partial policy" >&2; exit 2;; esac

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
work_root="${runs_root}/_resume_full_validation"
mkdir -p "${work_root}" "${repository_root}/slurmlogs/ppo-resume"
manifest="${work_root}/incomplete_runs.txt"
: > "${manifest}"

is_excluded() {
    local optimizer="$1" seed="$2" item
    for item in "${exclusions[@]}"; do
        if [[ "${item}" == "${optimizer}:${seed}" ]]; then return 0; fi
    done
    return 1
}

while IFS= read -r completion; do
    run_root="${completion%/training_complete.json}"
    if [[ -s "${run_root}/validation/full/selection.json" ]]; then
        echo "Skip complete: ${run_root}"
        continue
    fi
    seed="$(basename "${run_root}")"
    relative="${run_root#${runs_root}/}"
    optimizer="${relative%%/*}"
    if is_excluded "${optimizer}" "${seed}"; then
        echo "Exclude active run: ${optimizer}:${seed}"
        continue
    fi
    printf '%s\n' "${run_root}" >> "${manifest}"
done < <(find "${runs_root}" -name training_complete.json -type f -print | sort)

n_runs="$(grep -c . "${manifest}" || true)"
echo "Incomplete training-complete runs: ${n_runs}"
cat "${manifest}"
if (( n_runs == 0 )); then
    echo "Nothing to resume."
    exit 0
fi

array_spec="0-$((n_runs - 1))%${max_parallel}"
command=(
    sbatch
    --array="${array_spec}"
    --partition="${partition}"
    --time="${time_limit}"
    --cpus-per-task="${cpus}"
    --mem="${mem}"
    --chdir="${repository_root}"
    --output="${repository_root}/slurmlogs/ppo-resume/slurm-%A_%a.out"
    --error="${repository_root}/slurmlogs/ppo-resume/slurm-%A_%a.err"
    "${repository_root}/scripts/opt_ppo_resume_full_validation.sh"
    "${manifest}"
    "${partial_policy}"
    "${repository_root}"
)
printf 'Submit command: '; printf '%q ' "${command[@]}"; printf '\n'
if (( dry_run == 0 )); then
    "${command[@]}"
fi
