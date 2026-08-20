#!/usr/bin/env bash
# Build, submit, audit, and consolidate the CARP-S offline dataset on Otus.

set -euo pipefail
set -f

usage() {
    cat <<'EOF'
Usage:
  otus_collect_offline_dataset.sh OUTPUT_ROOT MODE [--dry-run]

Modes:
  prepare        Freeze the 2,380-episode inventory and 14 Hydra launch groups.
  all            Submit all static, double-random, and SAWEI launch groups.
  static         Submit only the five static WEI-alpha policies.
  double-random  Submit only the double-random policy.
  sawei          Submit only native SAWEI (duration 1).
  status         Validate every expected NPZ shard and report missing/failures.
  consolidate    Build one flat-transition NPZ after all shards validate.

The collection modes invoke `python -m carps.run -m`, Hydra's Submitit Slurm
launcher, the repository's `cpu_noctua` profile, native CARP-S task budgets,
and seeds 0..4. They never run collection work on the login process.
EOF
}

if (( $# < 2 )); then
    usage >&2
    exit 2
fi

output_root_input="$1"
mode="$2"
shift 2
dry_run=0
while (( $# > 0 )); do
    case "$1" in
        --dry-run)
            dry_run=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

case "${mode}" in
    prepare|all|static|double-random|sawei|status|consolidate) ;;
    *)
        echo "Unknown mode: ${mode}" >&2
        usage >&2
        exit 2
        ;;
esac

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"
python_bin="${repo_root}/.venv/bin/python"
if [[ ! -x "${python_bin}" ]]; then
    echo "The repository runtime is missing: ${python_bin}" >&2
    echo "Create/synchronize .venv before preparing the Otus collection." >&2
    exit 1
fi
mkdir -p -- "${output_root_input}"
output_root="$(cd "${output_root_input}" && pwd -P)"
reference_table="${repo_root}/dacboenv/experiment/analysis/yahpo_best_known_references.json"
campaign_config="carps_bbob_yahpo_v1"

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export HYDRA_FULL_ERROR=1

manage() {
    local operation="$1"
    shift
    "${python_bin}" -m dacboenv.experiment.offline_dataset_campaign \
        "offline_dataset.operation=${operation}" \
        "offline_dataset.output_root=${output_root}" \
        "offline_dataset.reference_table=${reference_table}" \
        "hydra.run.dir=${output_root}/management/${operation}" \
        hydra.job.chdir=false \
        "$@"
}

if [[ "${mode}" == "prepare" ]]; then
    manage build
    echo "Inventory: ${output_root}/inventory.json"
    echo "Launch groups: ${output_root}/launch_groups.tsv"
    exit 0
fi
if [[ "${mode}" == "status" ]]; then
    manage status
    exit 0
fi
if [[ "${mode}" == "consolidate" ]]; then
    manage consolidate
    exit 0
fi

manage build >/dev/null
inventory="${output_root}/inventory.json"
launch_groups="${output_root}/launch_groups.tsv"
if [[ ! -s "${inventory}" || ! -s "${launch_groups}" ]]; then
    echo "Offline inventory preparation failed." >&2
    exit 1
fi

if [[ "${mode}" == "all" ]]; then
    if find "${output_root}/submissions" -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
        echo "Refusing an overlapping submission: a marker already exists under ${output_root}/submissions." >&2
        exit 3
    fi
elif [[ -e "${output_root}/submissions/${mode}.json" || -e "${output_root}/submissions/all.json" ]]; then
    echo "Refusing duplicate submission for mode ${mode}." >&2
    exit 3
fi

if (( dry_run == 0 )); then
    # Claim before the first scheduler submission. If this shell is interrupted,
    # the immutable marker prevents an accidental duplicate array launch.
    manage record_submission "offline_dataset.launch_mode=${mode}" >/dev/null
fi

while IFS=$'\t' read -r \
    policy_id policy_kind task_group task_configs action_config interaction_config \
    observation_config policy_override extra_overrides_json
do
    if [[ "${policy_id}" == "policy_id" ]]; then
        continue
    fi
    case "${mode}:${policy_kind}" in
        all:*|static:static|double-random:double_random|sawei:sawei) ;;
        *) continue ;;
    esac

    extra_overrides=()
    if [[ "${extra_overrides_json}" != "[]" ]]; then
        mapfile -t extra_overrides < <(
            "${python_bin}" -c \
                'import json,sys; print("\n".join(json.loads(sys.argv[1])))' \
                "${extra_overrides_json}"
        )
    fi
    command=(
        "${python_bin}" -m carps.run -m
        "hydra.searchpath=[pkg://dacboenv.configs]"
        "+task/${task_group}=${task_configs}"
        "+offline_dataset=${campaign_config}"
        +eval=offline_dataset
        +env=base
        +env/opt=base
        "+env/action=${action_config}"
        "+env/interaction_freq=${interaction_config}"
        "+env/obs=${observation_config}"
        +env/reward=reference_regret_improvement
        +env/reference_provider=offline_dataset
        "${policy_override}"
        +cluster=cpu_noctua
        "seed=range(0,5)"
        "optimizer_id=${policy_id}"
        "policy_id=${policy_id}"
        "baserundir=${output_root}/runs"
        "offline_dataset.output_root=${output_root}"
        "offline_dataset.reference_table=${reference_table}"
        "hydra.sweep.dir=${output_root}/runs"
        'hydra.sweep.subdir=${optimizer_id}/${benchmark_id}/${task_id}/${seed}'
        hydra.sweeper.max_batch_size=500
        dacboenv.evaluation_mode=false
        dacboenv.terminate_after_reference_performance_reached=false
        dacboenv.context_split=test
        dacboenv.yahpo_training_budget_multiplier=1.0
        hydra.job.chdir=false
    )
    command+=("${extra_overrides[@]}")

    echo "Policy=${policy_id} tasks=${task_group} count=$(tr ',' '\n' <<< "${task_configs}" | wc -l) seeds=5"
    if (( dry_run == 1 )); then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done < "${launch_groups}"

if (( dry_run == 1 )); then
    echo "Dry-run only; no jobs submitted."
else
    manage record_launch_completion "offline_dataset.launch_mode=${mode}" >/dev/null
    echo "Hydra Submitit launchers completed. Monitor with your normal Slurm tools."
    echo "Dataset status: bash scripts/otus/otus_collect_offline_dataset.sh ${output_root} status"
fi
