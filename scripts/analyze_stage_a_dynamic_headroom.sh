#!/usr/bin/env bash
set -euo pipefail

export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

usage() {
    echo "Usage: $0 RUN_ROOT {best|final} SNAPSHOTS_JSONL BRANCHES_CSV [OUTPUT_JSON]" >&2
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
    usage
    exit 2
fi

run_root=$1
checkpoint=$2
snapshots=$3
branches=$4
output=${5:-${run_root}/analysis/dynamic_headroom_${checkpoint}.json}

if [[ ! -d ${run_root} ]]; then
    echo "SKIP: Stage-A run root does not exist: ${run_root}"
    exit 0
fi
if [[ ! -f ${run_root}/.hydra/config.yaml ]]; then
    echo "SKIP: incomplete Stage-A run (missing .hydra/config.yaml): ${run_root}"
    exit 0
fi
case ${checkpoint} in
    best)
        if [[ ! -f ${run_root}/validation/best_balanced_model.zip && ! -f ${run_root}/validation/best_model.zip ]]; then
            echo "SKIP: incomplete best checkpoint below ${run_root}"
            exit 0
        fi
        ;;
    final)
        if [[ ! -f ${run_root}/model.zip && ! -f ${run_root}/final_model.zip ]]; then
            echo "SKIP: incomplete final checkpoint below ${run_root}"
            exit 0
        fi
        ;;
    *)
        usage
        exit 2
        ;;
esac
if [[ ! -f ${snapshots} || ! -f ${branches} ]]; then
    echo "SKIP: saved validation snapshots or branch outcomes are incomplete"
    exit 0
fi

.env/bin/python -m dacboenv.experiment.analyze_snapshot_policy \
    --run-root "${run_root}" \
    --checkpoint "${checkpoint}" \
    --snapshots "${snapshots}" \
    --branches "${branches}" \
    --factory dacboenv.experiment.real_env:real_structured_bbob_env \
    --validation-manifest dacboenv/configs/instance_sets/bbob_validation.yaml \
    --forbidden-task-ids dacboenv/configs/instance_sets/bbob_test_strict.yaml \
    --output "${output}"
