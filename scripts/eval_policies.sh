#!/usr/bin/env bash

# Protocol-v2 unified evaluation entry point. This intentionally requires a
# pre-generated, hashed context inventory and a fresh output namespace.
set -euo pipefail
script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"
repository_root="$(cd -- "${script_directory}/.." && pwd -P)"
cd -- "${repository_root}"

: "${DACBO_EVAL_MANIFEST:?Set DACBO_EVAL_MANIFEST to a frozen manifest YAML}"
: "${DACBO_EVAL_CONTEXTS:?Set DACBO_EVAL_CONTEXTS to its protocol-v2 context JSON}"
: "${DACBO_EVAL_OUTPUT_ROOT:?Set DACBO_EVAL_OUTPUT_ROOT to a fresh output directory}"

python_bin="${DACBO_PYTHON:-${repository_root}/.env/bin/python}"
methods="${DACBO_EVAL_METHODS:-learned_validation_selected}"
action_family="${DACBO_ACTION_FAMILY:-wei}"
arguments=(
    --manifest "${DACBO_EVAL_MANIFEST}"
    --contexts "${DACBO_EVAL_CONTEXTS}"
    --output-dir "${DACBO_EVAL_OUTPUT_ROOT}"
    --action-family "${action_family}"
    --methods
)
read -r -a method_array <<< "${methods}"
arguments+=("${method_array[@]}")
if [[ -n "${DACBO_STAGE_A_RUN_ROOT:-}" ]]; then
    arguments+=(--run-root "${DACBO_STAGE_A_RUN_ROOT}")
fi
if [[ -n "${DACBO_CONTROL_PROVENANCE:-}" ]]; then
    arguments+=(--control-provenance "${DACBO_CONTROL_PROVENANCE}")
fi
if [[ -n "${DACBO_STATIC_SELECTION_PROVENANCE:-}" ]]; then
    arguments+=(--static-selection-provenance "${DACBO_STATIC_SELECTION_PROVENANCE}")
fi
if [[ -n "${DACBO_REFERENCE_TABLE:-}" ]]; then
    arguments+=(--reference-table "${DACBO_REFERENCE_TABLE}")
fi
if [[ "${DACBO_ALLOW_SEALED_TEST:-0}" == "1" ]]; then
    arguments+=(--allow-sealed-test)
fi

exec "${python_bin}" -m dacboenv.experiment.unified_evaluator "${arguments[@]}"
