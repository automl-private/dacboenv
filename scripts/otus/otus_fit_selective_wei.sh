#!/usr/bin/env bash
# Freeze and submit selective-WEI fitting cells; no scheduler concurrency cap.
set -euo pipefail
if (( $# < 3 )); then
  echo "Usage: $0 BRANCH_ROOT RUN_ROOT smoke|primary|all_gates|trees|conformal|multi_horizon|trust_ablation" >&2
  exit 2
fi
branch_root="$(realpath "$1")"; mkdir -p "$2"; run_root="$(realpath "$2")"; mode="$3"
shift 3
repository="$(git rev-parse --show-toplevel)"; cd "${repository}"
export UV_PROJECT_ENVIRONMENT="${repository}/.venv"
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
manifest="${run_root}/selective_fit_${mode}.tsv"
[[ ! -e "${manifest}" ]] || { echo "Manifest exists; refusing duplicate launch: ${manifest}" >&2; exit 3; }
printf 'policy\tseed\tpredictor\toutput_path\n' > "${manifest}"
add_row() {
  local policy="$1" seed="$2" predictor="$3"
  printf '%s\t%s\t%s\t%s\n' "${policy}" "${seed}" "${predictor}" \
    "${run_root}/runs/${mode}/${policy}/${predictor}/seed${seed}" >> "${manifest}"
}
case "${mode}" in
  smoke) add_row selective_b1_g3 0 neural_smoke ;;
  primary)
    for policy in selective_b1_g3 selective_b1_g5; do for seed in 0 1 2; do
      add_row "${policy}" "${seed}" neural_ensemble
    done; done ;;
  all_gates)
    for gate in 0 1 2 3 4 5 6 7 8 9; do for seed in 0 1 2; do
      add_row "selective_b1_g${gate}" "${seed}" neural_ensemble
    done; done ;;
  trees)
    for predictor in extra_trees hist_gradient_boosting; do for seed in 0 1 2; do
      add_row selective_b1_g3 "${seed}" "${predictor}"
    done; done ;;
  conformal) for seed in 0 1 2; do add_row selective_b1_g5 "${seed}" neural_ensemble; done ;;
  multi_horizon) for seed in 0 1 2; do add_row selective_b1_g8 "${seed}" neural_ensemble; done ;;
  trust_ablation) for seed in 0 1 2; do add_row selective_b1_g9 "${seed}" neural_ensemble; done ;;
  *) echo "Unknown mode ${mode}." >&2; exit 2 ;;
esac
count="$(( $(wc -l < "${manifest}") - 1 ))"
"${python_bin}" -c 'import csv,sys
p=sys.argv[1]; rows=list(csv.DictReader(open(p),delimiter="\t")); paths=[r["output_path"] for r in rows]
assert len(paths)==len(set(paths)), "duplicate selective output paths"
print("manifest={}".format(p)); print("rows={}".format(len(rows))); print("policies={}".format(sorted({r["policy"] for r in rows})))
print("seeds={}".format(sorted({r["seed"] for r in rows}))); print("predictors={}".format(sorted({r["predictor"] for r in rows})))
print("output_root={}".format(sys.argv[2]))' "${manifest}" "${run_root}"
mkdir -p slurmlogs/selective-wei
# No %N suffix: Otus schedules every eligible array element.
source_revision="$(uv run --frozen python -c 'from dacboenv.experiment.source_provenance import current_source_revision; print(current_source_revision())')"
sbatch --array="0-$((count - 1))" scripts/otus/opt_selective_wei.sh \
  "${repository}" "${manifest}" "${branch_root}" "expected_source_revision=$source_revision" "$@"
