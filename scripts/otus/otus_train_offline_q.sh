#!/usr/bin/env bash
# Freeze and submit one offline-Q array without an artificial concurrency cap.
set -euo pipefail
if (( $# < 4 )); then
  echo "Usage: $0 FINAL_DATASET_ROOT BRANCH_ROOT RUN_ROOT MODE [--cql-coefficient VALUE] [--updates N]" >&2
  exit 2
fi
final_root="$(realpath "$1")" branch_root="$(realpath "$2")"
mkdir -p "$3"; run_root="$(realpath "$3")"; mode="$4"
shift 4
o4_coefficient=""; o4_updates=15000
while (( $# )); do
  case "$1" in
    --cql-coefficient) o4_coefficient="${2:?missing coefficient}"; shift 2 ;;
    --updates) o4_updates="${2:?missing updates}"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
repository="$(git rev-parse --show-toplevel)"; cd "${repository}"
python_bin="${OTUS_PYTHON:-${repository}/.venv/bin/python}"
[[ -x "${python_bin}" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
manifest="${run_root}/offline_training_${mode}.tsv"
[[ ! -e "${manifest}" ]] || { echo "Manifest exists; refusing duplicate submission: ${manifest}" >&2; exit 3; }
printf 'cell\tseed\tcql_coefficient\tupdates\toutput_path\n' > "${manifest}"
add_row() {
  local cell="$1" seed="$2" coefficient="$3" updates="$4"
  printf '%s\t%s\t%s\t%s\t%s\n' "${cell}" "${seed}" "${coefficient}" "${updates}" \
    "${run_root}/runs/${mode}/${cell}/seed${seed}/cql${coefficient}" >> "${manifest}"
}
case "${mode}" in
  branch_smoke) add_row branch_q5_yahpo 0 0.0 200 ;;
  cql_smoke) add_row offline_cql_yahpo 0 0.5 200 ;;
  yahpo_stage1)
    for seed in 0 1 2; do add_row branch_q5_yahpo "${seed}" 0.0 5000; done
    for seed in 0 1 2; do add_row offline_fqi_yahpo "${seed}" 0.0 15000; done
    for seed in 0 1 2; do for coefficient in 0.1 0.5; do
      add_row offline_cql_yahpo "${seed}" "${coefficient}" 15000
    done; done
    ;;
  yahpo_o4)
    [[ "${o4_coefficient}" =~ ^(0\.1|0\.5|1\.0)$ ]] || {
      echo "yahpo_o4 requires --cql-coefficient 0.1, 0.5, or 1.0." >&2; exit 2;
    }
    [[ "${o4_updates}" =~ ^[1-9][0-9]*$ ]] || { echo "--updates must be positive." >&2; exit 2; }
    for seed in 0 1 2; do add_row branch_q5_cql_yahpo "${seed}" "${o4_coefficient}" "${o4_updates}"; done
    ;;
  yahpo_core)
    for cell in offline_fqi_yahpo branch_q5_yahpo; do
      for seed in 0 1 2; do add_row "${cell}" "${seed}" 0.0 50000; done
    done
    for cell in offline_cql_yahpo branch_q5_cql_yahpo; do
      for seed in 0 1 2; do for coefficient in 0.1 0.5 1.0; do
        add_row "${cell}" "${seed}" "${coefficient}" 50000
      done; done
    done
    ;;
  mixed_secondary)
    for cell in mixed_offline_cql mixed_branch_q5_cql; do
      for seed in 0 1 2; do for coefficient in 0.1 0.5 1.0; do
        add_row "${cell}" "${seed}" "${coefficient}" 50000
      done; done
    done
    ;;
  *) echo "Unknown mode ${mode}." >&2; exit 2 ;;
esac
count="$(( $(wc -l < "${manifest}") - 1 ))"
"${python_bin}" -c 'import csv,sys; p=sys.argv[1]; rows=list(csv.DictReader(open(p),delimiter="\t")); paths=[r["output_path"] for r in rows]; assert len(paths)==len(set(paths)),"duplicate output paths"; print("manifest={}".format(p)); print("rows={}".format(len(rows))); print("cells={}".format(sorted({r["cell"] for r in rows}))); print("seeds={}".format(sorted({r["seed"] for r in rows}))); print("coefficients={}".format(sorted({r["cql_coefficient"] for r in rows}))); print("updates={}".format(sorted({r["updates"] for r in rows}))); print("output_root={}".format(sys.argv[2]))' "${manifest}" "${run_root}"
mkdir -p slurmlogs/offline-q
# No %N suffix: Otus schedules every eligible array element.
sbatch --array="0-$((count - 1))" scripts/otus/opt_offline_q.sh \
  "${repository}" "${manifest}" "${final_root}" "${branch_root}"
