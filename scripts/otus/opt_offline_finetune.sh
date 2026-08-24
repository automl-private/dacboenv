#!/usr/bin/env bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --job-name=dacbo-offline-ft
#SBATCH --output=slurmlogs/offline-finetune/%A_%a.out
#SBATCH --error=slurmlogs/offline-finetune/%A_%a.err
set -euo pipefail
if (( $# != 5 )); then
  echo "Usage: $0 REPOSITORY MANIFEST FINAL_ROOT OUTPUT_ROOT DOMAIN" >&2
  exit 2
fi
repository="$1" manifest="$2" final_root="$3" output_root="$4" domain="$5"
cd "${repository}"
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
IFS=$'\t' read -r run_id source_policy_id checkpoint checkpoint_hash normalizer normalizer_hash seed < <(
  sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "${manifest}"
)
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
training_config="offline_ddqn_f5_finetune"
[[ "${domain}" == "mixed" ]] && training_config="offline_ddqn_f5_finetune_mixed"
exec uv run --frozen python -m dacboenv.experiment.rl \
  "hydra.searchpath=[file://${final_root}/hydra_configs]" \
  "+training=${training_config}" \
  "seed=${seed}" "baserundir=${output_root}/${run_id}" "optimizer_id=${run_id}" \
  "offline_initialization.checkpoint=${checkpoint}" \
  "offline_initialization.normalizer=${normalizer}" \
  "offline_initialization.checkpoint_sha256=${checkpoint_hash}" \
  "offline_initialization.normalizer_sha256=${normalizer_hash}" \
  "+offline_initialization.source_policy_id=${source_policy_id}" \
  "offline_replay_prefill.dataset=${final_root}/behavior_train.npz"
