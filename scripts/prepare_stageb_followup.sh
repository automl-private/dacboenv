#!/usr/bin/env bash

set -euo pipefail

if (( $# != 3 )); then
  echo "Usage: bash scripts/prepare_stageb_followup.sh /ABS/STAGEB_RUN_ROOT /ABS/EVAL_ROOT /ABS/FOLLOWUP_ROOT" >&2
  exit 2
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"
repository_root="$(cd -- "${script_directory}/.." && pwd -P)"
python_bin="${repository_root}/.env/bin/python"

exec "${python_bin}" -m dacboenv.experiment.stageb_followup prepare "$1" "$2" "$3" --repository "${repository_root}"
