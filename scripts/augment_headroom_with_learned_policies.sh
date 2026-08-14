#!/usr/bin/env bash

set -euo pipefail

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=scripts/evaluation_determinism_env.sh
source "${script_directory}/evaluation_determinism_env.sh"
repository_root="$(cd -- "${script_directory}/.." && pwd -P)"
python_bin="${DACBO_PYTHON:-${repository_root}/.env/bin/python}"

if (( $# == 0 )); then
    cat >&2 <<'EOF'
Usage: scripts/augment_headroom_with_learned_policies.sh \
  --run-root RUN [--run-root RUN ...] --action-family {wei|af_selection} \
  --domain {yahpo|mixed} --checkpoint {best|final} --output-root OUTPUT

The wrapper first performs strict checkpoint/config/validation preflight. It
never discovers or selects outer seeds from held-out performance.
EOF
    exit 2
fi

exec "${python_bin}" -m dacboenv.experiment.augment_headroom_with_learned_policies "$@"
