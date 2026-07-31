#!/usr/bin/env bash

set -euo pipefail

export HYDRA_FULL_ERROR=1

evaluation_set="${DACBO_EVALUATION_SET:-bbob_validation_holdout}"

launcher_args=()
if [[ "${DACBO_BASELINE_JOBS:-1}" -gt 1 ]]; then
    launcher_args+=(
        "hydra/launcher=joblib"
        "hydra.launcher.n_jobs=${DACBO_BASELINE_JOBS}"
    )
fi

# 3 fixed frequencies x 10 policy seeds.
python -m dacboenv.experiment.baseline \
    +baseline=structured_random \
    "+env/interaction_freq=f1,f5,f10" \
    "instance_sets@evaluation_instances=${evaluation_set}" \
    "seed=range(0,10)" \
    "+cluster=cpu_noctua" \
    "${launcher_args[@]}" \
    --multirun &

# 3 fixed frequencies x 5 exact static alpha levels.
python -m dacboenv.experiment.baseline \
    +baseline=structured_static \
    "+env/interaction_freq=f1,f5,f10" \
    "instance_sets@evaluation_instances=${evaluation_set}" \
    "policy/static/wei_discrete=level_000,level_025,level_050,level_075,level_100" \
    "seed=0" \
    "+cluster=cpu_noctua" \
    "${launcher_args[@]}" \
    --multirun
