#!/usr/bin/env bash
# Evaluate static discrete actions (0-4) on all BBOB and YAHPO tasks.

set -euo pipefail

export DACBO_YAHPO_REFERENCE_TABLE=/scratch/hpc-prf-intexml/tklenke/repos/dacboenv_new/dacboenv/experiment/analysis/yahpo_best_known_references.json
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

output_root="$(pwd)/static_results"
mkdir -p "${output_root}"

# 1. Generate the string for all BBOB tasks (dims 2,4,8,16 and funcs 1-24)
bbob_dims="2,4,8,16"
bbob_funcs=$(seq -s, 1 24)
bbob_tasks=""
for dim in ${bbob_dims//,/ }; do
    for func in ${bbob_funcs//,/ }; do
        bbob_tasks+="${bbob_tasks:+,}cfg_${dim}_${func}_0"
    done
done

# Define our actions to loop over
actions=("action_0" "action_1" "action_2" "action_3" "action_4")
pids=()

echo "Submitting BBOB and YAHPO static evaluations..."

for action in "${actions[@]}"; do
    # BBOB Submission
    uv run --frozen python -m carps.run -m \
        "hydra.searchpath=[pkg://dacboenv/configs]" \
        "+task/BBOB=${bbob_tasks}" \
        "+eval=base" \
        "+env=base" \
        "+env/opt=base" \
        "+env/reward=reference_regret_improvement" \
        "+env/obs=structured" \
        "+env/reference_provider=composite" \
        "+env/action=wei_alpha_discrete" \
        "+policy/static/discrete_action=${action}" \
        "+cluster=cpu_noctua" \
        "optimizer_id=static_${action}_\${seed}" \
        "seed=range(0,10)" \
        "baserundir=${output_root}" \
        "hydra.launcher.array_parallelism=5000" \
        "dacboenv.evaluation_mode=false" \
        "dacboenv.terminate_after_reference_performance_reached=false" \
        "dacboenv.context_split=test" &
    pids+=($!)

    # YAHPO Submission
    uv run --frozen python -m carps.run -m \
        "hydra.searchpath=[pkg://dacboenv/configs]" \
        "+task/YAHPO/SO=glob(*)" \
        "+eval=base" \
        "+env=base" \
        "+env/opt=base" \
        "+env/reward=reference_regret_improvement" \
        "+env/obs=structured" \
        "+env/reference_provider=composite" \
        "+env/action=wei_alpha_discrete" \
        "+policy/static/discrete_action=${action}" \
        "+cluster=cpu_noctua" \
        "optimizer_id=static_${action}_\${seed}" \
        "seed=range(0,10)" \
        "baserundir=${output_root}" \
        "hydra.launcher.array_parallelism=5000" \
        "dacboenv.evaluation_mode=false" \
        "dacboenv.terminate_after_reference_performance_reached=false" \
        "dacboenv.context_split=test" &
    pids+=($!)
done

# Wait for all background Hydra submissions to finish
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo "All static evaluations submitted to Slurm!"
echo "Results will be saved in: ${output_root}"