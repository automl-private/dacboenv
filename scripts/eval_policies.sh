#!/bin/bash

set -f

export HYDRA_FULL_ERROR=1

OBS="sawei"
BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
ARGS="+eval=base +env=base +env/obs=$OBS +env/reward=ep_done_scaled +env/opt=base +cluster=cpu_noctua seed=range(1,11)"

run_eval() {
    python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
}

TASKS_GENERAL=(
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*) hydra.launcher.mem_per_cpu=16G"
)

UCB_SUFFIX="_AUCB-cont_S${OBS}_Repisode_finished_scaled_Ibbob2d"
WEI_SUFFIX="_AWEI-cont_S${OBS}_Repisode_finished_scaled_Ibbob2d"

OPT_BASES=(
    # "+policy/optimized/PPO-Perceptron"
    # "+policy/optimized/SMAC-AC"
    "+policy/optimized/PPO-AlphaNet"
    # "+policy/optimized/SMAC-AC-WS"
    # "+policy/optimized/CMA-1.3"
)

OUTER_SEEDS="seed1,seed2,seed3,seed4,seed5"
OUTER_SEEDS="seed1,seed2,seed3"

for base in "${OPT_BASES[@]}"; do
    UCB_P2=(
        "+env/action=ucb_beta_continuous"
        "${base}/dacbo_Cepisode_length_scaled_plus_logregret${UCB_SUFFIX}_3seeds=${OUTER_SEEDS}"
    )
    WEI_P2=(
        "+env/action=wei_alpha_continuous"
        "${base}/dacbo_Cepisode_length_scaled_plus_logregret${WEI_SUFFIX}_3seeds=${OUTER_SEEDS}"
    )

    # Eval P1 on 2D and 8D training tasks
    for fid in 1 8; do
        for d in 2 8; do
            # run_eval "+task/BBOB=cfg_${d}_${fid}_0" \
            #         "${UCB_P2[0]}" \
            #         "${base}/dacbo_Cepisode_length_scaled_plus_logregret${UCB_SUFFIX}_fid${fid}_3seeds=${OUTER_SEEDS}"
            
            run_eval "+task/BBOB=cfg_${d}_${fid}_0" \
                    "${WEI_P2[0]}" \
                    "${base}/dacbo_Cepisode_length_scaled_plus_logregret${WEI_SUFFIX}_fid${fid}_3seeds=${OUTER_SEEDS}"
        done
    done

    # Eval P2 on training set
    # run_eval "+task/BBOB=glob(cfg_2_*_0)" "${UCB_P2[@]}"
    run_eval "+task/BBOB=glob(cfg_2_*_0)" "${WEI_P2[@]}"

    # Eval P2 for generalization
    for task in "${TASKS_GENERAL[@]}"; do
        # run_eval $task "${UCB_P2[@]}"
        run_eval $task "${WEI_P2[@]}"
    done

done

wait