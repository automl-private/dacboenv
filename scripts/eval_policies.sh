#!/bin/bash

set -f

export HYDRA_FULL_ERROR=1

TASKS_EVAL=(
    "+task/BBOB=glob(cfg_2_*_0)"
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*) hydra.launcher.mem_per_cpu=16G"
    "+task/OptBench=Ackley_2,Hartmann_3,Levy_2,Schwefel_2"
)

OUTER_SEEDS="seed1,seed2,seed3,seed4,seed5"
# OUTER_SEEDS="seed1,seed2,seed3"

BASEENV="+env=base +env/opt=base +env/action=wei_alpha_continuous +env/obs=sawei +env/reward=ep_done_scaled dacboenv.evaluation_mode=true"

OPT_BASES=(
    "$BASEENV +policy=noop"
    "$BASEENV +policy=random"
    "$BASEENV +policy=sawei"
    "+policy/optimized/PPO-AlphaNet/dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SAWEI-P_Ibbob2d_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SAWEI-P_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SMAC3-BlackBoxFacade_Ibbob2d_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SMAC3-BlackBoxFacade_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SAWEI-P_Ibbob2d_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SAWEI-P_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SMAC3-BlackBoxFacade_Ibbob2d_3seeds=$OUTER_SEEDS"
    "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SMAC3-BlackBoxFacade_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
)
# PPO-AlphaNet--{task_id}--seed{seed}

BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs,pkg://optbench/configs]"
ARGS="+eval=base baserundir=runs_eval_icml +cluster=cpu_noctua seed=range(1,11)"
run_eval() {
    python -m $BASE $ARGS "$@" --multirun &
}

for optbase in "${OPT_BASES[@]}"; do    
    for task in "${TASKS_EVAL[@]}"; do
        run_eval $task $optbase
    done
done

# # Eval SAWEI original adaptaf
# REWARDOVERRIDE="+env/reward=symlogregret"
# ARGS="baserundir=runs_eval +cluster=cpu_noctua seed=range(1,11)"
# run_eval() {
#     python -m $BASE $ARGS "$@" --multirun &
# }
# run_eval "+task/BBOB=glob(cfg_2_*_0)" "+method=sawei_20p"
# for task in "${TASKS_GENERAL[@]}"; do
#     run_eval $task "+method=sawei_20p"
# done

wait


# python -m carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs] +eval=base +env=base +env/obs=sawei +env/opt=base +env/action=wei_alpha_continuous seed=1 +task/BBOB=cfg_8_1_0 +policy/optimized/SMAC-AC/dacbo_C_AWEI-cont_Ssawei_Repisode_finished_scaled_Ibbob2d_3seeds=seed1 dacboenv.terminate_after_reference_performance_reached=false