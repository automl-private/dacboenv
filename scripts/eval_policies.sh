#!/bin/bash

set -f

export HYDRA_FULL_ERROR=1



ACTION_SPACES=(
    # "UCB-cont"
    "WEI-cont"
)
ACTIONSPACE="WEI-cont"
ACTIONSPACE_OVERRIDE="+env/action=wei_alpha_continuous"


OBS="sawei"
BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs,pkg://optbench/configs]"


TASKS_GENERAL=(
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*) hydra.launcher.mem_per_cpu=16G"
)
TASKS_OPTBENCH="+task/OptBench=Ackley_2,Hartmann_3,Levy_2,Schwefel_2"
TASKS_EVAL=(
    "+task/BBOB=glob(cfg_2_*_0)"
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*) hydra.launcher.mem_per_cpu=16G"
    "+task/OptBench=Ackley_2,Hartmann_3,Levy_2,Schwefel_2"
)

OUTER_SEEDS="seed1,seed2,seed3,seed4,seed5"
# OUTER_SEEDS="seed1,seed2,seed3"

OPT_BASES=(
    "+policy=noop"
    "+policy=random"
    "+policy=sawei"
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

REWARDS=(
    "episode_finished_scaled"
    "symlogregret"
)

INSTANCE_SETS=(
    "bbob2d_3seeds"
    "bbob2d_fid8_3seeds"
)



action_space=$ACTIONSPACE
# fid=8

for optbase in "${OPT_BASES[@]}"; do
    for reward in "${REWARDS[@]}"; do
        for refperf in "${REFPERFS[@]}"; do
            if [[ $reward = "episode_finished_scaled" ]]
            then
                REWARDOVERRIDE="+env/reward=ep_done_scaled"
                cost="episode_length_scaled_plus_logregret"
            elif [[ $reward = "symlogregret" ]]
            then
                REWARDOVERRIDE="+env/reward=symlogregret"
                cost="symlogregret"
            fi
            ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} $refperf +cluster=cpu_noctua seed=range(1,11)"

            run_eval() {
                python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
            }
            # TRAINTASK="dacbo_C${cost}_A${action_space}_S${OBS}_R${reward}_I${instance_set}"

            # instance_set=$INSTANCE_SET
            # TRAINTASK="dacbo_C${cost}_A${action_space}_S${OBS}_R${reward}_I${instance_set}"

            for task in "${TASKS_EVAL[@]}"; do
                run_eval $task $optbase
            done
        done
    done
done


REWARDOVERRIDE="+env/reward=symlogregret"
ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} +env/refperf=smacbb +cluster=cpu_noctua seed=range(1,11)"
run_eval() {
    python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
}

BASELINES=(
    "+policy=noop"
    "+policy=random"
    "+policy=sawei"
)

run_eval "+task/BBOB=glob(cfg_2_*_0)" $baseline
run_eval $TASKS_OPTBENCH "+policy=noop"
for task in "${TASKS_GENERAL[@]}"; do
    run_eval $task "+policy=noop"
done

# Eval Random Policy
run_eval "+task/BBOB=glob(cfg_2_*_0)" "+policy=random"
run_eval $TASKS_OPTBENCH "+policy=random"
for task in "${TASKS_GENERAL[@]}"; do
    run_eval $task "+policy=random"
done


# Eval SAWEI-P
run_eval "+task/BBOB=glob(cfg_2_*_0)" "+policy=sawei"
run_eval $TASKS_OPTBENCH "+policy=sawei"
for task in "${TASKS_GENERAL[@]}"; do
    run_eval $task "+policy=sawei"
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