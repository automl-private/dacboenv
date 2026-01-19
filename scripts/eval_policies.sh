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
BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs]"


TASKS_GENERAL=(
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*) hydra.launcher.mem_per_cpu=16G"
)

OPT_BASES=(
    # "+policy/optimized/PPO-Perceptron"
    # "+policy/optimized/SMAC-AC"
    # "+policy/optimized/PPO-AlphaNet"
    # "+policy/optimized/PPO-AlphaNet2"
    # "+policy/optimized/PPO-AlphaNet3"
    # "+policy/optimized/SMAC-AC-WS"
    # "+policy/optimized/CMA-1.3"
)

REWARDS=(
    "episode_finished_scaled"
    # "symlogregret"
)

INSTANCE_SET="bbob2d_3seeds"

OUTER_SEEDS="seed1,seed2,seed3,seed4,seed5"
OUTER_SEEDS="seed1,seed2,seed3"

action_space=$ACTIONSPACE
fid=8

for base in "${OPT_BASES[@]}"; do
    for reward in "${REWARDS[@]}"; do
        if [[ $reward = "episode_finished_scaled" ]]
        then
            REWARDOVERRIDE="+env/reward=ep_done_scaled"
            cost="episode_length_scaled_plus_logregret"
        elif [[ $reward = "symlogregret" ]]
        then
            REWARDOVERRIDE="+env/reward=symlogregret"
            cost="symlogregret"
        fi
        ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} +cluster=cpu_noctua seed=range(1,11)"

        # run_eval() {
        #     python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
        # }
        # Eval FID on 2D and 8D training tasks
        # instance_set="bbob2d_fid${fid}_3seeds"
        # TRAINTASK="dacbo_C${cost}_A${action_space}_S${OBS}_R${reward}_I${instance_set}"
        # for d in 2 8; do            
        #     run_eval "+task/BBOB=cfg_${d}_${fid}_0" \
        #             "${base}/${TRAINTASK}=${OUTER_SEEDS}"
        # done
        # d=8
        # run_eval "+task/BBOB=cfg_${d}_${fid}_0" \
        #             "${base}/${TRAINTASK}=${OUTER_SEEDS}"
        # run_eval "+task/BBOB=glob(cfg_2_*_0)" \
        #             "${base}/${TRAINTASK}=${OUTER_SEEDS}"

        # instance_set=$INSTANCE_SET
        # TRAINTASK="dacbo_C${cost}_A${action_space}_S${OBS}_R${reward}_I${instance_set}"
        # # Eval P2 on training set
        # # run_eval "+task/BBOB=glob(cfg_2_*_0)" "${base}/${TRAINTASK}=${OUTER_SEEDS}"

        # # Eval P2 for generalization
        # for task in "${TASKS_GENERAL[@]}"; do
        #     run_eval $task "${base}/${TRAINTASK}=${OUTER_SEEDS}"
        # done
    done
done

# # Eval Default Policy
# REWARDOVERRIDE="+env/reward=symlogregret"
# ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} +cluster=cpu_noctua seed=range(1,11)"
# run_eval() {
#     python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
# }
# run_eval "+task/BBOB=glob(cfg_2_*_0)" "+policy=default"
# for task in "${TASKS_GENERAL[@]}"; do
#     run_eval $task "+policy=default"
# done

# Eval Random Policy
REWARDOVERRIDE="+env/reward=symlogregret"
ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} +env/refperf=smacbb +cluster=cpu_noctua seed=range(1,11)"
run_eval() {
    python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
}
run_eval "+task/BBOB=glob(cfg_2_*_0)" "+policy=random"
for task in "${TASKS_GENERAL[@]}"; do
    run_eval $task "+policy=random"
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

# # Eval SAWEI-P
# REWARDOVERRIDE="+env/reward=symlogregret"
# ARGS="+eval=base +env=base +env/obs=$OBS $REWARDOVERRIDE +env/opt=base ${ACTIONSPACE_OVERRIDE} +cluster=cpu_noctua seed=range(1,11)"
# run_eval() {
#     python -m $BASE $ARGS "$@" "dacboenv.terminate_after_reference_performance_reached=false" --multirun &
# }
# # run_eval "+task/BBOB=glob(cfg_2_*_0)" "+policy=sawei" "baserundir=runs_eval_SAWEI"
# for task in "${TASKS_GENERAL[@]}"; do
#     run_eval $task "+policy=sawei"
# done

wait


# python -m carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs] +eval=base +env=base +env/obs=sawei +env/opt=base +env/action=wei_alpha_continuous seed=1 +task/BBOB=cfg_8_1_0 +policy/optimized/SMAC-AC/dacbo_C_AWEI-cont_Ssawei_Repisode_finished_scaled_Ibbob2d_3seeds=seed1 dacboenv.terminate_after_reference_performance_reached=false