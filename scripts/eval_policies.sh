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

BASEENV="+env=base +env/opt=base +env/action=wei_alpha_continuous +env/obs=sawei +env/reward=ep_done_scaled +env/refperf=saweip dacboenv.evaluation_mode=true"

OPT_BASES=(
    "$BASEENV +policy=defaultaction"
    "$BASEENV +policy=random"
    "$BASEENV +policy=sawei"
)

#PPO-MLP-norm--dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SAWEI-P_Ibbob2d_3seeds

POLICY_ROOT="+policy/optimized"
MODELS=(
    "PPO-RNN"
    "PPO-RNN-norm"
    "PPO-MLP"
    "PPO-MLP-norm"
)
ACTION_SPACES=(
    "AWEI-cont"
    "AWEI-skip"
)
REWARDS=(
    "dacbo_Cepisode_length_scaled_plus_logregret_${actionspace}_Ssawei_Repisode_finished_scaled"
    "dacbo_Csymlogregret_${actionspace}_Ssawei_Rsymlogregret"
)
REFPERFS=(
    "SAWEI-P"
    "DefaultAction"
)

INSTANCESETS=(
    "Iackley2d_3seeds"
    "Ibbob2d_3seeds"
    "Ibbob2d_fid8_3seeds"
)

for model in "${MODELS[@]}"; do
    for actionspace in "${ACTION_SPACES[@]}"; do
        REWARDS=(
            "dacbo_Cepisode_length_scaled_plus_logregret_${actionspace}_Ssawei_Repisode_finished_scaled"
            "dacbo_Csymlogregret_${actionspace}_Ssawei_Rsymlogregret"
        )
        for reward in "${REWARDS[@]}"; do
            for refperf in "${REFPERFS[@]}"; do
                for instanceset in "${INSTANCESETS[@]}"; do
                    # "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SMAC3-BlackBoxFacade_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
                    OPT_BASES+=(
                        "${POLICY_ROOT}/${model}/${reward}-${refperf}_${instanceset}=${OUTER_SEEDS}"
                    )
                done
            done
        done
    done
done


BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs,pkg://optbench/configs]"
ARGS="+eval=base baserundir=runs_eval_icml +cluster=cpu_noctua seed=range(1,11)"
run_eval() {
    python -m $BASE $ARGS "$@" --multirun &
}

for optbase in "${OPT_BASES[@]}"; do   
    for task in "${TASKS_EVAL[@]}"; do
        echo $task $optbase
        run_eval $task $optbase
    done
done

wait
