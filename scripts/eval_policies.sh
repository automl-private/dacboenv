#!/usr/bin/env bash

set -f

export HYDRA_FULL_ERROR=1

TASKS_EVAL=(
    "+task/BBOB=glob(cfg_2_*_1)"
    "+task/BBOB=glob(cfg_8_*_1)"
    "+task/YAHPO/SO=glob(*)"
    # "+task/nasengb=glob(*) hydra.launcher.mem_per_cpu=16G"
    # "+task/OptBench=Ackley_2,Hartmann_3,Levy_2,Schwefel_2"
)

OUTER_SEEDS="seed1,seed2,seed3,seed4,seed5"

BASEENV="+env=base +env/opt=base +env/action=wei_alpha_continuous +env/obs=sawei +env/reward=ep_done_scaled +env/refperf=saweip dacboenv.evaluation_mode=true"

OPT_BASES=(
    # "$BASEENV +policy=defaultaction"
    # "$BASEENV +policy=random"
    # "$BASEENV +policy=sawei"
)

#PPO-MLP-norm--dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled-SAWEI-P_Ibbob2d_3seeds

POLICY_ROOT="+policy/optimized"
MODELS=(
    # "PPO-RNN"
    # "PPO-RNN-norm"
    "PPO-MLP"
    # "PPO-MLP-norm"
    # "PPO-MLP-Def"
)
ACTION_SPACES=(
    # "AWEI-cont"
    "AWEI-skip"
    # "AWEI-step"
)
REFPERFS=(
    "SAWEI-P"
    # "DefaultAction"
)

INSTANCESETS=(
    # "Iackley2d_3seeds"
    # "Ibbob2d_3seeds"
    # "Ibbob2d_fid8_3seeds"
    # "Ibbob2d_fid1_3seeds"
    # "Iselected-random"
    # "Isel2-random"
    # "Ibbob2d-8-random"
	"Iyahpo-glmnet375-random"
	"Iyahpo-glmnet375-3seeds"
)

INTERACTION_FREQS=(
    "i1"
    # "i5"
    # "i10"
)

OBS=(
    "sawei"
    # "all"
)

for model in "${MODELS[@]}"; do
    for actionspace in "${ACTION_SPACES[@]}"; do
        for obs in "${OBS[@]}"; do
            REWARDS=(
                # "dacbo_Cepisode_length_scaled_plus_logregret_${actionspace}_Ssawei_Repisode_finished_scaled"
                "dacbo_Csymlogregret_${actionspace}_S${obs}_Rsymlogregret"
                # "dacbo_Csymlogregret_${actionspace}_S${obs}_Rsymlogregret-op"
            )
            for reward in "${REWARDS[@]}"; do
                for refperf in "${REFPERFS[@]}"; do
                    for instanceset in "${INSTANCESETS[@]}"; do
                        for interactfreq in "${INTERACTION_FREQS[@]}"; do
                            # "+policy/optimized/PPO-AlphaNet/dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret-SMAC3-BlackBoxFacade_Ibbob2d_fid8_3seeds=$OUTER_SEEDS"
                            OPT_BASES+=(
                                "${POLICY_ROOT}/${model}/${reward}-${refperf}_${instanceset}_${interactfreq}=${OUTER_SEEDS}"
                            )
                        done
                    done
                done
            done
        done
    done
done


BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://optbench/configs]"
ARGS="+eval=base baserundir=runs_eval_nips +cluster=cpu_noctua"
run_eval() {
    python -m $BASE $ARGS "$@" --multirun &
}

SEEDS=(
    # "seed=1"
    # "seed=2"
    # "seed=3"
    # "seed=4"
    # "seed=5"
    # "seed=6"
    # "seed=7"
    # "seed=8"
    # "seed=9"
    # "seed=10"
    # "seed=range(1,5)"
    # "seed=range(5,8)"
    # "seed=range(8,11)"
    "seed=range(1,11)"
)

for optbase in "${OPT_BASES[@]}"; do   
    for task in "${TASKS_EVAL[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo $task $optbase $seed
            run_eval $task $optbase $seed
        done
    done
done

wait
