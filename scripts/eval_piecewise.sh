#!/bin/bash

BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK20="+task/BBOB=cfg_2_20_0"
TASK8="+task/BBOB=cfg_2_8_0"
TASKALL="+task/BBOB=glob(cfg_2_*_0)"
CLUSTER="+cluster=cpu_noctua"

JUNK='+hydra.job.env_set.REWARD="" +hydra.job.env_set.FID=0 +hydra.job.env_set.DACBOENV="" +hydra.job.env_set.OBS=""'

# 1 8 20
schedule_all="[0.0409422418607,0.616540534334,0.3248869871912,0.0087254513079,0.0734833146024]"

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_all" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-all $TASKALL $CLUSTER $SEED --multirun &

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_all" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-all +task/BBOB=cfg_8_1_0 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_all" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-all +task/BBOB=cfg_8_8_0 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_all" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-all +task/BBOB=cfg_8_20_0 $CLUSTER $SEED --multirun &

# 1
schedule_1="[0.6366544173945,0.8552533543557,0.3720589811136,0.8666781584554,0.4694579767571]"

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_1" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-1 $TASKALL $CLUSTER $SEED --multirun &

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_1" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-1 +task/BBOB=cfg_8_1_0 $CLUSTER $SEED --multirun &

# 8
schedule_8="[0.6317839105595,0.0009985648028,0.0031076662264,0.0092381601143,0.4425322655247]"

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_8" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-8 $TASKALL $CLUSTER $SEED --multirun &

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_8" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-8 +task/BBOB=cfg_8_8_0 $CLUSTER $SEED --multirun &


# 20
schedule_20="[0.4118671508415,0.3249016104744,0.0381227803032,0.1338747159979,0.4269643590197]"

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_20" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-20 $TASKALL $CLUSTER $SEED --multirun &

python -m $BASE +base=dacboenv_beta_piecewise optimizer.policy_kwargs.splits="$schedule_20" $JUNK optimizer_id=DACBOEnv-SMAC3-beta-piecewise-20 +task/BBOB=cfg_8_20_0 $CLUSTER $SEED --multirun &

# Random

python -m $BASE +base=dacboenv_beta_random $JUNK optimizer_id=DACBOEnv-SMAC3-beta-random $TASKALL $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_random $JUNK optimizer_id=DACBOEnv-SMAC3-beta-random +task/BBOB=cfg_8_1_0 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_random $JUNK optimizer_id=DACBOEnv-SMAC3-beta-random +task/BBOB=cfg_8_8_0 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_random $JUNK optimizer_id=DACBOEnv-SMAC3-beta-random +task/BBOB=cfg_8_20_0 $CLUSTER $SEED --multirun &

# SMAC
python -m $BASE +optimizer/smac20=blackbox $TASKALL $CLUSTER $SEED --multirun &
python -m $BASE +optimizer/smac20=blackbox +task/BBOB=cfg_8_1_0 $CLUSTER $SEED --multirun &
python -m $BASE +optimizer/smac20=blackbox +task/BBOB=cfg_8_8_0 $CLUSTER $SEED --multirun &
python -m $BASE +optimizer/smac20=blackbox +task/BBOB=cfg_8_20_0 $CLUSTER $SEED --multirun