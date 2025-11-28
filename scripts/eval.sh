#!/bin/bash

BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK2="+task/BBOB=cfg_2_20_0"
TASK3="+task/BBOB=cfg_2_8_0"
CLUSTER="+cluster=cpu_noctua"

# Models

python -m $BASE +base=dacboenv_beta_model_dqn_step +hydra.job.env_set.REWARD="" +hydra.job.env_set.FID=0 +hydra.job.env_set.DACBOENV="" +hydra.job.env_set.OBS=SINGLE optimizer.action_mode="function" optimizer.policy_kwargs.model="/scratch/hpc-prf-intexml/tklenke/repos/dacboenv/training/dacbo_dqn_af_1_1_SINGLE" optimizer_id=DACBOEnv-SMAC3-beta-model-dqn-af_1_1_single $TASK1 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_model_dqn_step +hydra.job.env_set.REWARD="" +hydra.job.env_set.FID=0 +hydra.job.env_set.DACBOENV=BUCKET +hydra.job.env_set.OBS="" optimizer.policy_kwargs.model="/scratch/hpc-prf-intexml/tklenke/repos/dacboenv/training/dacbo_dqn_bucket_1_1_" optimizer_id=DACBOEnv-SMAC3-beta-model-dqn-bucket_1_1 $TASK1 $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_model_dqn_step +hydra.job.env_set.REWARD="" +hydra.job.env_set.FID=0 +hydra.job.env_set.DACBOENV=STEP +hydra.job.env_set.OBS=SMART optimizer.policy_kwargs.model="/scratch/hpc-prf-intexml/tklenke/repos/dacboenv/training/dacbo_dqn_step_1_1_SMART" optimizer_id=DACBOEnv-SMAC3-beta-model-dqn-step_1_1_smart $TASK1 $CLUSTER $SEED --multirun &

# Piecewise
python -m $BASE +optimizer/smac20=blackbox +base=dacbo_piecewise optimizer.smac_cfg.smac_kwargs.callbacks.0.splity=[0.0448121415486,0.284830500646,0.4613617941265,0.7448349801671,0.6110727317813] optimizer_id=DACBOEnv-SMAC3-piecewise-1-1 $TASK1 $CLUSTER $SEED --multirun & 

# Random
python -m $BASE +base=dacboenv_beta_random +hydra.job.env_set.REWARD="" +hydra.job.env_set.FID=0 +hydra.job.env_set.DACBOENV="" +hydra.job.env_set.OBS="" optimizer_id=DACBOEnv-SMAC3-beta-random $TASK1 $CLUSTER $SEED --multirun &


# SMAC
python -m $BASE +optimizer/smac20=blackbox $TASK1 $CLUSTER $SEED --multirun