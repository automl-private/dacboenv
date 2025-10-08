#!/bin/bash

BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK="+task/BBOB=cfg_2_1_0"
CLUSTER="+cluster=cpu_noctua"

# Models

# python -m $BASE +base=dacboenv_beta_model_ppo $TASK $CLUSTER $SEED +hydra.job.env_set.DACBOENV="" --multirun &
# python -m $BASE +base=dacboenv_beta_model_ppo_step +hydra.job.env_set.DACBOENV=STEP $TASK $CLUSTER $SEED --multirun &
# python -m $BASE +base=dacboenv_beta_model_ppo_bucket +hydra.job.env_set.DACBOENV=BUCKET $TASK $CLUSTER $SEED --multirun &
# python -m $BASE +base=dacboenv_beta_model_dqn_step +hydra.job.env_set.DACBOENV=STEP $TASK $CLUSTER $SEED --multirun &
# python -m $BASE +base=dacboenv_beta_model_dqn_bucket +hydra.job.env_set.DACBOENV=BUCKET $TASK $CLUSTER $SEED --multirun &
 

# Random

python -m $BASE +base=dacboenv_beta_random +hydra.job.env_set.DACBOENV="" optimizer_id=DACBOEnv-SMAC3-beta-random-default $TASK $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_random +hydra.job.env_set.DACBOENV=STEP optimizer_id=DACBOEnv-SMAC3-beta-random-step $TASK $CLUSTER $SEED --multirun &
python -m $BASE +base=dacboenv_beta_random +hydra.job.env_set.DACBOENV=BUCKET optimizer_id=DACBOEnv-SMAC3-beta-random-bucket $TASK $CLUSTER $SEED --multirun