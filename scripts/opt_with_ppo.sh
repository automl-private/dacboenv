#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "ppo4dacbo"
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-3


if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi

export HYDRA_FULL_ERROR=1

python -m dacboenv.experiment.ppo +opt=ppo  +task=dacboenv_epdonescaledpluslogregret_trialsleft +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
# python -m dacboenv.experiment.ppo +opt=ppo  +task=dacboenv_epdonescaledpluslogregret +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
# python -m dacboenv.experiment.ppo +opt=ppo  +task=dacboenv_epdonescaledpluslogregret_wei +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random