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

cd /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/
source /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/.venv/bin/activate

python -m dacboenv.experiment.ppo +opt=ppo experiment.n_workers=16 experiment.n_episodes=50 +dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.logging_level=9999 $@ seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random