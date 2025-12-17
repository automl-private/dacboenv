#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "ac4dacbo"
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-5
#SBATCH --output=slurmlog/smac/slurm-%A_%a.out

if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi
BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"

TASK_OVERRIDE=$1
INSTANCE_SET_OVERRIDE=$2

python -m $BASE seed=$SLURM_ARRAY_TASK_ID +opt=smac $TASK_OVERRIDE $INSTANCE_SET_OVERRIDE optimizer.smac_cfg.scenario.n_workers=10
