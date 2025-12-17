#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "cma4dacbo"
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-5
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi
export HYDRA_FULL_ERROR=1

TASK_OVERRIDE=$1
INSTANCE_SET_OVERRIDE=$2

python -m dacboenv.experiment.optimize_via_cma +opt=cma $TASK_OVERRIDE $INSTANCE_SET_OVERRIDE seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
