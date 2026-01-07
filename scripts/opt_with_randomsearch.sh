#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "rs4dacbo"
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-3
#SBATCH --output=slurmlog/randomsearch/slurm-%A_%a.out
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
fi
export HYDRA_FULL_ERROR=1

TASK_OVERRIDE=$1
INSTANCE_SET_OVERRIDE=$2

python -m dacboenv.experiment.optimize_via_randomsearch +opt=randomsearch $TASK_OVERRIDE $INSTANCE_SET_OVERRIDE seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random task.optimization_resources.n_trials=100000
