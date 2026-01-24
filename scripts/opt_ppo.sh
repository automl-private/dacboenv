#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "ppo4dacbo"
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH -p normal
#SBATCH --array=1-5
#SBATCH --output=slurmlogs/ppo/slurm-%j.out     # stdout log
#SBATCH --error=slurmlogs/ppo/slurm-%j.err      # stderr log

N_WORKERS=16
BASERUNDIR="runsicml"
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=1
    # N_WORKERS=1
    BASERUNDIR="tmpruns"
    # rm -rf tmpruns
fi

echo $BASERUNDIR

export HYDRA_FULL_ERROR=1

python -m dacboenv.experiment.ppo \
    $@ \
    experiment.n_workers=$N_WORKERS \
    experiment.n_episodes=1000 \
    seed=$SLURM_ARRAY_TASK_ID \
    +env/instance_selector=roundrobin \
    baserundir=$BASERUNDIR