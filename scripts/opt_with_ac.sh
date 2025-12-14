#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "ac4dacbo"
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-10


BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
CLUSTER="+cluster=cpu_noctua"

# python -m $BASE seed=$SLURM_ARRAY_TASK_ID +opt=smac +env=default +instances=bbob2d_5seeds optimizer.smac_cfg.scenario.n_workers=1
python -m $BASE seed=$SLURM_ARRAY_TASK_ID +opt=smac_inccost +env=default +instances=bbob2d_3seeds optimizer.smac_cfg.scenario.n_workers=1