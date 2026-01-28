#!/bin/bash
#SBATCH --job-name=eval_bbob
#SBATCH --partition=normal
#SBATCH --time=48:00:00            
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --output=slurmlogs/trainmetabo/slurm-%j.out     # stdout log
#SBATCH --error=slurmlogs/trainmetabo/slurm-%j.err      # stderr log

# conda init
# conda activate metabo
source ~/.bashrc
conda activate ./env 
python evaluate_metabo_bbob.py "$@"