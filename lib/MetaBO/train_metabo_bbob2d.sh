#!/bin/bash
#SBATCH --job-name=mbbbob2d
#SBATCH --partition=normal
#SBATCH --time=48:00:00            
#SBATCH --cpus-per-task=17
#SBATCH --mem-per-cpu=1G
#SBATCH --output=slurmlogs/trainmetabo/slurm-%j.out     # stdout log
#SBATCH --error=slurmlogs/trainmetabo/slurm-%j.err      # stderr log

# conda init
# conda activate metabo
source ~/.bashrc
conda activate ./env 
python train_metabo_bbob2d.py