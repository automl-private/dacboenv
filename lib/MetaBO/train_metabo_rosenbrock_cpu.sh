#!/bin/bash
#SBATCH --job-name=mbros
#SBATCH --partition=normal
#SBATCH --time=48:00:00            
#SBATCH --cpus-per-task=17
#SBATCH --output=slurmlogs/trainmetabo/slurm-%j.out     # stdout log
#SBATCH --error=slurmlogs/trainmetabo/slurm-%j.err      # stderr log

# conda init
# conda activate metabo
conda run -n metabo python train_metabo_rosenbrock.py