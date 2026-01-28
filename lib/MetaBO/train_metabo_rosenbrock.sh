#!/bin/bash
#SBATCH --job-name=mbros
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00            
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurmlogs/trainmetabo/slurm-%j.out     # stdout log
#SBATCH --error=slurmlogs/trainmetabo/slurm-%j.err      # stderr log

# conda init
# conda activate metabo
conda run -n metabo python train_metabo_rosenbrock.py