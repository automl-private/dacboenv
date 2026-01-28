#!/bin/bash
#SBATCH --job-name=gather
#SBATCH --partition=normal
#SBATCH --time=5:00:00            
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB    
#SBATCH --output=gather_data/slurm-%j.out     # stdout log
#SBATCH --error=gather_data/slurm-%j.err      # stderr log

source .env/bin/activate
python -m carps.analysis.gather_data \
	'--rundir=["runs_eval_icml/PPO-RNN*","runs_eval_icml/PPO-MLP*","runs_eval_icml/DefaultPolicy","runs_eval/Random","runs_eval_icml/Random","runs_eval/SAWEI-P"]' \
	--outdir=results_icml
