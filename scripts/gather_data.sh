#!/bin/bash
#SBATCH --job-name=gather
#SBATCH --time=7:00:00            
#SBATCH --cpus-per-task=48        
#SBATCH --output=slurm-%j.out     # stdout log
#SBATCH --error=gather_data/slurm-%j.err      # stderr log

source .env/bin/activate
# python -m carps.utils.check_missing runs
python -m carps.analysis.gather_data '--rundir=["runs_eval","/scratch/hpc-prf-intexml/tklenke/experiment_runs/dacboenv_ppo_semi"]'  --outdir=results

# sbatch scripts/generate_report.sh
