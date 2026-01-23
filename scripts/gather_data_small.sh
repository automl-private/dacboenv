#!/bin/bash
#SBATCH --job-name=gathers
#SBATCH --partition=normal
#SBATCH --time=3:00:00            
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB    
#SBATCH --output=gather_data/slurm-%j.out     # stdout log
#SBATCH --error=gather_data/slurm-%j.err      # stderr log

source .env/bin/activate
# python -m carps.utils.check_missing runs
# python -m carps.analysis.gather_data '--rundir=["runs_eval","/scratch/hpc-prf-intexml/tklenke/experiment_runs/dacboenv_ppo_semi"]'  --outdir=results

# sbatch scripts/generate_report.sh

# python -m carps.analysis.gather_data \
# 	'--rundir=["runs_eval/PPO-AlphaNet*","runs_eval/NoOpPolicy","runs_eval/SAWEI","runs_eval/SMAC-AC--dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled*","runs_eval/SMAC-AC--dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret*"]' \
# 	--outdir=results_alphanet2

python -m carps.analysis.gather_data \
	'--rundir=["runs_eval_icml","runs_eval/DefaultPolicy","runs_eval/Random","runs_eval/SAWEI-P"]' \
	--outdir=results_icml