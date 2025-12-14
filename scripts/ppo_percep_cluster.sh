#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "dacboenv"
#SBATCH --cpus-per-task=128
#SBATCH --mem=64G
#SBATCH -p normal

cd /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/scripts
source /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/.venv/bin/activate
python ppo_percep.py experiment.n_workers=128 "$@"