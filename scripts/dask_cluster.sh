#!/bin/bash
#SBATCH -t 48:00:00
#SBATCH -J "dacboenv"
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH -p normal

cd /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/scripts
source /scratch/hpc-prf-intexml/tklenke/repos/dacboenv/.venv/bin/activate
python piecewise.py "$@"