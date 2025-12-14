#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "cma4dacbo"
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-5

python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaled +instances=bbob2d_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random