#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -J "cma4dacbo"
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G
#SBATCH -p normal
#SBATCH --array=1-5

export HYDRA_FULL_ERROR=1
# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaled +instances=bbob2d_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaled +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaledpluslogregret +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaledpluslogregret_trialsleft +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random
python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaledpluslogregret_wei +instances=bbob2d_1_3seeds seed=$SLURM_ARRAY_TASK_ID +env/instance_selector=random



# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaledpluslogregret_trialsleft +instances=bbob2d_1_3seeds seed=1 +env/instance_selector=random n_workers=1
# python -m dacboenv.optimize_via_cma +opt=cma  +task=dacboenv_epdonescaled +instances=bbob2d_3seeds seed=1 +env/instance_selector=random n_workers=4 n_trials=100