BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK20="+task/BBOB=cfg_2_20_0"
TASK8="+task/BBOB=cfg_2_8_0"
TASKALL="+task/BBOB=glob(cfg_2_*_0)"
CLUSTER="+cluster=cpu_noctua"
export HYDRA_FULL_ERROR=1

# python -m $BASE $TASK1 seed=2 +eval=base +env=base +env/opt=base \
#     +env/reward=ep_done_scaled +env/obs=sawei +env/action=wei_alpha_continuous +policy=alpharulenet \
#     baserundir=tmp_runs_eval

SLURM_ARRAY_TASK_ID=3
TASK_OVERRIDE="+task=dacboenv_sawei"
INSTANCE_SET_OVERRIDE="+instances=bbob2d_3seeds"

python -m $BASE seed=$SLURM_ARRAY_TASK_ID +opt=smac \
    $TASK_OVERRIDE $INSTANCE_SET_OVERRIDE \
    optimizer.smac_cfg.scenario.n_workers=1 \
    baserundir=tmp_runs_opt
