BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK20="+task/BBOB=cfg_2_20_0"
TASK8="+task/BBOB=cfg_2_8_0"
TASKALL="+task/BBOB=glob(cfg_2_*_0)"
CLUSTER="+cluster=cpu_noctua"

python -m $BASE $TASK1 seed=2 +base=eval +env/opt=base +env/action=ucb_beta_continuous +env/obs=smart +env/reward=ep_done_scaled +policy=random