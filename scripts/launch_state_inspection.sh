BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,6)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK20="+task/BBOB=cfg_2_20_0"
TASK8="+task/BBOB=cfg_2_8_0"
TASKALL="+task/BBOB=glob(cfg_2_*_0)"
CLUSTER="+cluster=cpu_noctua"

UCB_ACT="+env/action=ucb_beta_continuous"
WEI_ACT="+env/action=wei_alpha_continuous"

POLICY_DEFAULT="+policy=defaultaction"
POLICY_JUMP05="+policy=jump_05"
POLICY_RANDOM="+policy=random"

ENV="+env=base +env/opt=base +env/reward=ep_done_scaled +env/obs=all +env/refperf=defaultaction"
BASERUNDIR='baserundir=runs_statespace_icml/${action_space_id}'

# CLUSTER=""
# SEED="seed=1"
# TASKALL=$TASK8

# 1
python -m $BASE $SEED $TASKALL $ENV +eval=base $WEI_ACT $POLICY_DEFAULT $CLUSTER $BASERUNDIR --multirun &

# 2, 3
python -m $BASE $SEED $TASKALL +eval=base $ENV $WEI_ACT $POLICY_JUMP05 $CLUSTER $BASERUNDIR --multirun &

# # 4
python -m $BASE $SEED $TASK20 +eval=base $ENV $WEI_ACT $POLICY_RANDOM $CLUSTER $BASERUNDIR --multirun &
python -m $BASE $SEED $TASK20 +eval=base $ENV $UCB_ACT $POLICY_RANDOM $CLUSTER $BASERUNDIR --multirun &

wait