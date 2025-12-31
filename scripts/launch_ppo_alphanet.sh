tasks=(
    # "+task=dacboenv_epdonescaledpluslogregret_budgetpercentage"
    # "+task=dacboenv_epdonescaledpluslogregret_budgetpercentage_wei"
    # "+task=dacboenv_epdonescaledpluslogregret_wei"
    # "+task=dacboenv_epdonescaledpluslogregret"
    # "+task=dacboenv_sawei_done"
    "+task=dacboenv_sawei_symlog"
)
instance_sets=(
    # "+instances=bbob2d_1_3seeds"
    # "+instances=bbob2d_20_3seeds"
    "+instances=bbob2d_8_3seeds"
    "+instances=bbob2d_3seeds"
)
opts=(
    "+opt=ppo_alphanet"
    "+opt=ppo_alphanet2"
    "+opt=ppo_alphanet3"
)

for task in "${tasks[@]}"
do
    for instance_set in "${instance_sets[@]}"
    do
        for opt in "${opts[@]}"
        do
            echo Launch for: $task $instance_set $opt
            sbatch scripts/opt_ppo_norm_alphanet.sh $instance_set $task $opt
        done
    done
done