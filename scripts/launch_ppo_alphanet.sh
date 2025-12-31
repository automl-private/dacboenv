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

for task in "${tasks[@]}"
do
    for instance_set in "${instance_sets[@]}"
    do
        echo Launch for: $task $instance_set
        sbatch scripts/opt_ppo_norm_alphanet.sh $instance_set $task
    done
done