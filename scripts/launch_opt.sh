tasks=(
    "+task=dacboenv_epdonescaledpluslogregret_budgetpercentage"
    "+task=dacboenv_epdonescaledpluslogregret_wei"
    "+task=dacboenv_epdonescaledpluslogregret"
)
instance_sets=(
    "+instances=bbob2d_1_3seeds"
    "+instances=bbob2d_3seeds"
)

for task in "${tasks[@]}"
do
    for instance_set in "${instance_sets[@]}"
    do
        echo Launch for: $task $instance_set
        sbatch scripts/opt_with_cmaes.sh $task $instance_set
        sbatch scripts/opt_with_ac.sh $task $instance_set
    done
done