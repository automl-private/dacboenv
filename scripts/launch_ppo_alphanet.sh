tasks=(
    # "+task=dacboenv_epdonescaledpluslogregret_budgetpercentage"
    # "+task=dacboenv_epdonescaledpluslogregret_budgetpercentage_wei"
    # "+task=dacboenv_epdonescaledpluslogregret_wei"
    # "+task=dacboenv_epdonescaledpluslogregret"

    "+task=dacboenv_sawei_done"
    "+task=dacboenv_sawei_symlog"
)
ref_perfs=(
    # "+env/refperf=saweip"
    # "+env/refperf=smacbb"
    "+env/refperf=noop"
)
instance_sets=(
    # "+instances=bbob2d_1_3seeds"
    # "+instances=bbob2d_20_3seeds"

    "+instances=bbob2d_8_3seeds"
    "+instances=bbob2d_3seeds"
)
opts=(
    "+opt=ppo_alphanet"

    # "+opt=ppo_alphanet2"
    # "+opt=ppo_alphanet3"
)

for task in "${tasks[@]}"
do
    for ref_perf in "${ref_perfs[@]}"
    do
        for instance_set in "${instance_sets[@]}"
        do
            for opt in "${opts[@]}"
            do
                echo Launch for: $task $instance_set $opt $ref_perf
                sbatch scripts/opt_ppo_norm_alphanet.sh $instance_set $task $opt $ref_perf
            done
        done
    done
done