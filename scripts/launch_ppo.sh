tasks=(
    # "+task=dacboenv_sawei_done"
    # "+task=dacboenv_sawei_symlog"
    # "+task=dacboenv_sawei_done_step"
    "+task=dacboenv_sawei_symlog_step"
)
ref_perfs=(
    "+env/refperf=saweip"
    # "+env/refperf=smacbb"
    # "+env/refperf=defaultaction"
)
instance_sets=(
    # "+instances=bbob2d_1_3seeds"
    # "+instances=bbob2d_20_3seeds"

    "+instances=ackley2_3seeds"
    # "+instances=bbob2d_8_3seeds"
    # "+instances=bbob2d_3seeds"
)
opts=(
    # "+opt/ppo=lstm"
    "+opt/ppo=lstm_obsnorm"
    # "+opt/ppo=mlp"
    # "+opt/ppo=mlp_obsnorm"
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
                bash scripts/opt_ppo.sh $instance_set $task $opt $ref_perf
            done
        done
    done
done