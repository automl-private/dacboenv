export HYDRA_FULL_ERROR=1

tasks=(
    # "+task=dacboenv_sawei_done"
    "+task=dacboenv_sawei_symlog"
    "+task=dacboenv_sawei_symlogop"
    # "+task=dacboenv_all_symlog"
    # "+task=dacboenv_sawei_done_step"
    # "+task=dacboenv_sawei_symlog_step"
    # "+task=dacboenv_sawei_symlog_skip"
    # "+task=dacboenv_all_symlog_skip"    
    # "+task=dacboenv_sawei_done_skip"
)
ref_perfs=(
    "+env/refperf=saweip"
    # "+env/refperf=defaultaction"
    # "+env/refperf=smacbb"
)
instance_sets=(
    # "+instances=ackley2_3seeds"
    # "+instances=bbob2d_8_3seeds"
    # "+instances=bbob2d_3seeds"
    # "+instances=selected_random"
    # "+instances=sel2_random"
    "+instances=yahpo_glmnet375_3seeds"
    "+instances=yahpo_glmnet375_random"
)
opts=(
    # "+opt/ppo=lstm"
    # "+opt/ppo=lstm_obsnorm"
    "+opt/ppo=mlp"
    # "+opt/ppo=mlp_default"
    # "+opt/ppo=mlp_default_obsnorm"
    "+opt/ppo=mlp_obsnorm"
)
interaction_frequencies=(
    # "+env/interaction_freq=f1"
    "+env/interaction_freq=f5"
    "+env/interaction_freq=f10"
)

for task in "${tasks[@]}"
do
    for ref_perf in "${ref_perfs[@]}"
    do
        for instance_set in "${instance_sets[@]}"
        do
            for interaction_freq in "${interaction_frequencies[@]}"
            do
                for opt in "${opts[@]}"
                do
                    echo Launch for: $task $instance_set $opt $ref_perf $interaction_freq
                    sbatch scripts/opt_ppo.sh $instance_set $task $opt $ref_perf $interaction_freq
                done
            done
        done
    done
done
