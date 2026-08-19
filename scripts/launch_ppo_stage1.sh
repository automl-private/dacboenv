for training_config in mixed_wei_ppo_stage_b; do
    sbatch scripts/opt_ppo_stage1.sh "+training=${training_config}" \
     "baserundir=runs_yahpoMixed_stageb"
done