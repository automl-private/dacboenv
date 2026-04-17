# DQN with acquisition function selection
for seed in 1 2 3; do
    sbatch dqn_af_cluster.sh "" "$seed" 1 ""
    sbatch dqn_af_cluster.sh "" "$seed" 1 "SINGLE"
done

# DQN with bucket action space
for size in 1 8 20; do
    sbatch dqn_bucket_cluster.sh "" 1 "$size" ""
    sbatch dqn_bucket_cluster.sh "" 1 "$size" "SINGLE"
done

# DQN with step action space
for size in 1 8 20; do
    sbatch dqn_step_cluster.sh "" 1 "$size" "SMART"
done