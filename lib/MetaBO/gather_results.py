import csv
import os
import sys
import pickle as pkl

base_logdir = "bbob/log"

for fid in range(1, 25):
    for dim in [2, 8]:

        env_dir = f"MetaBO-BBOB-{fid}-{dim}D-v0"
        env_path = os.path.join(base_logdir, env_dir)

        eval_base = os.path.join(env_path, "eval")
        if not os.path.exists(eval_base):
            continue
        
        inner_benchmarks = [f for f in os.listdir(eval_base) if f.startswith("MetaBO")]
        if not inner_benchmarks:
            continue
        
        for benchmark in inner_benchmarks:
            benchmark_path = os.path.join(eval_base, benchmark)
    
            result_files = [
                f for f in os.listdir(benchmark_path)
                if f.startswith("result_metabo")
            ]
            if not result_files:
                continue

            result_file = sorted(result_files)[-1]
            result_filepath = os.path.join(benchmark_path, result_file)

            with open(result_filepath, "rb") as f:
                result = pkl.load(f)

            rewards = result.rewards
            T = result.T
            n_episodes = result.n_episodes

            save_csv = os.path.join(benchmark_path, "rewards_per_timestep.csv")

            with open(save_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "timestep", "reward"])

                for ep in range(n_episodes):
                    for t in range(T):
                        idx = ep * T + t
                        writer.writerow([ep + 1, t + 1, rewards[idx]])

            print(f"Saved per-timestep rewards to: {save_csv}")

