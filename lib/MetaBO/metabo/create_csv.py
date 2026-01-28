import csv
import os
import sys
import pickle as pkl

sys.path.append("") # XXX: MetaBO base path

base_logdir = "" # XXX: Insert

for fid in range(1, 25):
    for dim in [2, 8]:

        env_dir = f"MetaBO-BBOB-{fid}-{dim}D-v0"
        env_path = os.path.join(base_logdir, env_dir)

        top_folders = [
            f for f in os.listdir(env_path)
            if f.startswith("2026") and os.path.isdir(os.path.join(env_path, f))
        ]
        if not top_folders:
            print(f"No top-level folders found for {env_dir}")
            continue

        top_folder = sorted(top_folders)[-1]
        top_folder_path = os.path.join(env_path, top_folder)

        eval_base = os.path.join(top_folder_path, "eval")
        if not os.path.exists(eval_base):
            print(f"No eval folder in {top_folder_path}")
            continue

        timestamp_folders = [
            f for f in os.listdir(eval_base)
            if os.path.isdir(os.path.join(eval_base, f))
        ]
        if not timestamp_folders:
            print(f"No timestamp folders in {eval_base}")
            continue

        latest_eval = sorted(timestamp_folders)[-1]
        eval_path = os.path.join(eval_base, latest_eval)
        
        inner_benchmarks = [f for f in os.listdir(eval_path) if f.startswith("MetaBO")]
        if not inner_benchmarks:
            print("No inner benchmarks found")
            continue
        
        for benchmark in inner_benchmarks:
            benchmark_path = os.path.join(eval_path, benchmark)
    
            result_files = [
                f for f in os.listdir(benchmark_path)
                if f.startswith("result_metabo")
            ]
            if not result_files:
                print(f"No result_metabo file found in {benchmark_path}")
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

