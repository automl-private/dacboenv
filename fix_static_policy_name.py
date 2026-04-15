import os
from pathlib import Path
from omegaconf import OmegaConf

def fix_static_policy_naming(root_dir: str):
    # Path to search for: runs_static/**/.hydra/config.yaml
    root_path = Path(root_dir)
    config_files = list(root_path.glob("**/.hydra/config.yaml"))

    print(f"Found {len(config_files)} config files to check...")

    for cfg_path in config_files:
        try:
            # 1. Load the config
            conf = OmegaConf.load(cfg_path)

            # 2. Check if the key exists and matches the pattern
            # Based on your pattern: optimizer_id: StaticPolicy-${optimizer.policy_kwargs.par_val}
            # Note: OmegaConf handles the interpolation automatically.

            if "optimizer_id" in conf:
                current_id = conf.optimizer_id

                if current_id.startswith("StaticPolicy"):
                    par_val = conf.optimizer.policy_kwargs.par_val
                    new_id = f"StaticPolicy-{par_val}"

                    # Update the value
                    conf.optimizer_id = new_id

                    # 3. Save the file back
                    OmegaConf.save(config=conf, f=cfg_path)
                    print(f"Updated: {cfg_path} with {new_id}")
                else:
                    # Skip if it doesn't match the target pattern
                    continue

        except Exception as e:
            print(f"Error processing {cfg_path}: {e}")

if __name__ == "__main__":
    # Point this to your runs_static folder
    target_folder = "runs_static"

    if os.path.exists(target_folder):
        fix_static_policy_naming(target_folder)
        print("Processing complete.")
    else:
        print(f"Folder '{target_folder}' not found.")