import pathlib
import re
import shutil

import fire


def cleanup_runs(target_dir: str, debug: bool = False):
    """
    Cleans up experiment directories by removing logs, old checkpoints, and folders.

    :param target_dir: The root directory to search for run folders.
    :param debug: If True, prints what would be deleted without deleting them.
    """
    root = pathlib.Path(target_dir).resolve()

    if not root.is_dir():
        print(f"Error: {target_dir} is not a valid directory.")
        return

    # Find all directories containing a .hydra folder
    run_dirs = [p.parent for p in root.rglob(".hydra") if p.is_dir()]

    if not run_dirs:
        print("No run directories (containing .hydra) were found.")
        return

    for run_dir in run_dirs:
        print(f"Processing run directory: {run_dir}")

        # --- 1. Remove the entire smac3_output folder ---
        smac_folder = run_dir / "smac3_output"
        if smac_folder.is_dir():
            _delete_path(smac_folder, debug)

        # --- 2. Remove old rl_model_X_steps.zip files ---
        model_pattern = re.compile(r"rl_model_(\d+)_steps\.zip")

        model_files = []
        for f in run_dir.glob("rl_model_*_steps.zip"):
            match = model_pattern.match(f.name)
            if match:
                step_count = int(match.group(1))
                model_files.append((step_count, f))

        if model_files:
            model_files.sort(key=lambda x: x[0])
            to_delete = model_files[:-1]

            for _, file_path in to_delete:
                _delete_path(file_path, debug)

def _delete_path(path: pathlib.Path, debug: bool):
    """Helper to handle deletion of files or directories."""
    path_type = "directory" if path.is_dir() else "file"

    if debug:
        print(f"[DEBUG] Would remove {path_type}: {path}")
    else:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"Removed {path_type}: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")

if __name__ == "__main__":
    fire.Fire(cleanup_runs)