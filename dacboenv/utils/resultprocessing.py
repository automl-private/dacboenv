"""Read DAC-BO env logs."""

from __future__ import annotations

from pathlib import Path

import fire
import pandas as pd
import tqdm
from omegaconf import OmegaConf


def read_env_logs(path: str) -> None:
    """Read env logs.

    Reads all DACBOEnvLogs.jsonl files in path and subdirectories,
    merges with DACBOEnvActions.jsonl files, adds config info, and saves as envlogs.csv

    Args:
        path (str): Path to directory containing DACBOEnvLogs.jsonl files. Can contain several run dirs.
    """
    # Read logs and create df

    dacboenv_jsons = list(Path(path).rglob("DACBOEnvLogs.jsonl"))
    json_dfs = []

    for j in tqdm.tqdm(dacboenv_jsons, total=len(dacboenv_jsons)):
        tmp = pd.read_json(j, lines=True)
        tmp["source"] = j
        action_log = j.parent / "DACBOEnvActions.jsonl"
        action_df = pd.read_json(action_log, lines=True)
        tmp = tmp.merge(action_df, on=["n_trials"])

        config_fn = j.parent / ".hydra/config.yaml"
        config = OmegaConf.load(config_fn)
        tmp["optimizer_id"] = config.optimizer_id
        tmp["seed"] = config.seed
        tmp["task_id"] = config.task_id

        json_dfs.append(tmp)

    dacboenv_df = pd.concat(json_dfs).reset_index(drop=True)
    dacboenv_df.to_csv(f"{path}/envlogs.csv")


if __name__ == "__main__":
    fire.Fire(read_env_logs)
