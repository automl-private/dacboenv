"""Merge reference performances."""

from __future__ import annotations

from pathlib import Path

import fire
import numpy as np
import pandas as pd
from carps.analysis.gather_data import normalize_logs
from carps.analysis.utils import filter_only_final_performance


def merge_logs(
    logs_list: list[pd.DataFrame], logs_cfg_list: list[pd.DataFrame], outdir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge logs, keep configs correct.

    Parameters
    ----------
    logs_list : list[pd.DataFrame]
        The list of logs to merge.
    logs_cfg_list : list[pd.DataFrame]
        The list of config logs to merge.
    outdir : str | Path
        The target outdir.

    Returns
    -------
    tuple[pd.DataFrame,pd.DataFrame]
        Merged logs, merged config logs
    """
    offsets = [len(l) for l in logs_cfg_list]  # noqa: E741
    offsets = np.concatenate((np.array([0]), np.cumsum(offsets)))

    for i in range(len(logs_cfg_list)):
        logs_cfg_list[i]["experiment_id"] += offsets[i]
        logs_list[i]["experiment_id"] += offsets[i]

    logs = pd.concat(logs_list)
    logs_cfg = pd.concat(logs_cfg_list)

    assert len(logs_cfg), logs_cfg["experiment_id"].nunique()

    logs.to_parquet(Path(outdir) / "logs.parquet")
    logs_cfg.to_parquet(Path(outdir) / "logs_cfg.parquet")
    return logs, logs_cfg


def append_to_ref_performance(
    performance_path: str = "resultssawei",
    reference_performance_path_old: str = "reference_performance",
    reference_performance_path: str = "reference_performance_merged",
    reference_performance_fn: str | Path = "reference_performance.parquet",
) -> None:
    """Append to reference performance and save at new location.

    Parameters
    ----------
    performance_path : str, optional
        Results to add, can contain any optimizer, external to carps, by default "resultssawei"
    reference_performance_path_old : str, optional
        The old reference performance path, by default "reference_performance"
    reference_performance_path : str, optional
        The new reference performance location, by default "reference_performance_merged"
    reference_performance_fn : str | Path, optional
        The new reference performance filename, by default "reference_performance.parquet"
    """
    logs_new = pd.read_parquet(Path(performance_path) / "logs.parquet")
    logs_cfg_new = pd.read_parquet(Path(performance_path) / "logs_cfg.parquet")

    logs_ref = pd.read_parquet(Path(reference_performance_path_old) / "logs.parquet")
    logs_cfg_ref = pd.read_parquet(Path(reference_performance_path_old) / "logs_cfg.parquet")

    merge_dir = Path()
    merge_dir.mkdir(exist_ok=True, parents=True)

    logs, logs_cfg = merge_logs(
        logs_list=[logs_ref, logs_new], logs_cfg_list=[logs_cfg_ref, logs_cfg_new], outdir=merge_dir
    )
    logs = normalize_logs(logs)
    ref_fn: Path = Path(reference_performance_path) / reference_performance_fn
    ref_fn.parent.mkdir(parents=True, exist_ok=True)
    reference_df = filter_only_final_performance(logs)
    reference_df.to_parquet(ref_fn)
    logs_cfg.to_parquet(ref_fn.with_stem(ref_fn.stem + "_cfg"))


if __name__ == "__main__":
    fire.Fire(append_to_ref_performance)
