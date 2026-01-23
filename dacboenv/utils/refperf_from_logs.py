"""Prepare reference performances."""

from __future__ import annotations

from pathlib import Path

import fire
from carps.analysis.gather_data import filelogs_to_df, normalize_logs
from carps.analysis.utils import filter_only_final_performance


def create_reference_performance(
    rundir: str = "runs_eval/DefaultPolicy",
    performance_path: str = "resultsdefault",
    reference_performance_fn: str | Path = "reference_performance.parquet",
) -> None:
    """Create reference performance dataframe from rundir.

    First collect all runs, then filter for final performance.

    Parameters
    ----------
    rundir : str, optional
        Run directory, contains raw results, by default "runs_eval/DefaultPolicy"
    performance_path : str, optional
        Where the performance data should be saved, by default "resultsdefault"
    reference_performance_fn : str | Path, optional
        Filename of the reference performance dataframe, by default "reference_performance.parquet"
    """
    logs, logs_cfg = filelogs_to_df(
        rundir=rundir,
        n_processes=1,
        outdir=performance_path,
    )
    logs = normalize_logs(logs)
    ref_fn: Path = Path(performance_path) / reference_performance_fn
    ref_fn.parent.mkdir(parents=True, exist_ok=True)
    reference_df = filter_only_final_performance(logs)
    reference_df.to_parquet(ref_fn)
    logs_cfg.to_parquet(ref_fn.with_stem(ref_fn.stem + "_cfg"))


if __name__ == "__main__":
    fire.Fire(create_reference_performance)
