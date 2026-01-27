"""Helper functions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ioh
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

RWBM = "nasengb"  # Name of the real-world benchmark. It was BNNBO before

tqdm.pandas()


def fix_subset_id(x: str) -> str:
    """Fix benchmark/subset ID.

    Parameters
    ----------
    x : str
        The ID.

    Returns
    -------
    str
        The formatted ID.
    """
    if x in {"hpobench", "HPOBENCH", "HPOBench"}:
        return "HPOBench"
    if x == "BNNBO":
        return RWBM
    if x == "bnnbo":
        return RWBM
    if x == RWBM:
        return RWBM
    return x.upper()


def process_task_id_for_grid(x: str) -> str:
    """Process task ids for nicer grid plotting.

    Parameters
    ----------
    x : str
        Task id.

    Returns
    -------
    str
        Modified task id per domain.

    Raises
    ------
    ValueError
        When the domain is unknown.
    """
    if x.startswith("blackbox"):
        return "/".join(x.split("/")[3:])
    if x.startswith("bbob"):
        dim, fid, _inst = x.split("/")[1:]
        return f"fid={int(fid):02d}, d={dim}"
    if x.startswith("yahpo"):
        search_space, openml_dataset_id = x.split("/")[2:-1]
        return f"cs={search_space}\ndid={openml_dataset_id}"
    if x.startswith("hpobench"):
        return "/".join(x.split("/")[2:])
    if x.startswith("bnnbo"):
        return "/".join(x.split("/")[1:])
    raise ValueError(f"Unknown task id {x}")


def add_log_regret(gdf: pd.DataFrame) -> pd.DataFrame:
    """Add log regret to BBOB tasks.

    Parameters
    ----------
    gdf : pd.DataFrame
        The dataframe containing only results on BBOB.

    Returns
    -------
    pd.DataFrame
        The dataframe with the log regret.
    """
    task_id = gdf.name
    fid = int(task_id.split("/")[2])
    inst = int(task_id.split("/")[3])
    func = ioh.get_problem(fid=fid, instance=inst)
    gdf.loc[:, "regret"] = gdf.loc[:, "trial_value__cost_inc"] - func.optimum.y
    gdf["log_regret"] = gdf["regret"].map(np.log)
    gdf["f_min"] = func.optimum.y
    return gdf


# HPOBench related
models = ["lr", "svm", "xgb", "nn", "rf"]
models_reduced = ["nn", "rf"]
open_ml_ids = [31, 53, 3917, 9952, 10101, 146818, 146821, 146822]


def filter_by_sawei(task_id: str) -> bool:
    """Filter HPOBench tasks by the tasks used in SAWEI paper.

    Parameters
    ----------
    task_id : str
        The task id.

    Returns
    -------
    bool
        Whether it belongs to the SAWEI experiment set.
    """
    if "tabular/ml" in task_id:
        model = task_id.split("/")[-2]
        open_ml_id = int(task_id.split("/")[-1])
        if model in models_reduced and open_ml_id in open_ml_ids:
            return True
    return False


def determine_hpobench_tasktype(x: str) -> str:
    """Subdivide HPOBench tasktypes.

    Parameters
    ----------
    x : str
        The task id.

    Returns
    -------
    str
        The update task id.

    Raises
    ------
    ValueError
        When neither 'tabular/ml', 'tabular/nas' nor 'surrogate' in task id.
    """
    if "surrogate" in x:
        return "surrogate"
    if "tabular/ml" in x:
        model = x.split("/")[-2]
        open_ml_id = int(x.split("/")[-1])
        if model in models and open_ml_id in open_ml_ids:
            return "tabular/ml SAWEI"
        return "tabular/ml"
    if "tabular/nas" in x:
        return "tabular/nas"
    raise ValueError(f"cannot determine task type for {x}")


def create_yaml_string(raw_str: str) -> str:
    r"""Convert a raw config string with escaped \n sequences into a proper YAML file.

    Lines containing ':' in the value are automatically quoted to avoid ScannerError.

    Args:
        raw_str (str): The raw configuration string (with \\n as line breaks).
    """
    # Step 1: Replace escaped \n with real newlines
    yaml_str = raw_str.replace("\\n", "\n")

    # Step 2: Quote any line with a colon in the value (after the key)
    yaml_lines = yaml_str.splitlines()
    for i, line in enumerate(yaml_lines):
        if ":" in line:
            # split at the first colon
            key, val = line.split(":", 1)
            val = val.strip()
            # quote if value contains colon or dollar sign (common in interpolations)
            if ":" in val or val.startswith("${"):
                yaml_lines[i] = f'{key}: "{val}"'
    return "\n".join(yaml_lines)


def extract_yahpo_info(cfg_str: str) -> dict:
    """Extract YAHPO info about task.

    Parameters
    ----------
    cfg_str : str
        The config string containing the config for a yahpo task.

    Returns
    -------
    dict
        'bench', 'instance', 'metric'.
    """
    yaml_str = create_yaml_string(cfg_str)
    cfg = OmegaConf.create(yaml_str)

    info = {
        "bench": cfg.task.objective_function.bench,
        "instance": cfg.task.objective_function.instance,
        "metric": list(cfg.task.objective_function.metric),  # use pure python types
    }
    index = list(info.keys())
    values = list(info.values())
    values[-1] = values[-1][0]
    yahpo_so_fmin_df = pd.read_csv(Path(__file__).parent / "yahpo_so_fmin.csv")
    yahpo_so_fmin_df = yahpo_so_fmin_df.set_index(index)
    info["f_min"] = yahpo_so_fmin_df.loc[tuple(values), "f_min"]

    return info


def add_yahpo_log_regret(row: pd.Series) -> float | None:
    """Add log regret based on metric.

    Only works for 'acc' and 'val_accuracy'.

    Parameters
    ----------
    row : pd.Series
        The row containing one data point.

    Returns
    -------
    float | None
        The log regret.
    """
    metrics = row["metric"]
    if (isinstance(metrics, list | ListConfig) and len(metrics) == 1) or isinstance(metrics, str):
        metric = metrics[0] if isinstance(metrics, list | ListConfig) else metrics
        performance = row["trial_value__cost_inc"]
        if metric == "acc":
            return np.log(np.abs(-1 - performance) + 1e-10)
        if metric == "val_accuracy":
            return np.log(np.abs(-100 - performance) + 1e-10)
    else:
        print(type(metrics))
    return None


def extract_cfg_info(
    data: pd.DataFrame, logs_cfg: pd.DataFrame, func_to_extract_info: Callable[[str], dict]
) -> pd.DataFrame:
    """Extract info from the config for each task and add to general logs.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe containing the run logs in carps format. Should contain
        a unique `experiment_id`.
    logs_cfg : pd.DataFrame
        The associated configuration per `experiment_id`.
    func_to_extract_info : Callable[[str], dict]
        Extracts info from a config string.

    Returns
    -------
    pd.DataFrame
        Data with added info.
    """
    # Map experiment_id to cfg_str
    data["cfg_str"] = logs_cfg.iloc[data["experiment_id"].astype(int).to_list()]["cfg_str"].to_list()

    # Extract unique task_ids with their cfg_str
    unique_tasks = data[["task_id", "cfg_str"]].drop_duplicates(subset="task_id").reset_index(drop=True)
    # Extract info once per unique task_id with progress bar
    unique_tasks["info"] = unique_tasks["cfg_str"].progress_map(func_to_extract_info)

    # Expand the info dict into columns
    info_df = pd.DataFrame(unique_tasks["info"].to_list())

    # Combine task_id with extracted info
    unique_tasks = pd.concat([unique_tasks[["task_id"]], info_df], axis=1)

    # Merge back to the original data
    data = data.merge(unique_tasks, on="task_id", how="left")

    # Optional: drop cfg_str if not needed
    return data.drop(columns=["cfg_str"])


def postprocess_yahpo(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:
    """Postprocess YAHPO rundata (add log regret).

    So far, works for every task containing 'acc' and 'val_accuracy'.

    Parameters
    ----------
    data : pd.DataFrame
        The data containing YAHPO runs.
    logs_cfg : pd.DataFrame
        The exact configurations per experiment.

    Returns
    -------
    pd.DataFrame
        YAHPO with task info and log regret.
    """
    data = extract_cfg_info(data=data, logs_cfg=logs_cfg, func_to_extract_info=extract_yahpo_info)

    data.loc[:, "task_id"] = data["task_id"].str.replace("yahpo/so/", "")

    ids_val_acc = data["metric"] == "val_accuracy"
    data.loc[ids_val_acc, "trial_value__cost_inc"] /= 100
    data.loc[ids_val_acc, "f_min"] /= 100

    # Add log regret
    # data["log_regret"] = data.progress_apply(add_yahpo_log_regret, axis=1)
    data = data.groupby("task_id").apply(calc_fmin).reset_index(drop=True)
    data["regret"] = data["trial_value__cost_inc"] - data["f_min"]
    data["log_regret"] = calc_log_regret(data["trial_value__cost_inc"], data["f_min"])

    return data


def extract_bnnbo_info(cfg_str: str) -> dict:
    """Extract info for RWBM benchmark.

    Parameters
    ----------
    cfg_str : str
        The configuration yaml str.

    Returns
    -------
    dict
        The info dict, containing `f_min`.
    """
    yaml_str = create_yaml_string(cfg_str)
    cfg = OmegaConf.create(yaml_str)
    task_id = cfg.task_id
    index = "task_id"
    bnnbo_fmin_df = pd.read_csv(Path(__file__).parent / "bnnbo_fmin.csv")
    bnnbo_fmin_df = bnnbo_fmin_df.set_index(index)
    f_min = bnnbo_fmin_df.loc[task_id, "f_min"] if task_id in bnnbo_fmin_df.index else None
    if f_min is None:
        # The optimal values count for the constrained tasks.
        # botorch_problem = instantiate(cfg.task.objective_function.botorch_problem)
        # f_min = botorch_problem._optimal_value if hasattr(botorch_problem, "_optimal_value") else None
        pass
    return {"f_min": f_min}


def calc_fmin(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate the global minimum.

    If the value is not present, estimate it by the lowest observed function value.
    Intended to be used as a groupby function.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing results, should only contain results for one task.

    Returns
    -------
    pd.DataFrame
        Updated dataframe.
    """
    f_min_opt = data["trial_value__cost_inc"].min()
    if data["f_min"].iloc[0] is None:
        data["f_min"] = f_min_opt
    else:
        data["f_min"] = data["f_min"].map(lambda x: min(x, f_min_opt))
    return data


def extract_optbench_info(cfg_str: str) -> dict:
    """Extract info for OptBench benchmark.

    Parameters
    ----------
    cfg_str : str
        The configuration yaml str.

    Returns
    -------
    dict
        The info dict, containing `f_min`.
    """
    yaml_str = create_yaml_string(cfg_str)
    cfg = OmegaConf.create(yaml_str)
    task = instantiate(cfg.task)
    f_min = task.objective_function.f_min
    return {"f_min": f_min}


def calc_fmax(data: pd.DataFrame) -> pd.DataFrame:
    """Estimate the function maximum by the maximum observed value.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with results, should only contain results for one task.

    Returns
    -------
    pd.DataFrame
        The updated dataframe with `f_max`.
    """
    data["f_max"] = data["trial_value__cost"].max()
    return data


def postprocess_bnnbo(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:
    """Postprocess RWBM results by calculating the log regret.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with RWBM results.
    logs_cfg : pd.DataFrame
        The logs containing the configurations of the experiment runs.

    Returns
    -------
    pd.DataFrame
        The dataframe with `log_regret`.
    """
    data = extract_cfg_info(data=data, logs_cfg=logs_cfg, func_to_extract_info=extract_bnnbo_info)
    # data["f_min"] = None
    data = data.groupby("task_id").apply(calc_fmin).reset_index(drop=True)
    data["regret"] = data["trial_value__cost_inc"] - data["f_min"]
    data["log_regret"] = calc_log_regret(data["trial_value__cost_inc"], data["f_min"])
    return data


def postprocess_bbob(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001
    """Postprocess BBOB results by calculating the log regret.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with BBOB results.
    logs_cfg : pd.DataFrame
        The logs containing the configurations of the experiment runs.

    Returns
    -------
    pd.DataFrame
        The dataframe with `log_regret`.
    """
    return data.groupby("task_id").apply(add_log_regret).reset_index(drop=True)


def postprocess_optbench(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:
    """Postprocess OptBench results by calculating the log regret.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with BBOB results.
    logs_cfg : pd.DataFrame
        The logs containing the configurations of the experiment runs.

    Returns
    -------
    pd.DataFrame
        The dataframe with `log_regret`.
    """
    data = extract_cfg_info(data=data, logs_cfg=logs_cfg, func_to_extract_info=extract_optbench_info)

    # Add log regret
    # data["log_regret"] = data.progress_apply(add_yahpo_log_regret, axis=1)
    data = data.groupby("task_id").apply(calc_fmin).reset_index(drop=True)
    data["regret"] = data["trial_value__cost_inc"] - data["f_min"]
    data["log_regret"] = calc_log_regret(data["trial_value__cost_inc"], data["f_min"])
    return data


def fix_tabnas_metric_range(x: pd.DataFrame) -> pd.DataFrame:
    """Fix the metric range for HPOBench tabular/nas tasks.

    From 0-100 to 0-1.

    Parameters
    ----------
    x : pd.DataFrame
        Results containing only one HPOBench task type.

    Returns
    -------
    pd.DataFrame
        Updated dataframe with the value range of tabular/nas fixed.
    """
    if x["tasktype"].iloc[0] == "tabular/nas":
        x["trial_value__cost_inc"] /= 100
    return x


def calc_log(x: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Calculate the logarithm with base 10 in a safe way.

    Parameters
    ----------
    x : np.ndarray | pd.Series
        Input array.

    Returns
    -------
    np.ndarray | pd.Series
        Log10(x)
    """
    eps = np.finfo(float).eps
    return np.log10(np.clip(x, eps, None))


def calc_log_regret(costs: pd.Series, f_mins: pd.Series) -> pd.Series:
    """Calculate the log regret.

    Parameters
    ----------
    costs : pd.Series
        The cost vector.
    f_mins : pd.Series
        The function minima.

    Returns
    -------
    _type_
        _description_
    """
    diff = np.abs(costs - f_mins)
    return calc_log(diff)  # pyright: ignore[reportReturnType]


def postprocess_hpobench(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:  # noqa: ARG001
    """Postprocess HPOBench results by calculating the log regret.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with HPOBench results.
    logs_cfg : pd.DataFrame
        The logs containing the configurations of the experiment runs.

    Returns
    -------
    pd.DataFrame
        The dataframe with `log_regret`.
    """
    data["task_id"] = data["task_id"].str.replace("hpobench/blackbox/", "")
    data["tasktype"] = data["task_id"].map(determine_hpobench_tasktype)
    data = data.groupby("tasktype").apply(fix_tabnas_metric_range).reset_index(drop=True)
    data["f_min"] = None
    data = data.groupby("task_id").apply(calc_fmin).reset_index(drop=True)
    data["regret"] = data["trial_value__cost_inc"] - data["f_min"]
    data["log_regret"] = calc_log_regret(data["trial_value__cost_inc"], data["f_min"])
    return data


def postprocess_benchmarks(data: pd.DataFrame, logs_cfg: pd.DataFrame) -> pd.DataFrame:
    """Postprocess benchmarks (add log regret).

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with results.
    logs_cfg : pd.DataFrame
        The configurations of the experiment runs.

    Returns
    -------
    pd.DataFrame
        The updated dataframe with `log_regret`.
    """

    def _postprocess(data: pd.DataFrame) -> pd.DataFrame:
        assert data["benchmark_id"].nunique() == 1, data["benchmark_id"].unique()  # noqa: PD101
        benchmark_id = data["benchmark_id"].iloc[0]
        print(f"Processing {benchmark_id}...")
        if benchmark_id == "BBOB":
            return postprocess_bbob(data=data, logs_cfg=logs_cfg)
        if benchmark_id == "HPOBench":
            return postprocess_hpobench(data=data, logs_cfg=logs_cfg)
        if benchmark_id == RWBM:
            return postprocess_bnnbo(data=data, logs_cfg=logs_cfg)
        if benchmark_id == "YAHPO":
            return postprocess_yahpo(data=data, logs_cfg=logs_cfg)
        if benchmark_id == "OptBench":
            return postprocess_optbench(data=data, logs_cfg=logs_cfg)
        raise ValueError(f"Unknown benchmark {benchmark_id}")

    data.loc[data["benchmark_id"] == "BNNBO", "benchmark_id"] = RWBM
    data = data.groupby("task_id").apply(calc_fmax).reset_index(drop=True)
    return data.groupby("benchmark_id").apply(_postprocess).reset_index(drop=True)


def sort_df_by_mean(data: pd.DataFrame, key_performance: str = "trial_value__cost_inc_norm") -> pd.DataFrame:
    """Sort the dataframe by the mean performance.

    Parameters
    ----------
    data : pd.DataFrame
        The dataframe with results.
    key_performance : str, optional
        The performance key to aggregate and sort by, by default "trial_value__cost_inc_norm"

    Returns
    -------
    pd.DataFrame
        The sorted dataframe
    """
    df_mean = data.groupby("optimizer_id")[key_performance].apply(np.nanmean).sort_values(ascending=False)
    sorter = list(df_mean.index)
    return data.sort_values(by="optimizer_id", key=lambda s: s.map({v: i for i, v in enumerate(sorter)}))
