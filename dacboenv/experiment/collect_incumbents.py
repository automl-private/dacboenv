"""Collect incumbents specifying the DAC policy from rundir."""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from carps.analysis.gather_data import read_jsonl_content
from fire import Fire
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print as printr
from tqdm import tqdm

import dacboenv  # Load omegaconf resolvers  # noqa: F401
from dacboenv.utils.loggingutils import get_logger

if TYPE_CHECKING:
    from carps.utils.task import Task

logger = get_logger("collect_incs")


def add_metadata_to_dict(D: dict | pd.DataFrame, cfg: DictConfig) -> dict | pd.DataFrame:  # noqa: N803
    """Add metadata to result dict/dataframe.

    Parameters
    ----------
    D : dict | pd.DataFrame
        The result dict or dataframe.
    cfg : DictConfig
        The hydra config.

    Returns
    -------
    dict | pd.DataFrame
        Updated with metadata.
    """
    D["seed"] = cfg.seed
    D["task_id"] = cfg.task_id
    D["optimizer_id"] = cfg.optimizer_id
    D["objective_function"] = cfg.task.objective_function._target_.split(".")[-1]
    cfg_dict = OmegaConf.to_container(cfg)
    if isinstance(D, pd.DataFrame):
        D["config"] = [cfg_dict] * len(D)
    else:
        D["config"] = cfg_dict
    return D


def gather_data_smac(rundir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gather optimization data and incumbent policies from SMAC run.

    Parameters
    ----------
    rundir : Path
        The rundir of the optimization runs finding the policies.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Trajectories and incumbent configs.
    """
    config_fns = list(rundir.glob("**/config.yaml"))
    config_fns.sort()
    logger.info(f"Found {len(config_fns)} SMAC runs.")

    _intensifier_fn = "intensifier.json"
    _configspace_fn = "configspace.json"
    _optimization_fn = "optimization.json"
    _runhistory_fn = "runhistory.json"
    _scenario_fn = "scenario.json"

    trajectories = []
    configs_inc = []
    for _, cfg_fn in enumerate(config_fns):
        cfg = OmegaConf.load(cfg_fn)
        seed = cfg.seed

        smac_folders = list(cfg_fn.parent.parent.glob(str(Path(str(seed)) / "smac3_output" / "*" / str(seed))))
        assert len(smac_folders) == 1
        smac_folder = smac_folders[0]
        with open(smac_folder / _intensifier_fn) as file:
            intensifier_info = json.load(file)
        assert len(intensifier_info["incumbent_ids"]) == 1, "Multi-objective not supported or sth went wrong."
        incumbent_id = intensifier_info["incumbent_ids"][0]
        trajectory = pd.DataFrame(intensifier_info["trajectory"])
        trajectory["cost"] = trajectory["costs"].map(lambda x: x[0])
        trajectory["config_id"] = trajectory["config_ids"].map(lambda x: x[0])
        cost_inc = trajectory["cost"].min()
        del trajectory["costs"]
        del trajectory["walltime"]
        del trajectory["config_ids"]
        with open(smac_folder / _runhistory_fn) as file:
            runhistory_info = json.load(file)
        config_inc = runhistory_info["configs"][str(incumbent_id)]
        search_space_dim = len(config_inc)
        trajectory = add_metadata_to_dict(trajectory, cfg)
        trajectory["search_space_dim"] = search_space_dim
        trajectories.append(trajectory)

        config_inc = add_metadata_to_dict(config_inc, cfg)
        overrides = OmegaConf.load(cfg_fn.parent / "overrides.yaml")
        env_overrides = [o for o in overrides if "env" in o]
        env_overrides_str = " ".join(env_overrides)
        config_inc["env_override"] = env_overrides_str
        config_inc["cost"] = cost_inc
        config_inc["search_space_dim"] = search_space_dim
        configs_inc.append(config_inc)

    trajectory_df = pd.concat(trajectories).reset_index(drop=True)
    configs_inc_df = pd.DataFrame(configs_inc)

    save_traj_and_cincs_df(rundir=rundir, trajectory_df=trajectory_df, configs_inc_df=configs_inc_df)

    return trajectory_df, configs_inc_df


def extract_from_dict(logs: pd.DataFrame, dictkey: str, key: str) -> pd.Series:
    """Extract field from dict contained in a pandas column.

    Parameters
    ----------
    logs : pd.DataFrame
        The dataframe.
    dictkey : str
        The column name containing the dict.
    key : str
        The dict key.

    Returns
    -------
    pd.Series
        Extracted info from dict.
    """
    return logs[dictkey].map(lambda x: x[key])


def load_cma_log(filename: str | Path) -> pd.DataFrame:
    """Load CMA log.

    Parameters
    ----------
    filename : str | Path
        CMA log filename, mostly ends with `results.jsonl`.

    Returns
    -------
    pd.DataFrame
        The content of the log as a dataframe.
    """
    logs = read_jsonl_content(filename)
    logs["cost"] = extract_from_dict(logs, "trial_value", "cost")
    logs["trial_value__additional_info"] = extract_from_dict(logs, "trial_value", "additional_info")
    logs["episode_length"] = extract_from_dict(logs, "trial_value__additional_info", "episode_length")
    return logs.sort_values(by="n_generation")


def gather_data_cma(rundir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Gather optimization data and incumbent policies from CMA run.

    Parameters
    ----------
    rundir : Path
        The rundir of the optimization runs finding the policies.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Trajectories and incumbent configs.
    """
    log_filenames = list(rundir.glob("**/results.jsonl"))
    logger.info(f"Found {len(log_filenames)} CMA runs.")
    logs_list = []
    configs_inc_list = []
    configs_inc_keys = [
        "seed",
        "task_id",
        "optimizer_id",
        "objective_function",
        "config",
        "env_override",
        "cost",
        "search_space_dim",
    ]
    for log_filename in tqdm(log_filenames):
        _logs = load_cma_log(log_filename)
        cfg_fn = log_filename.parent / ".hydra/config.yaml"
        cfg = OmegaConf.load(cfg_fn)
        _logs = add_metadata_to_dict(_logs, cfg)
        logs_list.append(_logs)
        final_generation = _logs[_logs["n_generation"] == _logs["n_generation"].max()].copy()
        final_generation["hp_config"] = extract_from_dict(final_generation, "trial_info", "config")
        # Convert dicts to DataFrame
        expanded = final_generation["hp_config"].apply(pd.Series)
        # Rename columns to w0, w1, ...
        expanded.columns = [f"w{i}" for i in range(expanded.shape[1])]
        # Join back
        final_generation = final_generation.drop(columns="hp_config").join(expanded)
        keep_keys = [
            c
            for c in final_generation.columns
            if (c in configs_inc_keys or c.startswith("w")) and c not in ["worker_idx"]
        ]
        configs_inc = final_generation[keep_keys]
        overrides = OmegaConf.load(cfg_fn.parent / "overrides.yaml")
        env_overrides = [o for o in overrides if "env" in o]
        env_overrides_str = " ".join(env_overrides)
        configs_inc["env_override"] = env_overrides_str
        configs_inc_list.append(configs_inc)

    trajectory_df = pd.concat(logs_list).reset_index(drop=True)
    configs_inc_df = pd.concat(configs_inc_list).reset_index(drop=True)

    save_traj_and_cincs_df(rundir=rundir, trajectory_df=trajectory_df, configs_inc_df=configs_inc_df)

    return trajectory_df, configs_inc_df


def save_traj_and_cincs_df(rundir: Path, trajectory_df: pd.DataFrame, configs_inc_df: pd.DataFrame) -> None:
    """Save trajectory and incumbent configs dataframe.

    Parameters
    ----------
    rundir : Path
        The base directory to save to.
    trajectory_df : pd.DataFrame
        Trajecories.
    configs_inc_df : pd.DataFrame
        Incumbent configs.
    """
    trajectory_fn = rundir / "trajectory.csv"
    trajectory_df.to_csv(trajectory_fn, index=False)
    configs_inc_fn = rundir / "configs_inc.csv"
    configs_inc_df.to_csv(configs_inc_fn, index=False)
    logger.info(f"💌 Saved trajectory to {trajectory_fn} and incumbent configs to {configs_inc_fn}.")


def create_configs(rundir: Path) -> str:
    """Create policy configs.

    Parameters
    ----------
    rundir : Path
        The rundir of the optimization runs finding the policies.

    Returns
    -------
    str
        The policy path to the newly created configs. Contains policies from one outer optimizer and one task.
        Probably one config per seed.
    """
    configs_inc_fn = rundir / "configs_inc.csv"
    configs_inc_df = pd.read_csv(configs_inc_fn)
    configs_inc_df["config"] = configs_inc_df["config"].map(literal_eval).map(OmegaConf.create)

    for idx in range(len(configs_inc_df)):
        cfg = configs_inc_df.loc[idx, "config"]
        task: Task = instantiate(cfg.task)
        configspace = task.objective_function.configspace
        hp_names = list(configspace.keys())
        configuration_dict = dict(configs_inc_df.loc[idx, hp_names])
        policy = task.objective_function.make_policy_from_config_dict(config_dict=configuration_dict, seed=cfg.seed)
        printr(policy)

        policy_cls = policy.__class__
        policy_class_path = f"{policy_cls.__module__}.{policy_cls.__qualname__}"
        policy_kwargs = policy.get_init_kwargs()

        env_override = configs_inc_df.loc[idx, "env_override"]

        with initialize_config_module(
            config_module="dacboenv.configs",  # <-- package.conf where env/ lives
            version_base=None,
        ):
            cfg_env = compose(
                config_name=None,
                overrides=env_override.split(" "),
            )

        opt_cfg = DictConfig({})
        opt_cfg.optimizer = {}
        opt_cfg.optimizer.policy_class = {"_target_": policy_class_path, "_partial_": True}  # type: ignore[attr-defined]
        opt_cfg.optimizer.policy_kwargs = policy_kwargs  # type: ignore[attr-defined]
        opt_cfg.policy_id = f"{cfg.optimizer_id}--{cfg.task_id}--seed{cfg.seed}"
        opt_cfg.dacboenv = cfg_env.dacboenv
        opt_cfg.optimizer_id = opt_cfg.policy_id

        eval_cfg_fn = Path(f"dacboenv/configs/policy/optimized/{opt_cfg.policy_id.replace('--','/')}.yaml")
        eval_cfg_fn.parent.mkdir(parents=True, exist_ok=True)
        yaml_str = OmegaConf.to_yaml(opt_cfg)
        yaml_str = f"# @package _global_\n\n{yaml_str}"
        with open(eval_cfg_fn, "w") as file:
            file.write(yaml_str)
    return str(eval_cfg_fn.parent)


def collect(rundir: str = "runs") -> None:
    """Collect incumbents specifying the DAC policy from rundir.

    Parameters
    ----------
    rundir: str = "runs"
        The rundir containing the optimization runs or whatever.
    """
    _rundir = Path(rundir)
    traj_df_smac, cincs_df_smac = gather_data_smac(_rundir / "SMAC-AC")
    traj_df_cma, cincs_df_cma = gather_data_cma(_rundir / "CMA-1.3")
    trajectory_df = pd.concat([traj_df_smac, traj_df_cma]).reset_index(drop=True)
    configs_inc_df = pd.concat([cincs_df_smac, cincs_df_cma]).reset_index(drop=True)
    save_traj_and_cincs_df(rundir=_rundir, trajectory_df=trajectory_df, configs_inc_df=configs_inc_df)

    policy_path = create_configs(_rundir)
    logger.info(f"Ready to run `bash scripts/evaluate.sh $TASK {policy_path}")


if __name__ == "__main__":
    # python -m dacboenv.experiment.collect_incumbents "runs/SMAC-AC-CostInc"
    Fire(collect)
