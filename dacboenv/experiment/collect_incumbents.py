"""Collect incumbents specifying the DAC policy from rundir."""

from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
from fire import Fire
from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich import print as printr

import dacboenv  # Load omegaconf resolvers  # noqa: F401
from dacboenv.utils.loggingutils import get_logger

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
    D["config"] = cfg_dict
    return D


def gather_data(rundir: Path) -> None:
    """Gather optimization data and incumbent policies.

    Parameters
    ----------
    rundir : Path
        The rundir of the optimization runs finding the policies.
    """
    config_fns = list(rundir.glob("**/config.yaml"))
    config_fns.sort()
    logger.info(f"Found {len(config_fns)} optimization runs.")

    _intensifier_fn = "intensifier.json"
    _configspace_fn = "configspace.json"
    _optimization_fn = "optimization.json"
    _runhistory_fn = "runhistory.json"
    _scenario_fn = "scenario.json"

    trajectories = []
    configs_inc = []
    experiment_configs = []
    for experiment_id, cfg_fn in enumerate(config_fns):
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
        del trajectory["costs"]
        del trajectory["walltime"]
        del trajectory["config_ids"]
        trajectory = add_metadata_to_dict(trajectory, cfg)
        trajectories.append(trajectory)

        with open(smac_folder / _runhistory_fn) as file:
            runhistory_info = json.load(file)
        config_inc = runhistory_info["configs"][str(incumbent_id)]

        config_inc = add_metadata_to_dict(config_inc, cfg)
        overrides = OmegaConf.load(cfg_fn.parent / "overrides.yaml")
        env_overrides = [o for o in overrides if "env" in o]
        env_overrides_str = " ".join(env_overrides)
        config_inc["env_override"] = env_overrides_str
        configs_inc.append(config_inc)

        experiment_configs.append({"experiment_id": experiment_id, "config": cfg})

    trajectory_df = pd.concat(trajectories).reset_index(drop=True)
    configs_inc_df = pd.DataFrame(configs_inc)

    trajectory_df.to_csv(rundir / "trajectory.csv", index=False)
    configs_inc_df.to_csv(rundir / "configs_inc.csv", index=False)


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
    non_config_keys = ["seed", "task_id", "optimizer_id", "objective_function", "config", "env_override"]

    configs_inc_fn = rundir / "configs_inc.csv"
    configs_inc_df = pd.read_csv(configs_inc_fn)
    configs_inc_df["config"] = configs_inc_df["config"].map(literal_eval).map(OmegaConf.create)
    hp_names = [c for c in configs_inc_df.columns if c not in non_config_keys]

    for idx in range(len(configs_inc_df)):
        cfg = configs_inc_df.loc[idx, "config"]
        task = instantiate(cfg.task)
        configuration_dict = dict(configs_inc_df.loc[idx, hp_names])
        policy = task.objective_function.make_policy_from_config_dict(config_dict=configuration_dict, seed=cfg.seed)
        printr(policy)

        policy_cls = policy.__class__
        policy_class_path = f"{policy_cls.__module__}.{policy_cls.__qualname__}"
        policy_kwargs = policy.get_init_kwargs()

        env_override = configs_inc_df.loc[idx, "env_override"]
        print(env_override)

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


def collect(rundir: str = "runs/SMAC-AC-CostInc") -> None:
    """Collect incumbents specifying the DAC policy from rundir.

    Parameters
    ----------
    rundir: str = "runs/SMAC-AC-CostInc"
        The rundir containing the optimization runs or whatever.
    """
    _rundir = Path(rundir)
    gather_data(_rundir)
    policy_path = create_configs(_rundir)
    logger.info(f"Ready to run `bash scripts/evaluate.sh $TASK {policy_path}")


if __name__ == "__main__":
    # python -m dacboenv.experiment.collect_incumbents "runs/SMAC-AC-CostInc"
    Fire(collect)
