"""Build carps optimizer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from carps.utils.running import make_optimizer, make_task
from hydra import compose, initialize_config_module
from omegaconf import OmegaConf
from rich import inspect

from dacboenv.utils.reference_performance import get_optimizer_overrides, get_seed_override, get_task_overrides

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer
    from omegaconf import DictConfig


def load_optimizer_config(optimizer_cfg_fn: str) -> DictConfig:
    """Load optimizer config from yaml file.

    The config can also have defaults=["base"], but not any other defaults structure.

    Parameters
    ----------
    optimizer_cfg_fn : str
        The filename of the optimizer config (yaml).

    Raise
    -----
    ValueError:
        If defaults present, but anything other than ['base'].

    Returns
    -------
    DictConfig
        The optimizer config.
    """
    cfg = OmegaConf.load(optimizer_cfg_fn)
    defaults = cfg.get("defaults", None)
    if defaults is not None:
        if list(cfg.defaults) == ["base"]:
            cfg = OmegaConf.merge(cfg, OmegaConf.load(Path(optimizer_cfg_fn).parent / "base.yaml"))
            del cfg.defaults
        else:
            raise ValueError(f"Can only handle defaults=['base'], but got {cfg.defaults}")
    return cfg


def build_carps_optimizer(optimizer_id: str, task_id: str, seed: int) -> Optimizer:
    """Build carps optimizer.

    Later, the built SMAC solver can be used.

    Parameters
    ----------
    optimizer_id : str
        The carps optimizer id.
    task_id : str
        The carps task id.
    seed : int
        The seed.

    Returns
    -------
    Optimizer
        carps optimizer.
    """
    cfg_opt = None
    if optimizer_id.endswith(".yaml"):
        optimizer_override = ""
        cfg_opt = load_optimizer_config(optimizer_id)
    else:
        optimizer_override = get_optimizer_overrides([optimizer_id])[0]
    task_override = get_task_overrides([task_id])[0]
    seed_override = get_seed_override([seed])

    carps_overrides = ["hydra.searchpath=['pkg://dacboenv/configs']", task_override, seed_override]
    if len(optimizer_override) > 0:
        carps_overrides.append(optimizer_override)

    with initialize_config_module(config_module="carps.configs", job_name="run_from_script", version_base="1.1"):
        cfg: DictConfig = compose(config_name="base.yaml", overrides=carps_overrides)

    if cfg_opt is not None:
        cfg = OmegaConf.merge(cfg, cfg_opt)

    del cfg.loggers
    task = make_task(cfg=cfg)
    inspect(task)

    optimizer = make_optimizer(cfg=cfg, task=task)
    inspect(optimizer)
    optimizer.setup_optimizer()
    return optimizer


if __name__ == "__main__":
    build_carps_optimizer(
        "/home/numina/Documents/repos/dacboenv/lib/CARP-S/carps/configs/optimizer/smac20/blackbox.yaml", "bbob/2/1/0", 2
    )
