"""Build carps optimizer."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from carps.utils.env_vars import CARPS_ROOT
from carps.utils.running import make_optimizer, make_task
from ConfigSpace import ConfigurationSpace, Float
from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer

_BBOB_TASK_ID = re.compile(r"^bbob/(?P<dimension>\d+)/(?P<function_id>\d+)/(?P<instance_id>\d+)$")
_BBOB_MAX_FUNCTION_ID = 24


def is_bbob_task_id(task_id: str) -> bool:
    """Return whether ``task_id`` is a supported canonical BBOB task ID."""
    match = _BBOB_TASK_ID.fullmatch(task_id)
    if match is None:
        return False
    dimension = int(match.group("dimension"))
    function_id = int(match.group("function_id"))
    return dimension > 0 and 1 <= function_id <= _BBOB_MAX_FUNCTION_ID


def make_bbob_configuration_space(dimension: int) -> ConfigurationSpace:
    """Create the continuous ``[-5, 5]^d`` BBOB configuration space."""
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError(f"BBOB dimension must be an integer, got {dimension!r}.")
    if dimension <= 0:
        raise ValueError(f"BBOB dimension must be > 0, got {dimension}.")

    configuration_space = ConfigurationSpace()
    configuration_space.add([Float(name=f"x{i}", bounds=(-5.0, 5.0)) for i in range(dimension)])
    return configuration_space


def get_bbob_n_trials(dimension: int) -> int:
    """Return the CARP-S BBOB budget for one dimension."""
    if not isinstance(dimension, int) or isinstance(dimension, bool):
        raise TypeError(f"BBOB dimension must be an integer, got {dimension!r}.")
    if dimension <= 0:
        raise ValueError(f"BBOB dimension must be > 0, got {dimension}.")
    return math.ceil(20 + 40 * math.sqrt(dimension))


def _yaml_scalar(value: str) -> str:
    """Read the simple scalar forms used by CARPS config identifiers."""
    return value.split(" #", maxsplit=1)[0].strip().strip("'\"")


def _config_id_from_text(text: str, id_column: str) -> str | None:
    """Extract a CARPS ID without resolving the complete Hydra config."""
    config_id: str | None = None
    task_name: str | None = None
    in_task_node = False

    for line in text.splitlines():
        if line.startswith(f"{id_column}:"):
            config_id = _yaml_scalar(line.split(":", maxsplit=1)[1])
            continue

        if id_column != "task_id":
            continue
        if line.startswith("task:"):
            in_task_node = True
            continue
        if in_task_node and line and not line.startswith((" ", "#")):
            in_task_node = False
        if in_task_node and line.startswith("  name:"):
            task_name = _yaml_scalar(line.split(":", maxsplit=1)[1])

    if config_id == "${task.name}":
        return task_name
    if config_id is None or config_id.startswith("${"):
        return None
    return config_id


@lru_cache(maxsize=2)
def get_carps_config_index(group_name: str) -> pd.DataFrame:
    """Return a read-only CARPS config index across CARPS releases.

    CARPS 1.1 moved its generated index to a user-cache file, which may not
    exist (and may not be writable on a cluster node). Older releases shipped
    ``configs/<group>/index.csv``. Prefer that file when present; otherwise
    build the same table in memory from packaged YAML files and cache it for
    this process.
    """
    id_column_by_group = {
        "task": "task_id",
        "optimizer": "optimizer_id",
    }
    if group_name not in id_column_by_group:
        raise ValueError(f"group_name must be one of {sorted(id_column_by_group)}, got {group_name!r}.")

    config_root = Path(CARPS_ROOT) / "configs" / group_name
    legacy_index = config_root / "index.csv"
    if legacy_index.is_file():
        return pd.read_csv(legacy_index)

    id_column = id_column_by_group[group_name]
    rows: list[dict[str, str]] = []
    for config_path in sorted(config_root.rglob("*.yaml")):
        config_id = _config_id_from_text(
            config_path.read_text(encoding="utf-8"),
            id_column,
        )
        if config_id is None:
            continue
        rows.append(
            {
                "config_fn": str(config_path),
                id_column: config_id,
            }
        )

    if not rows:
        raise FileNotFoundError(f"Could not index CARPS {group_name} configs below {config_root}.")
    return pd.DataFrame(rows)


def load_optimizer_config(optimizer_id: str) -> DictConfig:
    """Load optimizer config from yaml file.

    The config can also have defaults=["base"], but not any other defaults structure.

    Parameters
    ----------
    optimizer_id : str
        carps optimizer_id or the filename of the optimizer config (yaml).

    Returns
    -------
    DictConfig
        The optimizer config.
    """
    if optimizer_id.endswith(".yaml"):
        config_fn = optimizer_id
    else:
        df = get_carps_config_index("optimizer")
        ids = [optimizer_id]
        config_fn = df.set_index("optimizer_id").loc[ids].reset_index().iloc[0]["config_fn"]
    cfg = OmegaConf.load(config_fn)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping in optimizer config {config_fn}.")
    return maybe_add_defaults(cfg, config_fn)


def maybe_add_defaults(cfg: DictConfig, cfg_fn: str) -> DictConfig:
    """Maybe add default config to config.

    Only works with defaults = ["base"].

    Parameters
    ----------
    cfg : DictConfig
        The config.
    cfg_fn : str
        The source config filename.

    Returns
    -------
    DictConfig
        Cfg, possibly with defaults added.

    Raises
    ------
    ValueError
        When got other defaults than ['base'].
    """
    defaults = cfg.get("defaults", None)
    if defaults is not None:
        if list(cfg.defaults) == ["base"]:
            base_cfg = OmegaConf.load(Path(cfg_fn).parent / "base.yaml")
            merged_cfg = OmegaConf.merge(base_cfg, cfg)
            if not isinstance(merged_cfg, DictConfig):
                raise TypeError(f"Expected mapping configs while merging defaults for {cfg_fn}.")
            cfg = merged_cfg
            del cfg.defaults
        else:
            raise ValueError(f"Can only handle defaults=['base'], but got {cfg.defaults}")
    return cfg


def _build_bbob_task_config(task_id: str) -> DictConfig:
    """Build a CARPS BBOB task for any positive dimension.

    CARPS distributions commonly pre-generate only dimensions 2, 4, 8, 16,
    and 32. DACBO's training curriculum also uses dimensions 3 and 5, so
    constructing the same task schema dynamically avoids depending on a
    generated task index.
    """
    match = _BBOB_TASK_ID.fullmatch(task_id)
    if match is None:
        raise ValueError(f"Invalid BBOB task id: {task_id!r}")

    dimension = int(match.group("dimension"))
    function_id = int(match.group("function_id"))
    instance_id = int(match.group("instance_id"))
    if not 1 <= function_id <= _BBOB_MAX_FUNCTION_ID:
        raise ValueError(f"BBOB function id must be in [1, {_BBOB_MAX_FUNCTION_ID}], got {function_id}.")

    configuration_space = make_bbob_configuration_space(dimension)
    n_trials = get_bbob_n_trials(dimension)

    cfg = OmegaConf.create(
        {
            "benchmark_id": "BBOB",
            "task_id": "${task.name}",
            "task": {
                "_target_": "carps.utils.task.Task",
                "name": task_id,
                "seed": "${seed}",
                "objective_function": {
                    "_target_": "carps.objective_functions.bbob.BBOBObjectiveFunction",
                    "dimension": dimension,
                    "fid": function_id,
                    "instance": instance_id,
                    "seed": "${seed}",
                },
                "input_space": {
                    "_target_": "carps.utils.task.InputSpace",
                    "configuration_space": {
                        "_target_": ("ConfigSpace.configuration_space.ConfigurationSpace.from_serialized_dict"),
                        "_convert_": "object",
                        "d": configuration_space.to_serialized_dict(),
                    },
                    "fidelity_space": {
                        "_target_": "carps.utils.task.FidelitySpace",
                        "is_multifidelity": False,
                        "fidelity_type": None,
                        "min_fidelity": None,
                        "max_fidelity": None,
                    },
                    "instance_space": None,
                },
                "output_space": {
                    "_target_": "carps.utils.task.OutputSpace",
                    "n_objectives": 1,
                    "objectives": ["quality"],
                },
                "optimization_resources": {
                    "_target_": "carps.utils.task.OptimizationResources",
                    "n_trials": n_trials,
                    "time_budget": None,
                    "n_workers": 1,
                },
                "metadata": {
                    "_target_": "carps.utils.task.TaskMetadata",
                    "has_constraints": False,
                    "domain": "synthetic",
                    "objective_function_approximation": "real",
                    "has_virtual_time": False,
                    "deterministic": True,
                    "dimensions": dimension,
                    "search_space_n_categoricals": 0,
                    "search_space_n_ordinals": 0,
                    "search_space_n_integers": 0,
                    "search_space_n_floats": dimension,
                    "search_space_has_conditionals": False,
                    "search_space_has_forbiddens": False,
                    "search_space_has_priors": False,
                },
            },
        }
    )
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected the generated BBOB task to be a mapping.")
    return cfg


def get_task_config(task_id: str) -> DictConfig:
    """Get config filename for task id.

    Parameters
    ----------
    task_id : str
        The task id.

    Returns
    -------
    DictConfig
        The config with the node task.
    """
    if _BBOB_TASK_ID.fullmatch(task_id):
        return _build_bbob_task_config(task_id)

    df = get_carps_config_index("task")
    ids = [task_id]
    # TODO raise proper error if task_id not in index. Can happen when task comes from external module.
    # Find smart registering method.
    config_fn = df.set_index("task_id").loc[ids].reset_index().iloc[0]["config_fn"]
    cfg = OmegaConf.load(config_fn)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping in task config {config_fn}.")
    return maybe_add_defaults(cfg, config_fn)


def build_carps_optimizer(
    task_id: str, seed: int, optimizer_id: str | None = None, optimizer_cfg: DictConfig | None = None
) -> Optimizer:
    """Build carps optimizer.

    Later, the built SMAC solver can be used.
    Either specify `optimizer_id` or `optimizer_cfg`.

    Parameters
    ----------
    task_id : str
        The carps task id.
    seed : int
        The seed.
    optimizer_id : str, optional
        The carps optimizer id.
    optimizer_cfg : DictConfig, optional
        The optimizer config.

    Returns
    -------
    Optimizer
        carps optimizer.
    """
    if optimizer_id is None and optimizer_cfg is None:
        raise ValueError("Specify either optimizer_id or optimizer_cfg!")

    cfg_opt = optimizer_cfg or None
    if cfg_opt is None:
        cfg_opt = load_optimizer_config(optimizer_id=optimizer_id)  # type: ignore[arg-type]

    cfg = get_task_config(task_id=task_id)
    cfg.seed = seed

    if hasattr(cfg_opt, "optimizer"):
        cfg = OmegaConf.merge(cfg, cfg_opt)
    else:
        cfg.optimizer = cfg_opt

    if not hasattr(cfg.optimizer, "_target_"):
        cfg.optimizer._target_ = "carps.optimizers.smac20.SMAC3Optimizer"
        cfg.optimizer._partial_ = True

    if hasattr(cfg, "loggers"):
        del cfg.loggers
    task = make_task(cfg=cfg)

    if OmegaConf.select(cfg, "optimizer.smac_cfg.scenario.n_trials") is not None:
        cfg.optimizer.smac_cfg.scenario.n_trials = cfg.task.optimization_resources.n_trials

    optimizer = make_optimizer(cfg=cfg, task=task)
    optimizer.setup_optimizer()
    return optimizer


if __name__ == "__main__":
    # Use a relative path to the optimizer config for portability
    optimizer_config_path = (
        Path(__file__).parent.parent.parent
        / "lib"
        / "CARP-S"
        / "carps"
        / "configs"
        / "optimizer"
        / "smac20"
        / "blackbox.yaml"
    )
    build_carps_optimizer("bbob/2/1/0", 2, str(optimizer_config_path))
