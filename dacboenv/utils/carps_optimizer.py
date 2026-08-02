"""Build carps optimizer."""

from __future__ import annotations

import copy
import math
import re
from functools import lru_cache, partial
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from carps.utils.env_vars import CARPS_ROOT
from carps.utils.running import make_optimizer, make_task
from omegaconf import DictConfig, OmegaConf
from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.random_design.abstract_random_design import AbstractRandomDesign

from dacboenv.utils.seeding import episode_component_seeds

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer

_BBOB_TASK_ID = re.compile(r"^bbob/(?P<dimension>\d+)/(?P<function_id>\d+)/(?P<instance_id>\d+)$")
_BBOB_CONFIG_FILENAME = re.compile(r"^cfg_(?P<dimension>\d+)_(?P<function_id>\d+)_(?P<instance_id>\d+)\.yaml$")
_BBOB_MAX_FUNCTION_ID = 24
EXPECTED_NATIVE_BBOB_DIMENSIONS = (2, 4, 8, 16, 32)
_PACKAGE_RELATIVE_SMAC_LOGGING_PATH = Path("dacboenv/configs/logging/smac_internal.yaml")


def _resolve_package_relative_smac_logging_path(cfg: DictConfig) -> None:
    """Make the saved repository-relative SMAC logging path runtime-portable."""
    config_path = "optimizer.smac_cfg.smac_kwargs.logging_level"
    logging_level = OmegaConf.select(cfg, config_path)
    package_logging_path = Path(str(files("dacboenv.configs").joinpath("logging/smac_internal.yaml")))

    if isinstance(logging_level, Path):
        if logging_level == _PACKAGE_RELATIVE_SMAC_LOGGING_PATH:
            OmegaConf.update(cfg, config_path, package_logging_path, force_add=False)
        return

    if not isinstance(logging_level, DictConfig):
        return
    arguments = OmegaConf.select(logging_level, "_args_")
    if arguments and Path(str(arguments[0])) == _PACKAGE_RELATIVE_SMAC_LOGGING_PATH:
        arguments[0] = str(package_logging_path)


@lru_cache(maxsize=1)
def get_native_bbob_task_configs() -> dict[tuple[int, int, int], Path]:
    """Index native CARP-S BBOB task files and enforce the pinned dimensions."""
    config_root = Path(CARPS_ROOT) / "configs" / "task" / "BBOB"
    configs: dict[tuple[int, int, int], Path] = {}
    for config_path in sorted(config_root.glob("cfg_*.yaml")):
        match = _BBOB_CONFIG_FILENAME.fullmatch(config_path.name)
        if match is None:
            continue
        key = tuple(int(match.group(name)) for name in ("dimension", "function_id", "instance_id"))
        if key in configs:
            raise RuntimeError(f"Duplicate native CARP-S BBOB config for {key}: {configs[key]} and {config_path}.")
        configs[key] = config_path

    discovered_dimensions = tuple(sorted({dimension for dimension, _, _ in configs}))
    if discovered_dimensions != EXPECTED_NATIVE_BBOB_DIMENSIONS:
        raise RuntimeError(
            "Checked-out CARP-S BBOB dimensions differ from the scientifically audited set: "
            f"expected {list(EXPECTED_NATIVE_BBOB_DIMENSIONS)}, found {list(discovered_dimensions)} "
            f"below {config_root}."
        )
    return configs


def discover_native_bbob_dimensions() -> tuple[int, ...]:
    """Return the dimensions discovered from native CARP-S BBOB filenames."""
    return tuple(sorted({dimension for dimension, _, _ in get_native_bbob_task_configs()}))


def is_bbob_task_id(task_id: str) -> bool:
    """Return whether ``task_id`` maps to an existing native CARP-S BBOB task."""
    match = _BBOB_TASK_ID.fullmatch(task_id)
    if match is None:
        return False
    key = tuple(int(match.group(name)) for name in ("dimension", "function_id", "instance_id"))
    return key in get_native_bbob_task_configs()


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
    bbob_match = _BBOB_TASK_ID.fullmatch(task_id)
    if bbob_match is not None:
        key = tuple(int(bbob_match.group(name)) for name in ("dimension", "function_id", "instance_id"))
        config_path = get_native_bbob_task_configs().get(key)
        if config_path is None:
            raise FileNotFoundError(
                f"BBOB task {task_id!r} has no native CARP-S config; only dimensions "
                f"{list(discover_native_bbob_dimensions())} are allowed and every requested tuple must exist."
            )
        cfg = OmegaConf.load(config_path)
        if not isinstance(cfg, DictConfig):
            raise TypeError(f"Expected a mapping in task config {config_path}.")
        return maybe_add_defaults(cfg, str(config_path))

    df = get_carps_config_index("task")
    ids = [task_id]
    # TODO raise proper error if task_id not in index. Can happen when task comes from external module.
    # Find smart registering method.
    config_fn = df.set_index("task_id").loc[ids].reset_index().iloc[0]["config_fn"]
    cfg = OmegaConf.load(config_fn)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping in task config {config_fn}.")
    return maybe_add_defaults(cfg, config_fn)


def _prepare_episode_smac_config(cfg: DictConfig, seed: int) -> dict[str, int]:  # noqa: C901, PLR0915
    """Make the selected inner seed and stochastic components episode-local."""
    component_seeds = episode_component_seeds(seed)
    if OmegaConf.select(cfg, "optimizer.smac_cfg.scenario") is not None:
        cfg.optimizer.smac_cfg.scenario.seed = int(seed)

    initial_design_path = "optimizer.smac_cfg.smac_kwargs.initial_design"
    initial_design = OmegaConf.select(cfg, initial_design_path)
    uses_blackbox_defaults = str(OmegaConf.select(cfg, "optimizer.smac_cfg.smac_class", default="")).endswith(
        ".BlackBoxFacade"
    )
    if initial_design is None and uses_blackbox_defaults:
        OmegaConf.update(
            cfg,
            initial_design_path,
            {
                "_target_": "smac.initial_design.sobol_design.SobolInitialDesign",
                "_partial_": True,
                "n_configs": None,
                "n_configs_per_hyperparameter": 8,
                "max_ratio": 0.25,
                "seed": component_seeds["initial_design"],
            },
            force_add=True,
        )
        initial_design = OmegaConf.select(cfg, initial_design_path)
    if isinstance(initial_design, DictConfig):
        target = str(initial_design.get("_target_", ""))
        if target.endswith(".get_initial_design"):
            initial_design_cfg = OmegaConf.to_container(initial_design, resolve=False)
            assert isinstance(initial_design_cfg, dict)
            initial_design_cfg["_target_"] = "smac.initial_design.sobol_design.SobolInitialDesign"
            initial_design_cfg["_partial_"] = True
            initial_design_cfg.setdefault("n_configs_per_hyperparameter", 8)
            initial_design_cfg["seed"] = component_seeds["initial_design"]
            OmegaConf.update(cfg, initial_design_path, initial_design_cfg, force_add=False)
    elif isinstance(initial_design, partial) and getattr(initial_design.func, "__name__", "") == "get_initial_design":
        initial_design_kwargs = dict(initial_design.keywords or {})
        initial_design_kwargs.setdefault("n_configs_per_hyperparameter", 8)
        initial_design_kwargs["seed"] = component_seeds["initial_design"]
        cfg.optimizer.smac_cfg.smac_kwargs.initial_design = partial(SobolInitialDesign, **initial_design_kwargs)

    acquisition_function = OmegaConf.select(cfg, "optimizer.smac_cfg.smac_kwargs.acquisition_function")
    if acquisition_function is None and uses_blackbox_defaults:
        OmegaConf.update(
            cfg,
            "optimizer.smac_cfg.smac_kwargs.acquisition_function",
            {
                "_target_": "smac.acquisition.function.expected_improvement.EI",
                "xi": 0.0,
            },
            force_add=True,
        )
        acquisition_function = OmegaConf.select(cfg, "optimizer.smac_cfg.smac_kwargs.acquisition_function")
    if isinstance(acquisition_function, AbstractAcquisitionFunction):
        # OmegaConf preserves allow_objects values by identity even when its
        # containing DictConfig is deep-copied.
        cfg.optimizer.smac_cfg.smac_kwargs.acquisition_function = copy.deepcopy(acquisition_function)

    random_design_path = "optimizer.smac_cfg.smac_kwargs.random_design"
    random_design = OmegaConf.select(cfg, random_design_path)
    if random_design is None and uses_blackbox_defaults:
        OmegaConf.update(
            cfg,
            random_design_path,
            {
                "_target_": "smac.random_design.probability_design.ProbabilityRandomDesign",
                "probability": 0.08447232371720552,
                "seed": component_seeds["random_design"],
            },
            force_add=True,
        )
        random_design = OmegaConf.select(cfg, random_design_path)
    if isinstance(random_design, AbstractRandomDesign):
        # Hydra may already have materialized this nested object with the outer
        # worker seed. SMAC exposes no public reseed method, so copy it before
        # resetting its two seed-owned fields for this episode.
        random_design = copy.deepcopy(random_design)
        random_design_seed = component_seeds["random_design"]
        random_design._seed = random_design_seed
        random_design._rng = np.random.RandomState(seed=random_design_seed)
        cfg.optimizer.smac_cfg.smac_kwargs.random_design = random_design
    elif isinstance(random_design, DictConfig):
        target = str(random_design.get("_target_", ""))
        if target.endswith(".get_random_design"):
            random_design_cfg = OmegaConf.to_container(random_design, resolve=False)
            assert isinstance(random_design_cfg, dict)
            random_design_cfg["_target_"] = "smac.random_design.probability_design.ProbabilityRandomDesign"
            random_design_cfg.pop("scenario", None)
            random_design_cfg.pop("_partial_", None)
            random_design_cfg["seed"] = component_seeds["random_design"]
            OmegaConf.update(cfg, random_design_path, random_design_cfg, force_add=False)

    acquisition_maximizer_path = "optimizer.smac_cfg.smac_kwargs.acquisition_maximizer"
    acquisition_maximizer = OmegaConf.select(cfg, acquisition_maximizer_path)
    if acquisition_maximizer is None and acquisition_function is not None:
        OmegaConf.update(
            cfg,
            acquisition_maximizer_path,
            {
                "_target_": "smac.acquisition.maximizer.local_and_random_search.LocalAndSortedRandomSearch",
                "_partial_": True,
                "challengers": 1000,
                "local_search_iterations": 10,
                "seed": component_seeds["acquisition_maximizer"],
            },
            force_add=True,
        )

    return component_seeds


def build_carps_optimizer(
    task_id: str,
    seed: int,
    optimizer_id: str | None = None,
    optimizer_cfg: DictConfig | None = None,
    output_directory: str | Path | None = None,
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

    # Copy the config container for this episode. Hydra-materialized mutable
    # values preserve identity through OmegaConf deepcopy and are cloned
    # explicitly in _prepare_episode_smac_config below.
    cfg_opt = copy.deepcopy(optimizer_cfg) if optimizer_cfg is not None else None
    if cfg_opt is None:
        cfg_opt = load_optimizer_config(optimizer_id=optimizer_id)  # type: ignore[arg-type]

    cfg = get_task_config(task_id=task_id)
    cfg.seed = seed

    if hasattr(cfg_opt, "optimizer"):
        cfg = OmegaConf.merge(cfg, cfg_opt)
    else:
        cfg.optimizer = cfg_opt

    _resolve_package_relative_smac_logging_path(cfg)

    if output_directory is not None:
        cfg.optimizer.smac_cfg.scenario.output_directory = str(output_directory)

    # Reassert the selected inner BO seed after merging because Hydra may have
    # materialized nested components with the outer worker seed already.
    component_seeds = _prepare_episode_smac_config(cfg, seed)

    if not hasattr(cfg.optimizer, "_target_"):
        cfg.optimizer._target_ = "carps.optimizers.smac20.SMAC3Optimizer"
        cfg.optimizer._partial_ = True

    if hasattr(cfg, "loggers"):
        del cfg.loggers
    task = make_task(cfg=cfg)
    task.input_space.configuration_space.seed(component_seeds["configspace"])

    if OmegaConf.select(cfg, "optimizer.smac_cfg.scenario.n_trials") is not None:
        cfg.optimizer.smac_cfg.scenario.n_trials = cfg.task.optimization_resources.n_trials

    optimizer = make_optimizer(cfg=cfg, task=task)
    optimizer.setup_optimizer()
    optimizer.seed_stream_metadata = {
        "selected_inner_seed": int(seed),
        **component_seeds,
    }
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
