"""Logging utilities for the project."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from carps.loggers.file_logger import get_run_directory
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from rich.logging import RichHandler


def setup_logging() -> None:
    """Setup logging module."""
    FORMAT = "%(message)s"
    logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])


def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name.

    Parameters
    ----------
    logger_name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Logger object.
    """
    setup_logging()
    return logging.getLogger(logger_name)


class CustomEncoder(json.JSONEncoder):
    """- Serializes python/Numpy objects via customizing json encoder.
    - **Usage**
        - `json.dumps(python_dict, cls=EncodeFromNumpy)` to get json string.
        - `json.dump(*args, cls=EncodeFromNumpy)` to create a file.json.
    """

    def default(self, obj: Any) -> Any:
        """Converts numpy objects to pure python objects.

        Parameters
        ----------
        obj : Any
            Object to be converted.

        Returns
        -------
        Any
            Pure python object.
        """
        if isinstance(obj, np.int64 | np.int32):
            return int(obj)
        if isinstance(obj, np.float64 | np.float32):
            return float(obj)
        return super().default(obj)


def log_pip_freeze(file_path: str | Path) -> None:
    """Write the output of `pip freeze` directly to a file."""
    logger = get_logger("carps.utils.loggingutils.log_pip_freeze")
    try:
        # TODO: enable discovery and usage of uv
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)  # noqa: S603, S607
        with open(file_path, "a") as f:
            f.write("Installed packages (pip freeze):\n")
            f.write(result.stdout + "\n")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("Failed to run pip freeze. Error: %s", e)
        with open(file_path, "a") as f:
            f.write("Failed to run pip freeze:\n")
            f.write(str(e) + "\n")


def log_python_env(log_file: str | Path = "env_log.txt") -> None:
    """Log the Python environment details directly to a file."""
    with open(log_file, "w") as f:
        f.write("Python Environment Information\n")
        f.write("=" * 32 + "\n")
        f.write(f"Python Version: {sys.version}\n")
        f.write(f"Python Executable: {sys.executable}\n")
    log_pip_freeze(log_file)


def dump_logs(log_data: dict, filename: str) -> None:
    """Dump log dict in jsonl format.

    This appends one json dict line to the filename.

    Parameters
    ----------
    log_data : dict
        Data to log, must be json serializable.
    filename : str
        Filename without path. The path will be either the
        current working directory or if it is called during
        a hydra session, the hydra run dir will be the log
        dir.
    """
    log_data_str = json.dumps(log_data, cls=CustomEncoder) + "\n"

    try:
        # Check if we are in a hydra context
        hydra_cfg = HydraConfig.instance().get()
        if hydra_cfg.mode == RunMode.RUN:
            directory = Path(hydra_cfg.run.dir)
        else:  # MULTIRUN
            directory = Path(hydra_cfg.sweep.dir) / hydra_cfg.sweep.subdir

    except Exception:  # noqa: BLE001
        directory = Path()
    filepath = directory / filename
    with open(filepath, mode="a") as file:
        file.writelines([log_data_str])


def maybe_remove_logs(  # noqa: C901
    directory: str | None = None,
    overwrite: bool = True,  # noqa: FBT001, FBT002
    logfile: str = "results.jsonl",
    logger: logging.Logger | None = None,
) -> None:
    """Maybe remove log files.

    Parameters
    ----------
    directory : str | None, optional
        The log directory, by default None
    overwrite : bool, optional
        Whether to overwrite logs or not, by default True
    logfile : str, optional
        The log filename to look out for, by default "results.jsonl"
    logger : logging.Logger, optional
        Logs info messages.

    Raises
    ------
    RuntimeError
        When logs are found in directory but overwrite is false.
    """
    _directory = Path(directory) if directory is not None else get_run_directory()
    assert _directory is not None, "Directory must be specified in FileLogger or hydra run dir must be available."
    if (_directory / logfile).is_file():
        if overwrite:
            if logger is not None:
                logger.info(f"Found previous run. Removing '{_directory}'.")
            for root, dirs, files in os.walk(_directory, topdown=False):
                for f in files:
                    full_fn = Path(root) / f
                    if ".hydra" not in str(full_fn):
                        full_fn.unlink()
                        if logger:
                            logger.debug(f"Removed file {full_fn}")
                for d in dirs:
                    full_dir = Path(root) / d
                    if ".hydra" not in str(full_dir):
                        shutil.rmtree(full_dir)
                        if logger:
                            logger.debug(f"Removed directory {full_dir}")
        else:
            raise RuntimeError(
                f"Found previous run at '{_directory}'. Stopping run. If you want to overwrite, specify overwrite "
                f"for the file logger in the config (CARP-S/carps/configs/logger.yaml)."
            )
