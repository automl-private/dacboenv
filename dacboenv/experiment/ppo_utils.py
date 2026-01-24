"""Utils for PPO."""

from __future__ import annotations

import csv
from io import TextIOWrapper
from pathlib import Path

import numpy as np
from carps.loggers.file_logger import get_run_directory
from stable_baselines3.common.callbacks import BaseCallback


class ActionLoggingCallback(BaseCallback):
    """Callback to log actions.

    For each new episode, log the actions. Will be overwritten.
    Intended for quick inspection.
    """

    def __init__(self, n_envs: int, csv_path: str | None = None, verbose: int = 0) -> None:
        """Init.

        Parameters
        ----------
        n_envs : int
            Number of environments.
        csv_path : str | None, optional
            The target path for the actions file, by default None. Defaults to
            the current run directory / "tensorboard/actions.csv".
        verbose : int, optional
            Verbosity level of the callback, by default 0
        """
        super().__init__(verbose)
        if csv_path is None:
            csv_path = str(get_run_directory() / "tensorboard/actions.csv")
        self.csv_path = csv_path
        self.file: TextIOWrapper | None = None
        self.writer = None
        self.step = 0
        self._n_envs = n_envs

    def _reset_csv(self) -> None:
        # Close old file
        if self.file is not None:
            self.file.close()

        # Delete previous CSV
        if Path(self.csv_path).exists():
            Path(self.csv_path).unlink()

        # Create new CSV
        self.file = open(self.csv_path, "w", newline="")  # noqa: SIM115
        self.writer = csv.writer(self.file)  # type: ignore[assignment]
        assert self.writer is not None

        # Header
        header = (
            ["step"]
            + [f"env_{i}/action" for i in range(self._n_envs)]
            + [f"env_{i}/instance" for i in range(self._n_envs)]
        )
        self.writer.writerow(header)

        self.step = 0

    def _on_training_start(self) -> None:
        self._reset_csv()

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        dones = self.locals["dones"]

        # Flatten actions for CSV (handles discrete & continuous)
        row = [self.step]
        for action in actions:
            value = float(np.mean(action)) if np.ndim(action) > 0 else float(action)
            row.append(value)  # type: ignore[arg-type]

        env = self.locals["env"]  # VecEnv
        instances = env.get_attr("instance")
        row.extend(instances)

        self.writer.writerow(row)  # type: ignore[assignment,attr-defined]
        self.file.flush()  # type: ignore[assignment,union-attr]

        self.step += 1

        # If any env finishes → new episode → overwrite CSV
        if np.any(dones):
            self._reset_csv()

        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
