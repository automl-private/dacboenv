"""Utils for PPO."""

from __future__ import annotations

import csv
import json
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
        self._episode_ids = [0] * n_envs
        self._instances: list[object] = []

    def _open_csv(self) -> None:
        """Open a fresh append-only action log for this training run."""
        if self.file is not None:
            self.file.close()

        Path(self.csv_path).parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.csv_path, "w", newline="")  # noqa: SIM115
        self.writer = csv.writer(self.file)  # type: ignore[assignment]
        assert self.writer is not None

        header = (
            ["step"]
            + [f"env_{i}/action" for i in range(self._n_envs)]
            + [f"env_{i}/instance" for i in range(self._n_envs)]
            + [f"env_{i}/bo_evaluations" for i in range(self._n_envs)]
            + [f"env_{i}/done" for i in range(self._n_envs)]
            + [f"env_{i}/episode" for i in range(self._n_envs)]
        )
        self.writer.writerow(header)

        self.step = 0
        self._episode_ids = [0] * self._n_envs

    def _on_training_start(self) -> None:
        self._open_csv()
        self._instances = list(self.training_env.get_attr("instance"))

    def _on_step(self) -> bool:
        actions = self.locals["actions"]
        dones = self.locals["dones"]

        # Preserve every action component
        row = [self.step]
        for action in actions:
            action_array = np.asarray(action)
            value = action_array.item() if action_array.size == 1 else json.dumps(action_array.tolist())
            row.append(value)

        # VecEnv auto-resets completed environments before this callback. Keep
        # the cached pre-step instance for the terminal action, then refresh it.
        row.extend(self._instances)
        row.extend(info.get("bo_evaluations", "") for info in self.locals["infos"])
        row.extend(bool(done) for done in dones)
        row.extend(self._episode_ids)

        self.writer.writerow(row)  # type: ignore[assignment,attr-defined]
        self.file.flush()  # type: ignore[assignment,union-attr]

        self.step += 1
        if np.any(dones):
            next_instances = self.locals["env"].get_attr("instance")
            for env_id, done in enumerate(dones):
                if done:
                    self._episode_ids[env_id] += 1
                    self._instances[env_id] = next_instances[env_id]

        return True

    def _on_training_end(self) -> None:
        if self.file is not None:
            self.file.close()
