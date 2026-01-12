"""Policy utilities for DACBOEnv."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dacboenv.dacboenv import ObsType
from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, ObsType


class RandomPolicy(AbstractPolicy):
    """Policy that samples actions uniformly at random."""

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Select an action by sampling uniformly from the action space.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            A randomly sampled action.
        """
        return self._env.action_space.sample()

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the action space.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        self._env.action_space.seed(seed=seed)
