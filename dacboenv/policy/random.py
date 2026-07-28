"""Random policy."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class RandomPolicy(AbstractPolicy):
    """Policy that samples actions uniformly at random."""

    def __init__(self, env: DACBOEnv) -> None:
        """Initialize a policy-local copy of the environment action space."""
        super().__init__(env)
        self._policy_action_space = deepcopy(env.action_space)

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
        return self._policy_action_space.sample()

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the action space.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        self._policy_action_space.seed(seed=seed)
