"""Random policy."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
from gymnasium.spaces import Discrete

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


class MarginalRandomPolicy(AbstractPolicy):
    """Sample independently from a learned policy's marginal action rates."""

    def __init__(self, env: DACBOEnv, probabilities: list[float]) -> None:
        super().__init__(env, probabilities=probabilities)
        if not isinstance(env.action_space, Discrete):
            raise TypeError("MarginalRandomPolicy requires a discrete action space.")
        values = np.asarray(probabilities, dtype=float)
        if values.shape != (env.action_space.n,):
            raise ValueError(f"Expected {env.action_space.n} marginal probabilities, got shape {values.shape}.")
        if not np.isfinite(values).all() or np.any(values < 0.0) or not np.isclose(values.sum(), 1.0):
            raise ValueError("Marginal probabilities must be finite, non-negative, and sum to one.")
        self._probabilities = values
        self._rng = np.random.default_rng()

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Draw one state-independent action with the matched marginal."""
        return int(self._rng.choice(len(self._probabilities), p=self._probabilities))

    def set_seed(self, seed: int | None) -> None:
        """Reset the policy-local sampling stream."""
        self._rng = np.random.default_rng(seed)
