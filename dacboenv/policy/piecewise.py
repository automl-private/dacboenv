"""Piecewise policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class PiecewiseParameterPolicy(AbstractPolicy):
    """Policy that sets the parameter based on a fixed piecewise-linear function."""

    def __init__(self, env: DACBOEnv, splits: np.ndarray) -> None:
        """Initialize the jump parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        splits : np.ndarray
            y values of the splits between the linear sections.
        """
        super().__init__(env, splits=splits)
        self._splitsy = splits

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the parameter value based on progress and piecewise function.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            Parameter value based on piecewise-linear interpolation.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)
        splitsx = np.linspace(0, 1, len(self._splitsy), dtype=float) * budget
        val = np.interp(trials, splitsx, self._splitsy)

        return val**2  # XXX: Circumvent squareroot
