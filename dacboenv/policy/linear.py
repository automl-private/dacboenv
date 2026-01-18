"""Linear increase/decrease policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class LinearParameterPolicy(AbstractPolicy):
    """Policy that interpolates linearly between two parameter values
    across the optimization budget.
    """

    def __init__(self, env: DACBOEnv, high_to_low: bool, low: float, high: float) -> None:  # noqa: FBT001
        """Initialize the linear parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        high_to_low : bool
            Whether to start from the high value and decrease to the low value.
        low : float
            Lower bound of the parameter value.
        high : float
            Upper bound of the parameter value.
        """
        super().__init__(env, high_to_low=high_to_low, low=low, high=high)
        self._high_to_low = high_to_low
        self._low = low
        self._high = high

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return a linearly interpolated parameter value based on the optimization progress.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            Interpolated parameter value depending on the number of completed trials.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)

        weight = trials / budget

        if self._high_to_low:
            return (1 - weight) * self._high + weight * self._low
        return weight * self._high + (1 - weight) * self._low
