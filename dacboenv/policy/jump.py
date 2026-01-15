"""Jump policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dacboenv.env.reward import get_initial_design_size
from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv, ObsType


class JumpParameterPolicy(AbstractPolicy):
    """Policy that switches from a low to a high parameter value
    after a fraction of the optimization budget.
    """

    def __init__(self, env: DACBOEnv, low: float, high: float, jump: float, seed: int | None = None) -> None:
        """Initialize the jump parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        low : float
            Parameter value before the jump.
        high : float
            Parameter value after the jump.
        jump : float
            Fraction of the optimization budget at which to switch
            from ``low`` to ``high``.
        """
        super().__init__(env, low=low, high=high, jump=jump, seed=seed)
        self._low = low
        self._high = high
        self._jump = jump

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the parameter value based on progress and jump threshold.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            ``low`` before the jump threshold, otherwise ``high``.
        """
        smac = self._env._smac_instance
        n_finished = smac.runhistory.finished
        n_initial_design = get_initial_design_size(smac)
        n_smbo = smac._scenario.n_trials
        n_model_based = n_smbo - n_initial_design

        if (n_finished - n_initial_design) < self._jump * n_model_based:
            return self._low
        return self._high


class JumpFunctionPolicy(AbstractPolicy):
    """Policy that switches the optimizer's acquisition function
    after a fraction of the optimization budget.
    """

    def __init__(self, env: DACBOEnv, low: int, high: int, jump: float) -> None:
        """Initialize the jump function policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        low : int
            Acquisition function index before the jump.
        high : int
            Acquisition function index after the jump.
        jump : float
            Fraction of the optimization budget at which to switch
            from ``low`` to ``high``.
        """
        super().__init__(env, low=low, high=high, jump=jump)
        self._low = low
        self._high = high
        self._jump = jump

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the acquisition function index based on progress and jump threshold.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            ``low`` before the jump threshold, otherwise ``high``.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)

        if trials < self._jump * budget:
            return self._low
        return self._high
