"""Static policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium.spaces import Discrete

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class StaticParameterPolicy(AbstractPolicy):
    """Policy that always returns a fixed parameter value."""

    def __init__(self, env: DACBOEnv, par_val: int | float) -> None:
        """Initialize the static parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        par_val : int | float
            Fixed action to return for every decision. For discrete action
            spaces this is the action index; for legacy continuous spaces it
            is the parameter value.
        """
        super().__init__(env, par_val=par_val)
        if isinstance(env.action_space, Discrete) and not env.action_space.contains(par_val):
            raise ValueError(f"Static action {par_val!r} is not in {env.action_space}.")
        self._par_val = par_val

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the fixed parameter value.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            The fixed parameter value.
        """
        return self._par_val
