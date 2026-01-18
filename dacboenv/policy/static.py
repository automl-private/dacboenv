"""Static policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class StaticParameterPolicy(AbstractPolicy):
    """Policy that always returns a fixed parameter value."""

    def __init__(self, env: DACBOEnv, par_val: float) -> None:
        """Initialize the static parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        par_val : float
            Fixed parameter value to return for every action.
        """
        super().__init__(env, par_val=par_val)
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
