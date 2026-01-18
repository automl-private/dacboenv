"""No operation / default policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from dacboenv.env.observations.types import ObsType


class NoOpPolicy(AbstractPolicy):
    """Default policy that does nothing."""

    def __call__(self, obs: ObsType | None = None) -> None:  # noqa: ARG002
        """Returns None.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            None.
        """
        return
