"""Observation utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    TypeVar,
)

import numpy as np
from gymnasium.spaces import Box, Dict, Space

from dacboenv.signal.modelfit import calculate_model_fit
from dacboenv.signal.ubr import calculate_ubr

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

ObsType = dict[str, Any]
ActType = TypeVar("ActType")


@dataclass
class ObservationType:
    """Represents a single observation type.

    Attributes:
    ----------
    name : str
        Name of the observation.
    space : Space
        Gymnasium space for the observation's value range and type.
    compute : Callable[[SMBO], Any]
        Function to compute the observation value from a SMAC instance.
    """

    name: str
    space: Space
    compute: Callable[[SMBO], Any]


incumbent_change_observation = ObservationType(
    "incumbent_changes", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: smbo.intensifier.incumbents_changed
)
trials_passed_observation = ObservationType(
    "trials_passed", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: len(smbo.runhistory)
)
trials_left_observation = ObservationType(
    "trials_left", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: smbo.remaining_trials
)
ubr_observation = ObservationType(
    "ubr",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo: calculate_ubr(trial_infos=None, trial_values=None, configspace=None, seed=None, smbo=smbo)["ubr"],
)
modelfit_observation = ObservationType(
    "modelfit_mse",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan if np.isnan(scores := calculate_model_fit(smbo)["mean_scores"]).any() else scores[0],
)

ALL_OBSERVATIONS = [
    incumbent_change_observation,
    trials_passed_observation,
    trials_left_observation,
    ubr_observation,
    modelfit_observation,
]


class ObservationSpace:
    """Manages a collection of observation types and their Gymnasium spaces.

    Allows selection of a subset of available observation types and provides methods to
    compute observations from a SMAC instance.

    Parameters
    ----------
    keys : list[str], optional
        List of observation names to include. If None, all available observations are used.

    Attributes:
    ----------
    observation_space : gymnasium.spaces.Dict
        The Gymnasium Dict space describing the selected observations.

    Methods:
    -------
    get_observation(optimizer: SMBO) -> ObsType
        Computes the current observation values from the given optimizer.
    """

    _OBSERVATION_MAP: ClassVar[dict[str, ObservationType]] = {obs.name: obs for obs in ALL_OBSERVATIONS}

    def __init__(self, keys: list[str] | None = None) -> None:
        """Initialize the ObservationSpace.

        Parameters
        ----------
        keys : list[str], optional
            List of observation names to include. If None, all available observations are used.

        Raises:
        ------
        ValueError
            If any provided key is invalid.
        """
        # Default to all possible keys if not provided
        self._keys = keys if keys is not None else list(ObservationSpace._OBSERVATION_MAP.keys())

        # Check for invalid keys
        invalid_keys = set(self._keys) - set(ObservationSpace._OBSERVATION_MAP.keys())
        if invalid_keys:
            raise ValueError(f"Invalid observation keys: {invalid_keys}")

        self._observation_types = [ObservationSpace._OBSERVATION_MAP[key] for key in self._keys]
        self._observation_space = Dict(
            {name: obs.space for name, obs in zip(self._keys, self._observation_types, strict=False)}
        )

    @property
    def observation_space(self) -> Space:
        """Returns the Gymnasium Dict space for the selected observations.

        Returns:
        -------
        gymnasium.spaces.Dict
            The observation space.
        """
        return self._observation_space

    def get_observation(self, optimizer: SMBO) -> ObsType:
        """Compute the current observation values from the given optimizer.

        Parameters
        ----------
        optimizer : SMBO
            The SMAC instance.

        Returns:
        -------
        ObsType
            Dictionary mapping observation names to their computed values.
        """
        return {obs.name: obs.compute(optimizer) for obs in self._observation_types}
