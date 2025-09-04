"""Observation utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np
from gymnasium.spaces import Box, Dict, Space

from dacboenv.signal.modelfit import calculate_model_fit
from dacboenv.signal.ubr import calculate_ubr

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

    from dacboenv.dacboenv import ObsType

from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from scipy.stats import kurtosis, skew

from dacboenv.utils.X_features import exploration_tsp, knn_entropy
from dacboenv.utils.y_features import calc_variability


@dataclass
class ObservationType:
    """Represents a single observation type.

    Attributes
    ----------
    name : str
        Name of the observation.
    space : Space
        Gymnasium space for the observation's value range and type.
    compute : Callable[[SMBO], Any]
        Function to compute the observation value from a SMAC instance.
    default : int | float
        The observation's default value.
    """

    name: str
    space: Space
    compute: Callable[[SMBO], Any]
    default: int | float


incumbent_change_observation = ObservationType(
    "incumbent_changes", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: smbo.intensifier.incumbents_changed, 0
)
trials_passed_observation = ObservationType(
    "trials_passed", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: len(smbo.runhistory), 0
)
trials_left_observation = ObservationType(
    "trials_left", Box(low=0, high=np.inf, dtype=np.float32), lambda smbo: smbo.remaining_trials, -1
)
ubr_observation = ObservationType(
    "ubr",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo: calculate_ubr(trial_infos=None, trial_values=None, configspace=None, seed=None, smbo=smbo)["ubr"],
    np.nan,
)
modelfit_observation = ObservationType(
    "modelfit_mse",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan if np.isnan(scores := calculate_model_fit(smbo)["mean_scores"]).any() else scores[0],
    np.nan,
)
dimensions_observation = ObservationType(
    "searchspace_dim",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len(smbo._scenario.configspace),
    0,
)
continuous_hp_observation = ObservationType(
    "continuous_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for _, hp in smbo._scenario.configspace.items() if isinstance(hp, FloatHyperparameter)]),
    0,
)
categorical_hp_observation = ObservationType(
    "categorical_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for _, hp in smbo._scenario.configspace.items() if isinstance(hp, CategoricalHyperparameter)]),
    0,
)
ordinal_hp_observation = ObservationType(
    "ordinal_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for _, hp in smbo._scenario.configspace.items() if isinstance(hp, OrdinalHyperparameter)]),
    0,
)
int_hp_observation = ObservationType(
    "int_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for _, hp in smbo._scenario.configspace.items() if isinstance(hp, IntegerHyperparameter)]),
    0,
)
tsp_observation = ObservationType(
    "tsp",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: exploration_tsp(smbo.intensifier.config_selector._collect_data()[0])[-1],
    np.nan,
)
knn_entropy_observation = ObservationType(
    "knn_entropy",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: knn_entropy(smbo.intensifier.config_selector._collect_data()[0]),
    np.nan,
)
skewness_observation = ObservationType(
    "y_skewness",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: skew(smbo.intensifier.config_selector._collect_data()[1]).item(),
    np.nan,
)
kurtosis_observation = ObservationType(
    "y_kurtosis",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: kurtosis(smbo.intensifier.config_selector._collect_data()[1]).item(),
    np.nan,
)
mean_observation = ObservationType(
    "y_mean",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.mean(smbo.intensifier.config_selector._collect_data()[1]),
    np.nan,
)
std_observation = ObservationType(
    "y_std",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: np.std(smbo.intensifier.config_selector._collect_data()[1]),
    np.nan,
)
variability_observation = ObservationType(
    "y_variability",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: calc_variability(smbo.intensifier.config_selector._collect_data()[1]),
    np.nan,
)


ALL_OBSERVATIONS = [
    incumbent_change_observation,
    trials_passed_observation,
    trials_left_observation,
    ubr_observation,
    modelfit_observation,
    dimensions_observation,
    continuous_hp_observation,
    categorical_hp_observation,
    ordinal_hp_observation,
    int_hp_observation,
    tsp_observation,
    knn_entropy_observation,
    skewness_observation,
    kurtosis_observation,
    mean_observation,
    std_observation,
    variability_observation,
]


class ObservationSpace:
    """Manages a collection of observation types and their Gymnasium spaces.

    Allows selection of a subset of available observation types and provides methods to
    compute observations from a SMAC instance.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC instance.
    keys : list[str], optional
        List of observation names to include. If None, all available observations are used.

    Attributes
    ----------
    observation_types : list[ObservationType]
        The list containing all selected observation types.
    observation_space : gymnasium.spaces.Dict
        The Gymnasium Dict space describing the selected observations.

    Methods
    -------
    get_observation(optimizer: SMBO) -> ObsType
        Computes the current observation values from the given optimizer.
    """

    _OBSERVATION_MAP: ClassVar[dict[str, ObservationType]] = {obs.name: obs for obs in ALL_OBSERVATIONS}

    def __init__(self, smac_instance: SMBO, keys: list[str] | None = None) -> None:
        """Initialize the ObservationSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.
        keys : list[str], optional
            List of observation names to include. If None, all available observations are used.

        Raises
        ------
        ValueError
            If any provided key is invalid.
        """
        self._smac_instance = smac_instance

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
    def space(self) -> Space:
        """Returns the Gymnasium Dict space for the selected observations.

        Returns
        -------
        gymnasium.spaces.Dict
            The observation space.
        """
        return self._observation_space

    def get_observation(self) -> ObsType:
        """Compute the current observation values from the given optimizer.

        Returns
        -------
        ObsType
            Dictionary mapping observation names to their computed values.
        """
        return {obs.name: obs.compute(self._smac_instance) for obs in self._observation_types}
