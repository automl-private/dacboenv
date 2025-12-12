"""Observation utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np
from gymnasium.spaces import Box, Dict, Space

from dacboenv.features.signal.modelfit import calculate_model_fit
from dacboenv.features.signal.ubr import calculate_ubr

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

from dacboenv.features.X_features import exploration_tsp, knn_entropy
from dacboenv.features.y_features import calc_variability


def get_best_percentile_configs(smbo: SMBO, p: int = 10, min_samples: int = 1) -> np.ndarray:
    """Returns the best 1/p percent of configs."""
    configs_sorted = [k.config_id for k, _ in sorted(smbo.runhistory._data.items(), key=lambda x: x[1].cost)]
    n = max(min_samples, len(configs_sorted) // p)
    return np.array([smbo.runhistory.get_config(config_id).get_array() for config_id in configs_sorted[:n]])


def get_best_percentile_costs(smbo: SMBO, p: int = 10, min_samples: int = 1) -> np.ndarray:
    """Returns the best 1/p percent of costs."""
    costs_sorted = [v.cost for _, v in sorted(smbo.runhistory._data.items(), key=lambda x: x[1].cost)]
    n = max(min_samples, len(costs_sorted) // p)
    return np.array(costs_sorted[:n])


def enumerate_offset(hyperparameters: Sequence[Any]) -> Iterator[tuple[int, Any]]:
    """Enumerates the given hyperparameters along with their running length as offset."""
    offset = 0
    for hp in hyperparameters:
        yield offset, hp
        offset += hp.n_elements


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
    default: Any


@dataclass
class MultiObservationType:
    """Represents a multi observation type.
    A multi observation is a collection of observation types that are created together.

    Attributes
    ----------
    name : str
        Name of the observation.
    create : Callable[[SMBO], Sequence[ObservationType]]
        Function to create the collection of ObservervationTypes from a SMAC instance.
    """

    name: str
    create: Callable[[SMBO], Sequence[ObservationType]]


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
    -1,
)
modelfit_observation = ObservationType(
    "modelfit_mse",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo: -1 if np.isnan(scores := calculate_model_fit(smbo)["mean_scores"]).any() else scores[0],
    -1,
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
    lambda smbo: len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, FloatHyperparameter)]),
    0,
)
categorical_hp_observation = ObservationType(
    "categorical_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, CategoricalHyperparameter)]),
    0,
)
ordinal_hp_observation = ObservationType(
    "ordinal_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, OrdinalHyperparameter)]),
    0,
)
int_hp_observation = ObservationType(
    "int_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo: len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, IntegerHyperparameter)]),
    0,
)
tsp_observation = ObservationType(
    "tsp",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: exploration_tsp(smbo.intensifier.config_selector._collect_data()[0])[-1],
    -1,
)
knn_entropy_observation = ObservationType(
    "knn_entropy",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: knn_entropy(configs)
    if len(configs := smbo.intensifier.config_selector._collect_data()[0]) > 3  # noqa: PLR2004 (default k == 3)
    else 0,
    0,
)
y_skewness_observation = ObservationType(
    "y_skewness",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan_to_num(skew(costs).item(), nan=0)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else 0,
    0,
)
y_kurtosis_observation = ObservationType(
    "y_kurtosis",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan_to_num(kurtosis(costs).item(), nan=0)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
    else 0,
    0,
)
y_mean_observation = ObservationType(
    "y_mean",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.mean(costs) if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0 else 0,
    0,
)
std_observation = ObservationType(
    "y_std",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: np.std(costs) if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0 else -1,
    -1,
)
variability_observation = ObservationType(
    "y_variability",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: calc_variability(costs)
    if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 3  # noqa: PLR2004
    else -1,
    -1,
)
tsp_best_observation = ObservationType(
    "tsp_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: exploration_tsp(get_best_percentile_configs(smbo))[-1],
    -1,
)
knn_entropy_best_observation = ObservationType(
    "knn_entropy_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: knn_entropy(configs)
    if len(configs := get_best_percentile_configs(smbo, min_samples=4)) > 3  # noqa: PLR2004 (default k == 3)
    else 0,
    0,
)
skewness_best_observation = ObservationType(
    "y_skewness_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan_to_num(skew(costs).item(), nan=0) if len(costs := get_best_percentile_costs(smbo)) > 0 else 0,
    0,
)
kurtosis_best_observation = ObservationType(
    "y_kurtosis_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.nan_to_num(kurtosis(costs).item(), nan=0)
    if len(costs := get_best_percentile_costs(smbo)) > 0
    else 0,
    0,
)
mean_best_observation = ObservationType(
    "y_mean_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo: np.mean(costs) if len(costs := get_best_percentile_costs(smbo)) > 0 else 0,
    0,
)
std_best_observation = ObservationType(
    "y_std_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: np.std(costs) if len(costs := get_best_percentile_costs(smbo)) > 0 else -1,
    -1,
)
variability_best_observation = ObservationType(
    "y_variability_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo: calc_variability(costs)
    if len(costs := get_best_percentile_costs(smbo, min_samples=4)) > 3  # noqa: PLR2004
    else -1,
    -1,
)
gp_hp_observation = MultiObservationType(
    "gp_hp_observations",
    lambda smbo: [
        ObservationType(
            f"gp_hp_{hp.name}{i}_observation",
            Box(hp.bounds[i][0], hp.bounds[i][1]),
            lambda smbo_, idx=i + offset: smbo_._intensifier._config_selector._acquisition_function.model._kernel.theta[  # type: ignore
                idx
            ],
            0,
        )
        for offset, hp in enumerate_offset(
            smbo._intensifier._config_selector._acquisition_function.model._kernel.hyperparameters
        )
        for i in range(hp.n_elements)
        if not hp.fixed
    ],
)

ALL_OBSERVATIONS = [
    incumbent_change_observation,
    trials_passed_observation,
    trials_left_observation,
    ubr_observation,
    # modelfit_observation, # Disabled due to high computation time, behavior similar to UBR
    dimensions_observation,
    continuous_hp_observation,
    categorical_hp_observation,
    ordinal_hp_observation,
    int_hp_observation,
    tsp_observation,
    knn_entropy_observation,
    y_skewness_observation,
    y_kurtosis_observation,
    y_mean_observation,
    std_observation,
    variability_observation,
    tsp_best_observation,
    knn_entropy_best_observation,
    skewness_best_observation,
    kurtosis_best_observation,
    mean_best_observation,
    std_best_observation,
    variability_best_observation,
]

MULTI_OBSERVATIONS = [gp_hp_observation]


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
    _MULTI_OBSERVATION_MAP: ClassVar[dict[str, MultiObservationType]] = {obs.name: obs for obs in MULTI_OBSERVATIONS}

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
        self._keys = (
            keys
            if keys is not None
            else list(ObservationSpace._OBSERVATION_MAP.keys()) + list(ObservationSpace._MULTI_OBSERVATION_MAP.keys())
        )

        # Check for invalid keys
        invalid_keys = (
            set(self._keys)
            - set(ObservationSpace._OBSERVATION_MAP.keys())
            - set(ObservationSpace._MULTI_OBSERVATION_MAP.keys())
        )
        if invalid_keys:
            raise ValueError(f"Invalid observation keys: {invalid_keys}")

        self._observation_types = [
            ObservationSpace._OBSERVATION_MAP[key] for key in self._keys if key in ObservationSpace._OBSERVATION_MAP
        ] + [
            space
            for key in self._keys
            if key in ObservationSpace._MULTI_OBSERVATION_MAP
            for space in ObservationSpace._MULTI_OBSERVATION_MAP[key].create(smac_instance)
        ]
        if len(self._observation_types) == 1:
            self._observation_space = self._observation_types[0].space
        else:
            self._observation_space = Dict({obs.name: obs.space for obs in self._observation_types})

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
        if len(self._observation_types) == 1:
            return np.atleast_1d(self._observation_types[0].compute(self._smac_instance)).astype(np.float32)
        return {
            obs.name: np.atleast_1d(obs.compute(self._smac_instance)).astype(np.float32)
            for obs in self._observation_types
        }
