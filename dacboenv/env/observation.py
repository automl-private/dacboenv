"""Observation utilities for DACBOEnv."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterator, Sequence
from contextlib import suppress
from itertools import islice, takewhile
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

import numpy as np
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    FloatHyperparameter,
    IntegerHyperparameter,
    OrdinalHyperparameter,
)
from ConfigSpace.util import get_one_exchange_neighbourhood
from gymnasium.spaces import Box, Dict, Space
from scipy.stats import kurtosis, norm, rankdata, skew
from smac.main.smbo import SMBO
from smac.runhistory.enumerations import StatusType

from dacboenv.env.action import (
    AbstractActionSpace,
    PosteriorModeActionSpace,
    PosteriorQuantileActionSpace,
)
from dacboenv.env.observations.acquisition_function import (
    GetAFandAcqValue,
    acq_value_ei_observation,
    acq_value_pi_observation,
    acq_value_wei_explore_observation,
    acq_value_wei_observation,
)
from dacboenv.env.observations.types import MultiObservationType, ObservationType, ObsType
from dacboenv.features.signal.modelfit import calculate_model_fit
from dacboenv.features.signal.ubr import calculate_ubr, model_fitted
from dacboenv.features.X_features import exploration_tsd, knn_entropy
from dacboenv.features.y_features import calc_variability
from dacboenv.policy.sawei import apply_moving_iqm
from dacboenv.utils.posterior_decision import (
    EXPECTED_IMPROVEMENT,
    LOWER_CONFIDENCE_BOUND,
    POSTERIOR_MEAN,
    POSTERIOR_MODE_NAMES,
    PosteriorModeAcquisition,
)

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

    from dacboenv.env.observations.types import Memory, ObsType


WEI_ALPHA_LEVELS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
ACTION_CANDIDATE_POOL_SIZE = 512
ACTION_LOCAL_CANDIDATES = 32
ACTION_FEATURE_SEED = 0
MIN_TRANSITION_POINTS = 2
PREDICTION_STD_EPS = 1e-12
ACTION_FEATURE_NDIM = 2
_SELECTED_ACTION_CANDIDATES_ATTRIBUTE = "_dacboenv_selected_action_feature_candidates"

GLOBAL_STATE_NAMES = (
    "budget_percentage",
    "rho_B",
    "d_eff",
    "p_cat",
    "p_int",
    "has_conditionals",
    "normalized_noise",
    "previous_alpha",
    "a_age",
    "p_ts",
    "p_tl",
    "stagnation_age",
    "calibration_error",
)
GLOBAL_STATE_INDEX = {name: index for index, name in enumerate(GLOBAL_STATE_NAMES)}
GLOBAL_STATE_INDEX["previous_control"] = GLOBAL_STATE_INDEX["previous_alpha"]

ACTION_FEATURE_NAMES = (
    "alpha",
    "standardized_improvement",
    "normalized_uncertainty",
    "novelty",
)
ACTION_FEATURE_INDEX = {name: index for index, name in enumerate(ACTION_FEATURE_NAMES)}
ACTION_FEATURE_INDEX["control_value"] = ACTION_FEATURE_INDEX["alpha"]

AF_ACTION_FEATURE_NAMES = (
    *(f"mode_{mode}" for mode in POSTERIOR_MODE_NAMES),
    "parameter_value",
    "standardized_improvement",
    "normalized_uncertainty",
    "novelty",
    "normalized_ei_rank",
)
AF_ACTION_FEATURE_INDEX = {name: index for index, name in enumerate(AF_ACTION_FEATURE_NAMES)}


def _scalar_cost(cost: Any) -> float:
    """Convert a single-objective SMAC cost to a scalar."""
    values = np.asarray(cost, dtype=float).reshape(-1)
    if values.size == 0:
        return np.inf

    value = float(values[0])
    return value if np.isfinite(value) else np.inf


def _cost_history(smbo: SMBO) -> np.ndarray:
    """Return costs in run-history insertion order.

    Non-finite trials are retained as ``+inf``. Dropping them would shift the
    trial timeline and could make progress or rewards from an earlier trial be
    emitted a second time.
    """
    return np.asarray(
        [
            _scalar_cost(value.cost) if getattr(value, "status", StatusType.SUCCESS) == StatusType.SUCCESS else np.inf
            for value in smbo.runhistory._data.values()
        ],
        dtype=float,
    )


def _incumbent_history(smbo: SMBO) -> np.ndarray:
    """Return the incumbent after every finished evaluation."""
    costs = _cost_history(smbo)
    return np.minimum.accumulate(costs) if costs.size > 0 else costs


def _robust_location_and_scale(values: np.ndarray) -> tuple[float, float]:
    """Return a robust location and a conservative positive scale."""
    finite_values = np.asarray(values, dtype=float).reshape(-1)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return 0.0, 1.0

    location = float(np.median(finite_values))
    mad = 1.4826 * float(np.median(np.abs(finite_values - location)))
    q25, q75 = np.quantile(finite_values, [0.25, 0.75])
    iqr = float(q75 - q25) / 1.349
    std = float(np.std(finite_values))
    scale = max(mad, iqr, std)

    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = max(abs(location), 1.0)

    return location, scale


def _initial_design_size(smbo: SMBO) -> int:
    selector = smbo.intensifier.config_selector
    return len(selector._initial_design_configs)


def _objective_scale(smbo: SMBO) -> float:
    """Return an episode-fixed objective scale from the initial design."""
    costs = _cost_history(smbo)
    n_initial = min(_initial_design_size(smbo), costs.size)
    _, scale = _robust_location_and_scale(costs[:n_initial])
    return scale


def _effective_dimension(smbo: SMBO) -> float:
    """Return the mean number of active hyperparameters in evaluated configs."""
    configs = smbo.runhistory.get_configs()
    if len(configs) == 0:
        return float(len(smbo._scenario.configspace))

    active_dimensions = [float(np.isfinite(np.asarray(config.get_array(), dtype=float)).sum()) for config in configs]
    return float(np.mean(active_dimensions))


def calculate_budget_density(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return the log budget per effective dimension."""
    del memory
    return float(np.log1p(smbo._scenario.n_trials / (_effective_dimension(smbo) + 1.0)))


def calculate_log_effective_dimension(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return log effective dimensionality."""
    del memory
    return float(np.log1p(_effective_dimension(smbo)))


def _fraction_of_hyperparameters(smbo: SMBO, hp_types: tuple[type, ...]) -> float:
    configspace = smbo._scenario.configspace
    n_hps = len(configspace)
    if n_hps == 0:
        return 0.0

    return float(sum(isinstance(hp, hp_types) for hp in configspace.values()) / n_hps)


def calculate_categorical_fraction(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return the fraction of nominal hyperparameters that are categorical."""
    del memory
    return _fraction_of_hyperparameters(smbo, (CategoricalHyperparameter,))


def calculate_integer_ordinal_fraction(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return the fraction of nominal hyperparameters that are integer/ordinal."""
    del memory
    return _fraction_of_hyperparameters(smbo, (IntegerHyperparameter, OrdinalHyperparameter))


def calculate_has_conditionals(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return whether the configuration space contains conditions."""
    del memory
    return float(len(smbo._scenario.configspace.conditions) > 0)


def _model_target_scale(smbo: SMBO) -> float:
    """Return a robust scale in the surrogate model's output space."""
    try:
        Y = smbo.intensifier.config_selector._collect_data()[1]
    except (AttributeError, RuntimeError, ValueError):
        return 1.0

    return _robust_location_and_scale(np.asarray(Y, dtype=float))[1]


def calculate_normalized_noise(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return surrogate noise relative to the initial objective scale.

    ``-1`` denotes an unavailable model-specific noise estimate and ``0``
    denotes a deterministic objective.
    """
    del memory
    if bool(getattr(smbo._scenario, "deterministic", False)):
        return 0.0

    model = smbo.intensifier.config_selector._model
    gp = getattr(model, "_gp", None)
    kernel = getattr(gp, "kernel_", None)
    if kernel is None:
        kernel = getattr(model, "_kernel", None)
    if kernel is None or not hasattr(kernel, "theta"):
        return -1.0

    theta = np.asarray(kernel.theta, dtype=float).reshape(-1)
    theta_offset = 0
    noise_variances: list[float] = []
    for hp in kernel.hyperparameters:
        if hp.fixed:
            continue

        n_elements = int(hp.n_elements)
        values = theta[theta_offset : theta_offset + n_elements]
        theta_offset += n_elements
        if "noise" in hp.name.lower():
            noise_variances.extend(np.exp(values).tolist())

    if len(noise_variances) == 0:
        return -1.0

    noise_std = float(np.sqrt(max(noise_variances)))
    if bool(getattr(model, "_normalize_y", False)):
        noise_std *= float(getattr(model, "std_y_", 1.0))

    return float(np.log1p(noise_std / _objective_scale(smbo)))


def calculate_current_alpha(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return the actual WEI alpha currently installed in SMAC."""
    del memory
    acquisition_function = smbo.intensifier.config_selector._acquisition_function
    alpha = getattr(acquisition_function, "_alpha", 0.5)
    return float(np.clip(np.asarray(alpha).item(), 0.0, 1.0))


def calculate_parameter_age(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return the budget fraction since alpha last changed."""
    alpha_history = [] if memory is None else memory.get("alpha", [])
    if len(alpha_history) == 0:
        return 0.0

    current_alpha = float(alpha_history[-1])
    trailing_run = sum(
        1
        for _ in takewhile(
            lambda alpha: np.isclose(alpha, current_alpha),
            reversed(alpha_history),
        )
    )
    # The first record establishes the action and has age zero.
    age = max(trailing_run - 1, 0)
    return float(min(age / max(smbo._scenario.n_trials, 1), 1.0))


def calculate_recent_progress(smbo: SMBO, horizon_fraction: float) -> float:
    """Return scale-normalized incumbent improvement over a trial horizon."""
    incumbent_history = _incumbent_history(smbo)
    if incumbent_history.size < MIN_TRANSITION_POINTS:
        return 0.0

    horizon = max(1, round(horizon_fraction * smbo._scenario.n_trials))
    start = max(0, incumbent_history.size - 1 - horizon)
    previous = float(incumbent_history[start])
    current = float(incumbent_history[-1])
    if not np.isfinite(previous) or not np.isfinite(current):
        return 0.0

    improvement = max(previous - current, 0.0)
    return float(np.arcsinh(improvement / _objective_scale(smbo)) / horizon)


def calculate_short_progress(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return recent progress over five percent of the BO budget."""
    del memory
    return calculate_recent_progress(smbo, horizon_fraction=0.05)


def calculate_long_progress(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return recent progress over twenty percent of the BO budget."""
    del memory
    return calculate_recent_progress(smbo, horizon_fraction=0.20)


def calculate_stagnation_age(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return trials since the last significant incumbent improvement."""
    del memory
    incumbent_history = _incumbent_history(smbo)
    if incumbent_history.size < MIN_TRANSITION_POINTS:
        return 0.0

    threshold = 1e-3 * _objective_scale(smbo)
    improvements = incumbent_history[:-1] - incumbent_history[1:]
    significant = np.flatnonzero(np.isfinite(improvements) & (improvements > threshold))
    last_improvement = int(significant[-1] + 1) if significant.size > 0 else 0
    age = int(incumbent_history.size - 1 - last_improvement)
    normalization = max(1, round(0.20 * smbo._scenario.n_trials))
    return float(np.clip(age / normalization, 0.0, 1.0))


def calculate_prequential_calibration(smbo: SMBO) -> float:
    """Return the latest clipped prequential standardized residual."""
    selector = smbo.intensifier.config_selector
    if smbo.runhistory.finished <= _initial_design_size(smbo):
        return -1.0

    model = selector._model
    if not model_fitted(model):
        return -1.0

    try:
        X, Y = selector._collect_data()[:2]
        if X.shape[0] == 0 or Y.shape[0] == 0:
            return -1.0

        # A valid prequential residual requires exactly one unseen response.
        if getattr(selector, "_previous_entries", -1) != Y.shape[0] - 1:
            return -1.0

        mean, variance = model.predict_marginalized(X[-1:].copy())
        predicted_mean = float(np.asarray(mean).reshape(-1)[0])
        predicted_variance = max(float(np.asarray(variance).reshape(-1)[0]), 0.0)
        target = float(np.asarray(Y[-1]).reshape(-1)[0])
    except (AttributeError, RuntimeError, ValueError, IndexError):
        return -1.0

    standardized_error = abs(target - predicted_mean) / max(np.sqrt(predicted_variance), 1e-12)
    return float(np.clip(standardized_error, 0.0, 10.0) / 10.0)


def calculate_calibration_error(smbo: SMBO, memory: Memory | None = None) -> float:
    """Return an EMA of valid prequential calibration errors."""
    del smbo
    if memory is None:
        return -1.0

    values = [float(value) for value in memory.get("calibration", []) if value is not None and float(value) >= 0.0]
    if len(values) == 0:
        return -1.0

    ema = values[0]
    for value in values[1:]:
        ema = 0.9 * ema + 0.1 * value
    return float(np.clip(ema, 0.0, 1.0))


def _synchronize_model(smbo: SMBO) -> None:
    """Fit the sequential selector to the current history after ``tell``.

    ConfigSelector normally performs this work on the next ``ask``. Structured
    action probes are part of the state before that ask, so for sequential
    selectors we mirror the selector's own fit/update block and mark the data
    as consumed to avoid a duplicate fit.
    """
    selector = smbo.intensifier.config_selector
    if getattr(selector, "_retrain_after", 1) != 1:
        return

    model = selector._model
    if model is None:
        return

    try:
        X, Y, X_configurations = selector._collect_data()
    except (AttributeError, RuntimeError, ValueError):
        return

    if X.shape[0] == 0 or getattr(selector, "_previous_entries", -1) == Y.shape[0]:
        return

    try:
        model.train(X, Y)
        x_best_array, best_observation = selector._get_x_best(X_configurations)
        selector._acquisition_function.update(
            model=model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(selector._get_evaluated_configs()),
            X=X_configurations,
        )
    except (AttributeError, RuntimeError, ValueError, np.linalg.LinAlgError):
        return

    selector._previous_entries = Y.shape[0]


def calculate_global_state(smbo: SMBO, memory: Memory | None = None) -> np.ndarray:
    """Return the compact global BO state."""
    previous_alpha = (
        float(memory["alpha"][-1])
        if memory is not None and len(memory.get("alpha", [])) > 0
        else calculate_current_alpha(smbo)
    )
    return np.asarray(
        [
            len(smbo.runhistory) / max(smbo._scenario.n_trials, 1),
            calculate_budget_density(smbo),
            calculate_log_effective_dimension(smbo),
            calculate_categorical_fraction(smbo),
            calculate_integer_ordinal_fraction(smbo),
            calculate_has_conditionals(smbo),
            calculate_normalized_noise(smbo),
            previous_alpha,
            calculate_parameter_age(smbo, memory),
            calculate_short_progress(smbo),
            calculate_long_progress(smbo),
            calculate_stagnation_age(smbo),
            calculate_calibration_error(smbo, memory),
        ],
        dtype=np.float32,
    )


def _feature_sampling_seed(smbo: SMBO) -> int:
    scenario_seed = int(getattr(smbo._scenario, "seed", 0) or 0)
    trial = int(getattr(smbo.runhistory, "finished", len(smbo.runhistory)))
    return int((ACTION_FEATURE_SEED + 1_000_003 * scenario_seed + 9_176 * trial) % (2**31 - 1))


def _sample_action_candidates(smbo: SMBO) -> list[Any]:
    """Build one deterministic candidate pool shared by every alpha."""
    seed = _feature_sampling_seed(smbo)
    candidates: list[Any] = []

    incumbents = smbo.intensifier.get_incumbents(sort_by="cost")
    if len(incumbents) > 0 and ACTION_LOCAL_CANDIDATES > 0:
        with suppress(RuntimeError, ValueError):
            candidates.extend(
                islice(
                    get_one_exchange_neighbourhood(
                        incumbents[0],
                        seed=seed,
                        num_neighbors=4,
                        stdev=0.2,
                    ),
                    ACTION_LOCAL_CANDIDATES,
                )
            )

    n_random = max(ACTION_CANDIDATE_POOL_SIZE - len(candidates), 0)
    if n_random > 0:
        # Sampling observation probes must not advance SMAC's ConfigSpace RNG.
        configspace = copy.deepcopy(smbo._scenario.configspace)
        configspace.seed(seed + 1)
        sampled = configspace.sample_configuration(size=n_random)
        if isinstance(sampled, list):
            candidates.extend(sampled)
        else:
            candidates.append(sampled)

    evaluated_keys = {
        tuple(
            np.nan_to_num(
                np.asarray(config.get_array(), dtype=float),
                nan=-1e30,
            )
            .round(12)
            .tolist()
        )
        for config in smbo.runhistory.get_configs()
    }
    unique: dict[tuple[float, ...], Any] = {}
    for config in candidates:
        vector = np.asarray(config.get_array(), dtype=float)
        key = tuple(np.nan_to_num(vector, nan=-1e30).round(12).tolist())
        if key in evaluated_keys:
            continue
        unique.setdefault(key, config)
    return list(unique.values())


def _is_inactive(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False


def _configuration_distance(config_a: Any, config_b: Any, hyperparameters: list[Any]) -> float:
    """Return a mixed-space Gower distance in ``[0, 1]``."""
    if len(hyperparameters) == 0:
        return 0.0

    distances: list[float] = []
    for hp in hyperparameters:
        value_a = config_a.get(hp.name, None)
        value_b = config_b.get(hp.name, None)
        inactive_a = _is_inactive(value_a)
        inactive_b = _is_inactive(value_b)
        if inactive_a and inactive_b:
            distances.append(0.0)
            continue
        if inactive_a != inactive_b:
            distances.append(1.0)
            continue

        if isinstance(hp, CategoricalHyperparameter):
            distance = float(value_a != value_b)
        elif isinstance(hp, OrdinalHyperparameter):
            index_a = hp.sequence.index(value_a)
            index_b = hp.sequence.index(value_b)
            distance = abs(index_a - index_b) / max(len(hp.sequence) - 1, 1)
        elif isinstance(hp, (FloatHyperparameter, IntegerHyperparameter)):
            lower = float(hp.lower)
            upper = float(hp.upper)
            transformed_a = float(value_a)
            transformed_b = float(value_b)
            if hp.log:
                lower, upper = np.log([lower, upper])
                transformed_a = float(np.log(transformed_a))
                transformed_b = float(np.log(transformed_b))
            distance = abs(transformed_a - transformed_b) / max(upper - lower, 1e-12)
        else:
            distance = float(value_a != value_b)

        distances.append(float(np.clip(distance, 0.0, 1.0)))

    return float(np.mean(distances))


def _candidate_novelty(smbo: SMBO, candidate: Any) -> float:
    evaluated = smbo.runhistory.get_configs()
    if len(evaluated) == 0:
        return 1.0

    hyperparameters = list(smbo._scenario.configspace.values())
    return float(min(_configuration_distance(candidate, config, hyperparameters) for config in evaluated))


def calculate_action_features(smbo: SMBO, memory: Memory | None = None) -> np.ndarray:
    """Return WEI consequences for each candidate alpha.

    Columns are ``[alpha, clipped z, normalized uncertainty, novelty]``.
    """
    del memory
    features = np.zeros((len(WEI_ALPHA_LEVELS), len(ACTION_FEATURE_NAMES)), dtype=np.float32)
    features[:, ACTION_FEATURE_INDEX["alpha"]] = WEI_ALPHA_LEVELS
    selected_candidates: list[Any | None] = [None] * len(WEI_ALPHA_LEVELS)
    setattr(smbo, _SELECTED_ACTION_CANDIDATES_ATTRIBUTE, selected_candidates)

    selector = smbo.intensifier.config_selector
    model = selector._model
    if not model_fitted(model):
        return features

    candidates = _sample_action_candidates(smbo)
    if len(candidates) == 0:
        return features

    try:
        candidate_arrays = np.vstack([np.asarray(config.get_array(), dtype=float) for config in candidates])
        mean, variance = model.predict_marginalized(candidate_arrays)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        variance = np.maximum(np.asarray(variance, dtype=float).reshape(-1), 0.0)
        std = np.sqrt(variance)

        acquisition_function = selector._acquisition_function
        eta = float(np.asarray(acquisition_function._eta).item())
        xi = float(getattr(acquisition_function, "_xi", 0.0))
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return features

    improvement = eta - mean - xi
    z = np.zeros_like(improvement)
    nonzero_std = std > PREDICTION_STD_EPS
    z[nonzero_std] = improvement[nonzero_std] / std[nonzero_std]

    exploitation = np.zeros_like(improvement)
    exploitation[nonzero_std] = improvement[nonzero_std] * norm.cdf(z[nonzero_std])
    exploration = np.zeros_like(std)
    exploration[nonzero_std] = std[nonzero_std] * norm.pdf(z[nonzero_std])

    scores = (
        WEI_ALPHA_LEVELS[:, None] * exploitation[None, :] + (1.0 - WEI_ALPHA_LEVELS[:, None]) * exploration[None, :]
    )
    scores[:, ~np.isfinite(scores).all(axis=0)] = -np.inf
    model_scale = _model_target_scale(smbo)

    for row, alpha in enumerate(WEI_ALPHA_LEVELS):
        if not np.isfinite(scores[row]).any():
            continue

        best_index = int(np.nanargmax(scores[row]))
        selected_candidates[row] = candidates[best_index]
        features[row] = np.asarray(
            [
                alpha,
                np.clip(z[best_index], -5.0, 5.0) / 5.0,
                np.log1p(std[best_index] / model_scale),
                _candidate_novelty(smbo, candidates[best_index]),
            ],
            dtype=np.float32,
        )

    return features


def _current_model_incumbent(smbo: SMBO) -> float | None:
    """Return the incumbent in the surrogate model's target representation."""
    selector = smbo.intensifier.config_selector
    acquisition_function = selector._acquisition_function
    eta = getattr(acquisition_function, "_eta", None)
    if eta is not None:
        value = float(np.asarray(eta).item())
        return value if np.isfinite(value) else None

    try:
        _X, _Y, configurations = selector._collect_data()
        _incumbent, best_observation = selector._get_x_best(configurations)
        value = float(np.asarray(best_observation).item())
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None
    return value if np.isfinite(value) else None


def _action_candidate_posterior(
    smbo: SMBO,
) -> tuple[list[Any], np.ndarray, np.ndarray] | None:
    """Predict mean and standard deviation on one shared candidate pool."""
    selector = smbo.intensifier.config_selector
    model = selector._model
    if not model_fitted(model):
        return None

    candidates = _sample_action_candidates(smbo)
    if len(candidates) == 0:
        return None

    try:
        candidate_arrays = np.vstack(
            [np.asarray(configuration.get_array(), dtype=float) for configuration in candidates]
        )
        mean, variance = model.predict_marginalized(candidate_arrays)
        mean = np.asarray(mean, dtype=float).reshape(-1)
        variance = np.maximum(
            np.asarray(variance, dtype=float).reshape(-1),
            0.0,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError, IndexError):
        return None

    if mean.shape != variance.shape or mean.size != len(candidates):
        return None
    return candidates, mean, np.sqrt(variance)


def _standardized_improvement(
    eta: float | None,
    mean: float,
    std: float,
) -> float:
    """Return the clipped, normalized improvement z-score used by WEI."""
    if eta is None or not np.isfinite(mean) or not np.isfinite(std):
        return 0.0
    if std <= PREDICTION_STD_EPS:
        return 0.0
    return float(np.clip((eta - mean) / std, -5.0, 5.0) / 5.0)


def calculate_posterior_quantile_action_features(
    smbo: SMBO,
    quantile_levels: Sequence[float],
    memory: Memory | None = None,
) -> np.ndarray:
    """Return consequences for exact posterior-quantile actions.

    Columns retain the WEI structured contract:
    ``[control value, standardized improvement, uncertainty, novelty]``.
    Here the first value is the posterior quantile ``q``. Lower and upper
    quantiles use the same score, ``-(mu + Phi^-1(q) sigma)``, because SMAC
    maximizes acquisition values for a minimization problem.
    """
    del memory
    quantiles = np.asarray(quantile_levels, dtype=float)
    features = np.zeros(
        (len(quantiles), len(ACTION_FEATURE_NAMES)),
        dtype=np.float32,
    )
    features[:, ACTION_FEATURE_INDEX["control_value"]] = quantiles
    selected_candidates: list[Any | None] = [None] * len(quantiles)
    setattr(smbo, _SELECTED_ACTION_CANDIDATES_ATTRIBUTE, selected_candidates)

    posterior = _action_candidate_posterior(smbo)
    if posterior is None:
        return features

    candidates, mean, std = posterior
    quantile_z = norm.ppf(quantiles)
    scores = -(mean[None, :] + quantile_z[:, None] * std[None, :])
    scores[:, ~np.isfinite(scores).all(axis=0)] = -np.inf
    eta = _current_model_incumbent(smbo)
    model_scale = _model_target_scale(smbo)

    for row, quantile in enumerate(quantiles):
        if not np.isfinite(scores[row]).any():
            continue

        best_index = int(np.argmax(scores[row]))
        selected_candidates[row] = candidates[best_index]
        features[row] = np.asarray(
            [
                quantile,
                _standardized_improvement(
                    eta,
                    mean[best_index],
                    std[best_index],
                ),
                np.log1p(std[best_index] / model_scale),
                _candidate_novelty(smbo, candidates[best_index]),
            ],
            dtype=np.float32,
        )

    return features


def _normalized_ranks(values: np.ndarray) -> np.ndarray:
    """Return stable ranks in ``[0, 1]`` with non-finite values ranked last."""
    flat_values = np.asarray(values, dtype=float).reshape(-1)
    if flat_values.size <= 1:
        return np.ones(flat_values.size, dtype=float)

    finite = np.isfinite(flat_values)
    if not finite.any():
        return np.zeros(flat_values.size, dtype=float)

    floor = float(np.min(flat_values[finite]))
    safe_values = np.where(finite, flat_values, np.nextafter(floor, -np.inf))
    return (rankdata(safe_values, method="average") - 1.0) / (flat_values.size - 1.0)


def calculate_af_action_features(
    smbo: SMBO,
    memory: Memory | None = None,
) -> np.ndarray:
    """Return candidate consequences for the five posterior operation modes.

    The feature table contains a one-hot operation identity, the mode's fixed
    scalar parameter, standardized improvement, uncertainty, novelty, and the
    candidate's rank under ordinary EI. No raw values are compared across
    acquisition families.
    """
    del memory
    n_modes = len(POSTERIOR_MODE_NAMES)
    features = AF_ACTION_FEATURE_DEFAULT.copy()
    selected_candidates: list[Any | None] = [None] * n_modes
    setattr(smbo, _SELECTED_ACTION_CANDIDATES_ATTRIBUTE, selected_candidates)
    selector = smbo.intensifier.config_selector
    acquisition_function = selector._acquisition_function
    if not isinstance(acquisition_function, PosteriorModeAcquisition):
        return features
    features[
        POSTERIOR_MODE_NAMES.index(LOWER_CONFIDENCE_BOUND),
        AF_ACTION_FEATURE_INDEX["parameter_value"],
    ] = acquisition_function.lower_quantile

    posterior = _action_candidate_posterior(smbo)
    if posterior is None:
        return features

    candidates, mean, std = posterior

    eta = _current_model_incumbent(smbo)
    if eta is None:
        return features
    if acquisition_function._eta is None:
        acquisition_function.update(model=selector._model, eta=eta)

    variance = np.square(std)
    scores = acquisition_function.score_predictions(mean, variance)
    best_indices = acquisition_function.best_candidate_indices(mean, variance)
    ei_ranks = _normalized_ranks(scores[EXPECTED_IMPROVEMENT])
    model_scale = _model_target_scale(smbo)

    parameter_values = {
        POSTERIOR_MEAN: 0.5,
        LOWER_CONFIDENCE_BOUND: acquisition_function.lower_quantile,
    }
    for row, mode in enumerate(POSTERIOR_MODE_NAMES):
        best_index = best_indices.get(mode)
        if best_index is None or not 0 <= best_index < len(candidates):
            continue
        if not (np.isfinite(mean[best_index]) and np.isfinite(std[best_index]) and np.isfinite(ei_ranks[best_index])):
            continue

        selected_candidates[row] = candidates[best_index]

        features[
            row,
            AF_ACTION_FEATURE_INDEX["parameter_value"],
        ] = parameter_values.get(mode, 0.0)
        features[
            row,
            AF_ACTION_FEATURE_INDEX["standardized_improvement"],
        ] = _standardized_improvement(
            eta,
            mean[best_index],
            std[best_index],
        )
        features[
            row,
            AF_ACTION_FEATURE_INDEX["normalized_uncertainty"],
        ] = np.log1p(std[best_index] / model_scale)
        features[row, AF_ACTION_FEATURE_INDEX["novelty"]] = _candidate_novelty(
            smbo,
            candidates[best_index],
        )
        features[
            row,
            AF_ACTION_FEATURE_INDEX["normalized_ei_rank"],
        ] = ei_ranks[best_index]

    if features.shape[0] != n_modes:
        raise RuntimeError(f"Expected one feature row per posterior mode, got {features.shape}.")
    return features


GLOBAL_STATE_LOW = np.asarray([0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1], dtype=np.float32)
GLOBAL_STATE_HIGH = np.asarray(
    [1, np.inf, np.inf, 1, 1, 1, np.inf, 1, 1, np.inf, np.inf, 1, 1],
    dtype=np.float32,
)
GLOBAL_STATE_DEFAULT = np.zeros(len(GLOBAL_STATE_NAMES), dtype=np.float32)
GLOBAL_STATE_DEFAULT[GLOBAL_STATE_INDEX["normalized_noise"]] = -1.0
GLOBAL_STATE_DEFAULT[GLOBAL_STATE_INDEX["previous_alpha"]] = 0.5
GLOBAL_STATE_DEFAULT[GLOBAL_STATE_INDEX["calibration_error"]] = -1.0

ACTION_FEATURE_LOW = np.tile(
    np.asarray([0, -1, 0, 0], dtype=np.float32),
    (len(WEI_ALPHA_LEVELS), 1),
)
ACTION_FEATURE_HIGH = np.tile(
    np.asarray([1, 1, np.inf, 1], dtype=np.float32),
    (len(WEI_ALPHA_LEVELS), 1),
)
ACTION_FEATURE_DEFAULT = np.zeros_like(ACTION_FEATURE_LOW)
ACTION_FEATURE_DEFAULT[:, ACTION_FEATURE_INDEX["alpha"]] = WEI_ALPHA_LEVELS

AF_ACTION_FEATURE_LOW = np.tile(
    np.asarray(
        [*([0.0] * len(POSTERIOR_MODE_NAMES)), 0.0, -1.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    ),
    (len(POSTERIOR_MODE_NAMES), 1),
)
AF_ACTION_FEATURE_HIGH = np.tile(
    np.asarray(
        [*([1.0] * len(POSTERIOR_MODE_NAMES)), 1.0, 1.0, np.inf, 1.0, 1.0],
        dtype=np.float32,
    ),
    (len(POSTERIOR_MODE_NAMES), 1),
)
AF_ACTION_FEATURE_DEFAULT = np.zeros_like(AF_ACTION_FEATURE_LOW)
for mode_index, mode in enumerate(POSTERIOR_MODE_NAMES):
    AF_ACTION_FEATURE_DEFAULT[
        mode_index,
        AF_ACTION_FEATURE_INDEX[f"mode_{mode}"],
    ] = 1.0
AF_ACTION_FEATURE_DEFAULT[
    POSTERIOR_MODE_NAMES.index(POSTERIOR_MEAN),
    AF_ACTION_FEATURE_INDEX["parameter_value"],
] = 0.5
AF_ACTION_FEATURE_DEFAULT[
    POSTERIOR_MODE_NAMES.index(LOWER_CONFIDENCE_BOUND),
    AF_ACTION_FEATURE_INDEX["parameter_value"],
] = 0.1


def get_best_percentile_configs(
    smbo: SMBO,
    p: int = 10,
    min_samples: int = 1,
    memory: Memory | None = None,  # noqa: ARG001
) -> np.ndarray:
    """Returns the best 1/p percent of configs."""
    configs_sorted = [k.config_id for k, _ in sorted(smbo.runhistory._data.items(), key=lambda x: x[1].cost)]
    n = max(min_samples, len(configs_sorted) // p)
    return np.array([smbo.runhistory.get_config(config_id).get_array() for config_id in configs_sorted[:n]])


def get_best_percentile_costs(
    smbo: SMBO,
    p: int = 10,
    min_samples: int = 1,
    memory: Memory | None = None,  # noqa: ARG001
) -> np.ndarray:
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


def calc_last_diff(memory: Memory, key: str) -> float:
    """Calc the last difference in a signal.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features.
    key : str
        The name of the observation.

    Returns
    -------
    float
        The last diff.
    """
    return 0 if len(memory[key]) < 2 else memory[key][-2] - memory[key][-1]  # noqa: PLR2004


class ComputeUBR:
    """Compute the UBR.

    Currently no need to have it as a class.
    """

    def __call__(self, smbo: SMBO) -> float | None:
        """Compute the UBR.

        Parameters
        ----------
        smbo : SMBO
            The SMAC instance.

        Returns
        -------
        float
            The current UBR.
        """
        result_dict = calculate_ubr(trial_infos=None, trial_values=None, configspace=None, smbo=smbo)
        return result_dict.get("ubr", 0)


def get_last_val(memory: Memory, key: str) -> float:
    """Get the last value of a signal in memory.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features.
    key : str
        The name of the observation.

    Returns
    -------
    float
        The last/newest value.
    """
    return memory[key][-1]


def ubr_difference(memory: Memory) -> float:
    """Computes the difference between the last two KNN values."""
    return calc_last_diff(memory=memory, key="ubr")


def knn_difference(memory: Memory) -> float:
    """Computes the difference between the last two KNN values."""
    return calc_last_diff(memory=memory, key="knn")


incumbent_change_observation = ObservationType(
    "incumbent_changes",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: smbo.intensifier.incumbents_changed,  # noqa: ARG005
    0,
)
trials_passed_observation = ObservationType(
    "trials_passed",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: len(smbo.runhistory),  # noqa: ARG005
    0,
)
trials_left_observation = ObservationType(
    "trials_left",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: smbo.remaining_trials,  # noqa: ARG005
    -1,
)
ubr_observation = ObservationType(
    "ubr",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: get_last_val(memory=memory, key="ubr"),  # type: ignore[arg-type] # noqa: ARG005
    -1,
)


def calc_gradient(memory: Memory, key: str, smooth_signal: bool = False) -> np.ndarray:  # noqa: FBT001, FBT002
    """Calc the gradient of a signal in memory.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features/observations.
    key : str
        The name of the observation.
    smooth_signal : bool, optional
        Whether to smooth the signal, by default False. If True, a moving IQM is applied with a window length of 7.
        This also introduces a slight delay in the signal.

    Returns
    -------
    np.ndarray
        The gradient of a signal, possibly smoothed.
    """
    raw_signal = memory[key]
    raw_signal = [r for r in raw_signal if r is not None]
    if len(raw_signal) == 0:
        return np.array([0])
    maybe_smoothed_signal = apply_moving_iqm(raw_signal, window_size=7) if smooth_signal else raw_signal
    if len(maybe_smoothed_signal) == 1:
        return np.array([0])
    if len(maybe_smoothed_signal) < 3:  # noqa: PLR2004
        return np.diff(maybe_smoothed_signal)
    return np.gradient(maybe_smoothed_signal)


def calc_ubr_gradient(memory: Memory, smooth_signal: bool = False) -> np.ndarray:  # noqa: FBT001, FBT002
    """Calc the gradient of the UBR.

    Parameters
    ----------
    memory : Memory
        The memory/history of state features/observations.
    smooth_signal : bool, optional
        Whether to smooth the signal, by default False. If True, a moving IQM is applied with a window length of 7.
        This also introduces a slight delay in the signal.

    Returns
    -------
    np.ndarray
        The gradient of UBR, possibly smoothed.
    """
    return calc_gradient(memory=memory, key="ubr", smooth_signal=smooth_signal)


ubr_gradient_observation = ObservationType(
    "ubr_gradient",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_ubr_gradient(memory=memory, smooth_signal=False)[-1],  # type: ignore[arg-type] # noqa: ARG005
    0,
)
ubr_smoothed_gradient_observation = ObservationType(
    "ubr_smoothed_gradient",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: calc_ubr_gradient(memory=memory, smooth_signal=True)[-1],  # type: ignore[arg-type] # noqa: ARG005
    0,
)
ubr_smoothed_gradient_std_observation = ObservationType(
    "ubr_smoothed_gradient_std",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(calc_ubr_gradient(memory=memory, smooth_signal=True)),  # type: ignore[arg-type] # noqa: ARG005
    0,
)
modelfit_observation = ObservationType(
    "modelfit_mse",
    Box(low=0.0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: -1 if np.isnan(scores := calculate_model_fit(smbo)["mean_scores"]).any() else scores[0],  # noqa: ARG005
    -1,
)
dimensions_observation = ObservationType(
    "searchspace_dim",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(smbo._scenario.configspace),  # noqa: ARG005
    0,
)
continuous_hp_observation = ObservationType(
    "continuous_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, FloatHyperparameter)]),  # noqa: ARG005
    0,
)
categorical_hp_observation = ObservationType(
    "categorical_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(  # noqa: ARG005
        [hp for hp in smbo._scenario.configspace.values() if isinstance(hp, CategoricalHyperparameter)]
    ),
    0,
)
ordinal_hp_observation = ObservationType(
    "ordinal_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(  # noqa: ARG005
        [hp for hp in smbo._scenario.configspace.values() if isinstance(hp, OrdinalHyperparameter)]
    ),
    0,
)
int_hp_observation = ObservationType(
    "int_hps",
    Box(low=0, high=np.inf, dtype=np.int32),
    lambda smbo, memory: len(  # noqa: ARG005
        [hp for hp in smbo._scenario.configspace.values() if isinstance(hp, IntegerHyperparameter)]
    ),
    0,
)
tsd_observation = ObservationType(
    "tsd",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: exploration_tsd(smbo.intensifier.config_selector._collect_data()[0])[-1],  # noqa: ARG005
    -1,
)


def calculate_knn(smbo: SMBO) -> float:
    """Calculate KNN exploration measure following Papenmeier et al. (2025, Exploring exploration in BO).

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.

    Returns
    -------
    float
        The KNN value.
    """
    if len(configs := smbo.intensifier.config_selector._collect_data()[0]) > 3:  # noqa: PLR2004 (default k == 3)
        return knn_entropy(configs)
    return 0


knn_entropy_observation = ObservationType(
    "knn_entropy",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: get_last_val(memory=memory, key="knn"),  # type: ignore[arg-type] # noqa: ARG005
    0,
)
y_skewness_observation = ObservationType(
    "y_skewness",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        np.nan_to_num(skew(costs).item(), nan=0)
        if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
        else 0
    ),
    0,
)
y_kurtosis_observation = ObservationType(
    "y_kurtosis",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        np.nan_to_num(kurtosis(costs).item(), nan=0)
        if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0
        else 0
    ),
    0,
)
y_mean_observation = ObservationType(
    "y_mean",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.mean(costs) if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0 else 0,  # noqa: ARG005
    0,
)
std_observation = ObservationType(
    "y_std",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(costs) if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 0 else -1,  # noqa: ARG005
    -1,
)
variability_observation = ObservationType(
    "y_variability",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        calc_variability(costs)
        if len(costs := smbo.intensifier.config_selector._collect_data()[1]) > 3  # noqa: PLR2004
        else -1
    ),
    -1,
)
tsd_best_observation = ObservationType(
    "tsd_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: exploration_tsd(get_best_percentile_configs(smbo))[-1],  # noqa: ARG005
    -1,
)
knn_entropy_best_observation = ObservationType(
    "knn_entropy_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        knn_entropy(configs)  # type: ignore[arg-type]
        if len(configs := get_best_percentile_configs(smbo, min_samples=4)) > 3  # noqa: PLR2004 (default k == 3)
        else 0
    ),
    0,
)
skewness_best_observation = ObservationType(
    "y_skewness_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        np.nan_to_num(skew(costs).item(), nan=0) if len(costs := get_best_percentile_costs(smbo)) > 0 else 0
    ),
    0,
)
kurtosis_best_observation = ObservationType(
    "y_kurtosis_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        np.nan_to_num(kurtosis(costs).item(), nan=0) if len(costs := get_best_percentile_costs(smbo)) > 0 else 0
    ),
    0,
)
mean_best_observation = ObservationType(
    "y_mean_best",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.mean(costs) if len(costs := get_best_percentile_costs(smbo)) > 0 else 0,  # noqa: ARG005
    0,
)
std_best_observation = ObservationType(
    "y_std_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, memory: np.std(costs) if len(costs := get_best_percentile_costs(smbo)) > 0 else -1,  # noqa: ARG005
    -1,
)
variability_best_observation = ObservationType(
    "y_variability_best",
    Box(low=0, high=np.inf, dtype=np.float32),
    lambda smbo, _memory: (
        calc_variability(costs)  # type: ignore[arg-type]
        if len(costs := get_best_percentile_costs(smbo, min_samples=4)) > 3  # noqa: PLR2004
        else -1
    ),
    -1,
)
budget_percentage_observation = ObservationType(
    "budget_percentage",
    Box(low=0, high=1, dtype=np.float32),
    lambda smbo, memory: len(smbo.runhistory) / smbo._scenario.n_trials,  # noqa: ARG005
    0,
)
inc_improvement_scaled_observation = ObservationType(
    "inc_improvement_scaled",
    Box(low=0, high=1, dtype=np.float32),
    lambda smbo, _memory: (
        1 - min(curr, prev) / max(curr, prev)
        if len(t := smbo.intensifier.trajectory) > 1
        and t[-1].trial == len(smbo.runhistory)
        and max(curr := abs(t[-1].costs[-1]), prev := abs(t[-2].costs[-1])) != 0
        else 0
    ),
    0,
)
has_categorical_hps = ObservationType(
    "has_categorical_hps",
    Box(low=0, high=1, dtype=bool),
    lambda smbo, _memory: (
        len([hp for hp in smbo._scenario.configspace.values() if isinstance(hp, CategoricalHyperparameter)]) > 0
    ),
    False,  # noqa: FBT003
)
knn_difference_observation = ObservationType(
    "knn_difference",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: knn_difference(memory=memory),  # type: ignore[arg-type] # noqa: ARG005
    0,
)
ubr_difference_observation = ObservationType(
    "ubr_difference",
    Box(low=-np.inf, high=np.inf, dtype=np.float32),
    lambda smbo, memory: ubr_difference(memory),  # type: ignore[arg-type] # noqa: ARG005
    0,
)

# Must be computed INSIDE DACBOEnv, because observationspace does not have access to action space and last action
previous_param_observation = ObservationType(
    name="previous_param",
    space=Box(low=-np.inf, high=np.inf, dtype=np.float32),
    compute=lambda smbo, memory: None,  # noqa: ARG005
    default=None,
)

rho_observation = ObservationType(
    "rho_B",
    Box(low=0, high=np.inf, dtype=np.float32),
    calculate_budget_density,
    0,
)
d_log_observation = ObservationType(
    "d_eff",
    Box(low=0, high=np.inf, dtype=np.float32),
    calculate_log_effective_dimension,
    0,
)
p_cat_observation = ObservationType(
    "p_cat",
    Box(low=0, high=1, dtype=np.float32),
    calculate_categorical_fraction,
    0,
)
p_int_observation = ObservationType(
    "p_int",
    Box(low=0, high=1, dtype=np.float32),
    calculate_integer_ordinal_fraction,
    0,
)
has_conditionals_observation = ObservationType(
    "has_conditionals",
    Box(low=0, high=1, dtype=np.float32),
    calculate_has_conditionals,
    0,
)
normalized_noise_observation = ObservationType(
    "normalized_noise",
    Box(low=-1, high=np.inf, dtype=np.float32),
    calculate_normalized_noise,
    -1,
)
parameter_age_observation = ObservationType(
    "a_age",
    Box(low=0, high=1, dtype=np.float32),
    calculate_parameter_age,
    0,
)
recent_progress_short_observation = ObservationType(
    "p_ts",
    Box(low=0, high=np.inf, dtype=np.float32),
    calculate_short_progress,
    0,
)
recent_progress_long_observation = ObservationType(
    "p_tl",
    Box(low=0, high=np.inf, dtype=np.float32),
    calculate_long_progress,
    0,
)
stagnation_age_observation = ObservationType(
    "stagnation_age",
    Box(low=0, high=1, dtype=np.float32),
    calculate_stagnation_age,
    0,
)
calibration_error_observation = ObservationType(
    "calibration_error",
    Box(low=-1, high=1, dtype=np.float32),
    calculate_calibration_error,
    -1,
)
global_state_observation = ObservationType(
    "global_state",
    Box(low=GLOBAL_STATE_LOW, high=GLOBAL_STATE_HIGH, dtype=np.float32),
    calculate_global_state,
    GLOBAL_STATE_DEFAULT,
)
action_features_observation = ObservationType(
    "action_features",
    Box(low=ACTION_FEATURE_LOW, high=ACTION_FEATURE_HIGH, dtype=np.float32),
    calculate_action_features,
    ACTION_FEATURE_DEFAULT,
)
af_action_features_observation = ObservationType(
    "af_action_features",
    Box(
        low=AF_ACTION_FEATURE_LOW,
        high=AF_ACTION_FEATURE_HIGH,
        dtype=np.float32,
    ),
    calculate_af_action_features,
    AF_ACTION_FEATURE_DEFAULT,
)


def build_gp_hp_observations(smbo: SMBO) -> list[ObservationType]:
    """Build the GP Hyperparameter Observations.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.

    Returns
    -------
    list[ObservationType]
        A list of the single GP HPs.
    """
    observations = []

    for offset, hp in enumerate_offset(
        smbo._intensifier._config_selector._acquisition_function.model._kernel.hyperparameters
    ):
        if hp.fixed:
            continue

        for i in range(hp.n_elements):
            idx = i + offset

            observations.append(
                ObservationType(
                    name=f"gp_hp_{hp.name}{i}_observation",
                    space=Box(hp.bounds[i][0], hp.bounds[i][1]),
                    compute=lambda smbo_, memory=None, idx=idx: (  # type: ignore[misc] # noqa: ARG005
                        smbo_._intensifier._config_selector._acquisition_function.model._kernel.theta[idx]
                    ),
                    default=0,
                )
            )

    return observations


gp_hp_observation = MultiObservationType(
    "gp_hp_observations",
    build_gp_hp_observations,
)

LEGACY_OBSERVATIONS = [
    incumbent_change_observation,
    trials_passed_observation,
    trials_left_observation,
    # modelfit_observation, # Disabled due to high computation time, behavior similar to UBR
    dimensions_observation,
    continuous_hp_observation,
    categorical_hp_observation,
    ordinal_hp_observation,
    int_hp_observation,
    tsd_observation,
    y_skewness_observation,
    y_kurtosis_observation,
    y_mean_observation,
    std_observation,
    variability_observation,
    tsd_best_observation,
    skewness_best_observation,
    kurtosis_best_observation,
    mean_best_observation,
    std_best_observation,
    variability_best_observation,
    budget_percentage_observation,
    inc_improvement_scaled_observation,
    has_categorical_hps,
    acq_value_ei_observation,
    acq_value_wei_observation,
    acq_value_pi_observation,
    acq_value_wei_explore_observation,
    previous_param_observation,
    ubr_observation,
    ubr_gradient_observation,
    ubr_smoothed_gradient_observation,
    ubr_smoothed_gradient_std_observation,
    ubr_difference_observation,
    knn_entropy_observation,
    knn_entropy_best_observation,
    knn_difference_observation,
]

STRUCTURED_OBSERVATIONS = [
    rho_observation,
    d_log_observation,
    p_cat_observation,
    p_int_observation,
    has_conditionals_observation,
    normalized_noise_observation,
    parameter_age_observation,
    recent_progress_short_observation,
    recent_progress_long_observation,
    stagnation_age_observation,
    calibration_error_observation,
    global_state_observation,
    action_features_observation,
    af_action_features_observation,
]
STRUCTURED_OBSERVATION_NAMES = frozenset(observation.name for observation in STRUCTURED_OBSERVATIONS)

ALL_OBSERVATIONS = LEGACY_OBSERVATIONS + STRUCTURED_OBSERVATIONS

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
        Observation names to include. If None, the legacy default set is used;
        structured observations must be selected explicitly.
    action_space : AbstractActionSpace, optional
        Installed DACBO action controller. Structured observations use it to
        expose the consequences and current value of non-WEI actions.

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

    def __init__(
        self,
        smac_instance: SMBO,
        keys: list[str] | None = None,
        action_space: AbstractActionSpace | None = None,
    ) -> None:
        """Initialize the ObservationSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.
        keys : list[str], optional
            Observation names to include. If None, the legacy default set is
            used; structured observations must be selected explicitly.
        action_space : AbstractActionSpace, optional
            Installed action controller used by acquisition-conditioned
            structured observations.

        Raises
        ------
        ValueError
            If any provided key is invalid.
        """
        self._smac_instance = smac_instance
        self._action_space = action_space

        # Preserve the pre-structured-state default interface. New observations
        # are explicit opt-ins through their keys or the structured YAML group.
        self._keys = (
            keys
            if keys is not None
            else [observation.name for observation in LEGACY_OBSERVATIONS]
            + list(ObservationSpace._MULTI_OBSERVATION_MAP.keys())
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
            copy.copy(ObservationSpace._OBSERVATION_MAP[key])
            for key in self._keys
            if key in ObservationSpace._OBSERVATION_MAP
        ] + [
            copy.copy(space)
            for key in self._keys
            if key in ObservationSpace._MULTI_OBSERVATION_MAP
            for space in ObservationSpace._MULTI_OBSERVATION_MAP[key].create(smac_instance)
        ]
        for obs in self._observation_types:
            if inspect.isclass(obs.compute):
                obs.compute = obs.compute()
        self._configure_action_conditioned_defaults()
        self._observation_space = Dict({obs.name: obs.space for obs in self._observation_types})

        self._register_to_memory: dict[str, Callable] = {}
        self._memory: Memory = {}
        for obs in self._observation_types:
            if obs.name.startswith("ubr"):
                self._register_to_memory["ubr"] = ComputeUBR()
                self._memory["ubr"] = []
            elif obs.name.startswith("knn") and "best" not in obs.name:
                self._register_to_memory["knn"] = calculate_knn
                self._memory["knn"] = []

        selected_keys = set(self._keys)
        self._structured_selected = bool(selected_keys & STRUCTURED_OBSERVATION_NAMES)
        if selected_keys & {"a_age", "global_state"}:
            if self._action_space is not None and hasattr(
                self._action_space,
                "current_control_value",
            ):
                self._register_to_memory["alpha"] = (
                    lambda _smbo: self._action_space.current_control_value  # type: ignore[union-attr]
                )
            else:
                self._register_to_memory["alpha"] = calculate_current_alpha
            self._memory["alpha"] = []
        if selected_keys & {"calibration_error", "global_state"}:
            self._register_to_memory["calibration"] = calculate_prequential_calibration
            self._memory["calibration"] = []

        self._last_observation_trial: int | None = None
        self._cached_observation: ObsType | None = None

    def _configure_action_conditioned_defaults(self) -> None:
        """Bind reset defaults to the installed structured action controller."""
        for obs in self._observation_types:
            if obs.name == "action_features" and isinstance(self._action_space, PosteriorQuantileActionSpace):
                default = np.zeros(
                    (
                        len(self._action_space.quantile_levels),
                        len(ACTION_FEATURE_NAMES),
                    ),
                    dtype=np.float32,
                )
                default[:, ACTION_FEATURE_INDEX["control_value"]] = self._action_space.quantile_levels
                obs.default = default
            elif obs.name in {"action_features", "af_action_features"} and isinstance(
                self._action_space, PosteriorModeActionSpace
            ):
                # New policies use the common ``action_features`` key. Keep the
                # former AF-specific key as an identical compatibility alias for
                # already-saved policy configurations.
                obs.space = Box(
                    low=AF_ACTION_FEATURE_LOW,
                    high=AF_ACTION_FEATURE_HIGH,
                    dtype=np.float32,
                )
                default = AF_ACTION_FEATURE_DEFAULT.copy()
                acquisition_function = self._smac_instance.intensifier.config_selector._acquisition_function
                if isinstance(acquisition_function, PosteriorModeAcquisition):
                    default[
                        POSTERIOR_MODE_NAMES.index(LOWER_CONFIDENCE_BOUND),
                        AF_ACTION_FEATURE_INDEX["parameter_value"],
                    ] = acquisition_function.lower_quantile
                obs.default = default

    @property
    def space(self) -> Space:
        """Returns the Gymnasium Dict space for the selected observations.

        Returns
        -------
        gymnasium.spaces.Dict
            The observation space.
        """
        return self._observation_space

    def get_action_feature_diagnostics(self) -> dict[str, Any]:
        """Summarize selected proxy candidates without exposing them to PPO."""
        if self._cached_observation is None or "action_features" not in self._cached_observation:
            return {}
        features = np.asarray(self._cached_observation["action_features"], dtype=float)
        if features.ndim != ACTION_FEATURE_NDIM or features.shape[0] == 0:
            return {}

        candidates = list(getattr(self._smac_instance, _SELECTED_ACTION_CANDIDATES_ATTRIBUTE, []))
        if len(candidates) != features.shape[0]:
            candidates = [None] * features.shape[0]
        valid_candidates = [candidate for candidate in candidates if candidate is not None]
        candidate_keys = [
            tuple(np.nan_to_num(np.asarray(candidate.get_array(), dtype=float), nan=-1e30).round(12).tolist())
            for candidate in valid_candidates
        ]
        unique_count = len(set(candidate_keys))
        duplicate_fraction = 1.0 - unique_count / len(candidate_keys) if candidate_keys else 0.0

        hyperparameters = list(self._smac_instance._scenario.configspace.values())
        distances = [
            _configuration_distance(left, right, hyperparameters)
            for left_index, left in enumerate(valid_candidates)
            for right in valid_candidates[left_index + 1 :]
        ]
        if features.shape[1] == len(ACTION_FEATURE_NAMES):
            z_values = features[:, ACTION_FEATURE_INDEX["standardized_improvement"]]
            uncertainties = features[:, ACTION_FEATURE_INDEX["normalized_uncertainty"]]
            novelties = features[:, ACTION_FEATURE_INDEX["novelty"]]
            identity_width = 1
        elif features.shape[1] == len(AF_ACTION_FEATURE_NAMES):
            z_values = features[:, AF_ACTION_FEATURE_INDEX["standardized_improvement"]]
            uncertainties = features[:, AF_ACTION_FEATURE_INDEX["normalized_uncertainty"]]
            novelties = features[:, AF_ACTION_FEATURE_INDEX["novelty"]]
            identity_width = len(POSTERIOR_MODE_NAMES)
        else:
            return {}

        uncertainty_ranks = rankdata(uncertainties, method="average")
        action_ranks = np.arange(features.shape[0], dtype=float)
        if np.ptp(uncertainty_ranks) <= np.finfo(float).eps:
            spearman = 0.0
        else:
            spearman = float(np.corrcoef(action_ranks, uncertainty_ranks)[0, 1])
        consequence_rows = features[:, identity_width:]
        zero_rows = np.all(np.isclose(consequence_rows, 0.0), axis=1)
        return {
            "action_features/unique_candidate_count": unique_count,
            "action_features/duplicate_candidate_fraction": duplicate_fraction,
            "action_features/mean_pairwise_candidate_distance": float(np.mean(distances)) if distances else 0.0,
            "action_features/uncertainty_by_action": uncertainties.tolist(),
            "action_features/novelty_by_action": novelties.tolist(),
            "action_features/z_by_action": z_values.tolist(),
            "action_features/spearman_action_vs_uncertainty": spearman,
            "action_features/zero_consequence_row_fraction": float(np.mean(zero_rows)),
        }

    def get_observation(self) -> ObsType:
        """Compute the current observation values from the given optimizer.

        Returns
        -------
        ObsType
            Dictionary mapping observation names to their computed values.
        """
        current_trial = int(self._smac_instance.runhistory.finished)
        if self._last_observation_trial == current_trial and self._cached_observation is not None:
            return {key: value.copy() for key, value in self._cached_observation.items()}

        if self._structured_selected and current_trial < _initial_design_size(self._smac_instance):
            observation = self._default_observation()
            self._cache_observation(current_trial, observation)
            return {key: value.copy() for key, value in observation.items()}

        if "calibration" in self._register_to_memory:
            calibration = self._register_to_memory["calibration"](self._smac_instance)
            self._memory["calibration"].append(calibration)

        for reg_key, compute_function in self._register_to_memory.items():
            if reg_key == "calibration":
                continue
            val = compute_function(self._smac_instance)
            self._memory[reg_key].append(val)

        observation = {
            obs.name: self._compute_observation(obs)
            for obs in self._observation_types
            if obs.name not in STRUCTURED_OBSERVATION_NAMES
        }
        if self._structured_selected:
            _synchronize_model(self._smac_instance)
            observation.update(
                {
                    obs.name: self._compute_observation(obs)
                    for obs in self._observation_types
                    if obs.name in STRUCTURED_OBSERVATION_NAMES
                }
            )

        self._cache_observation(current_trial, observation)
        return {key: value.copy() for key, value in observation.items()}

    def get_initial_observation(self) -> ObsType:
        """Return the reset observation without changing legacy reset semantics.

        Legacy observations retain their historical defaults. Explicitly
        selected structured observations expose the already-consumed initial
        design and the surrogate fitted to it.
        """
        if not self._structured_selected:
            return self._default_observation()

        current_trial = int(self._smac_instance.runhistory.finished)
        if self._last_observation_trial == current_trial and self._cached_observation is not None:
            return {key: value.copy() for key, value in self._cached_observation.items()}

        if current_trial < _initial_design_size(self._smac_instance):
            observation = self._default_observation()
            self._cache_observation(current_trial, observation)
            return {key: value.copy() for key, value in observation.items()}

        if "calibration" in self._register_to_memory:
            calibration = self._register_to_memory["calibration"](self._smac_instance)
            self._memory["calibration"].append(calibration)
        if "alpha" in self._register_to_memory:
            alpha = self._register_to_memory["alpha"](self._smac_instance)
            self._memory["alpha"].append(alpha)

        _synchronize_model(self._smac_instance)
        observation = {
            obs.name: (
                self._compute_observation(obs)
                if obs.name in STRUCTURED_OBSERVATION_NAMES
                else np.atleast_1d(obs.default).astype(np.float32)
            )
            for obs in self._observation_types
        }
        self._cache_observation(current_trial, observation)
        return {key: value.copy() for key, value in observation.items()}

    def _compute_observation(self, observation_type: ObservationType) -> np.ndarray:
        if observation_type.name == "action_features" and isinstance(self._action_space, PosteriorQuantileActionSpace):
            value = calculate_posterior_quantile_action_features(
                self._smac_instance,
                quantile_levels=self._action_space.quantile_levels,
                memory=self._memory,
            )
        elif observation_type.name in {"action_features", "af_action_features"} and isinstance(
            self._action_space, PosteriorModeActionSpace
        ):
            value = calculate_af_action_features(
                self._smac_instance,
                memory=self._memory,
            )
        else:
            value = observation_type.compute(self._smac_instance, self._memory)
        return np.atleast_1d(value).astype(np.float32)

    def _default_observation(self) -> ObsType:
        return {obs.name: np.atleast_1d(obs.default).astype(np.float32) for obs in self._observation_types}

    def _cache_observation(self, trial: int, observation: ObsType) -> None:
        self._last_observation_trial = trial
        self._cached_observation = {key: value.copy() for key, value in observation.items()}

    def reset(self) -> None:
        """Reset any stateful observations. Should be called when the env is reset."""
        for values in self._memory.values():
            values.clear()
        self._last_observation_trial = None
        self._cached_observation = None
        for obs in self._observation_types:
            if isinstance(obs.compute, GetAFandAcqValue):
                obs.compute.reset()
