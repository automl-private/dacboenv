"""Fixed-shape, side-effect-free Gaussian-process hyperparameter features.

The compatibility layer in this module is the only policy-observation code
that accesses SMAC's private model attributes.  SMAC 2.4.0 exposes the active
surrogate through ``SMBO.intensifier.config_selector._model``.  Its Gaussian
process wrapper stores the configured kernel in ``_kernel`` and the fitted
scikit-learn kernel in ``_gp.kernel_``.  Reading these attributes does not fit
the model, advance the runhistory, or consume randomness.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from gymnasium.spaces import Box

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

GP_HP_SUMMARY_NAMES = (
    "available",
    "is_gp",
    "uses_ard",
    "lengthscale_available",
    "signal_available",
    "noise_available",
    "n_free_parameters_scaled",
    "n_lengthscales_scaled",
    "lengthscale_mean",
    "lengthscale_std",
    "lengthscale_min",
    "lengthscale_max",
    "lengthscale_q10",
    "lengthscale_median",
    "lengthscale_q90",
    "lengthscale_anisotropy",
    "lengthscale_effective_dimension_fraction",
    "lengthscale_lower_bound_fraction",
    "lengthscale_upper_bound_fraction",
    "signal_level",
    "noise_level",
    "noise_minus_signal",
    "all_parameter_lower_bound_fraction",
    "all_parameter_upper_bound_fraction",
)
GP_HP_SUMMARY_INDEX = {name: index for index, name in enumerate(GP_HP_SUMMARY_NAMES)}
GP_HP_SUMMARY_DIM = len(GP_HP_SUMMARY_NAMES)

GP_HP_CHANGE_NAMES = (
    "previous_available",
    "theta_delta_l2",
    "theta_delta_max",
    "lengthscale_mean_delta",
    "lengthscale_anisotropy_delta",
    "effective_dimension_delta",
    "signal_delta",
    "noise_delta",
)
GP_HP_CHANGE_INDEX = {name: index for index, name in enumerate(GP_HP_CHANGE_NAMES)}
GP_HP_CHANGE_DIM = len(GP_HP_CHANGE_NAMES)

GP_HP_ROLE_NAMES = ("lengthscale", "signal", "noise", "other")
GP_HP_ROLE_INDEX = {name: index for index, name in enumerate(GP_HP_ROLE_NAMES)}

GP_HP_OBSERVATION_NAMES = frozenset({"gp_hp_summary", "gp_hp_change", "gp_hp_raw", "gp_hp_raw_mask", "gp_hp_raw_roles"})

GP_HP_SUMMARY_LOW = np.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0],
    dtype=np.float32,
)
GP_HP_SUMMARY_HIGH = np.ones(GP_HP_SUMMARY_DIM, dtype=np.float32)
GP_HP_CHANGE_LOW = np.asarray([0, 0, 0, -1, -1, -1, -1, -1], dtype=np.float32)
GP_HP_CHANGE_HIGH = np.ones(GP_HP_CHANGE_DIM, dtype=np.float32)

Role = Literal["lengthscale", "signal", "noise", "other"]
MAX_NEAR_BOUND_FRACTION = 0.5


@dataclass(frozen=True)
class GPHyperparameterSettings:
    """Configuration for fixed-shape GP observations."""

    enabled: bool = False
    max_raw_parameters: int = 64
    overflow_policy: Literal["error", "truncate"] = "error"
    near_bound_fraction: float = 0.05
    unsupported_model_policy: Literal["zeros", "error"] = "zeros"
    strict_kernel_validation: bool = False

    @classmethod
    def from_mapping(cls, values: Any | None) -> GPHyperparameterSettings:
        """Build validated settings from a Hydra mapping or plain dict."""
        if values is None:
            return cls()
        plain = dict(values)
        settings = cls(**plain)
        if settings.max_raw_parameters <= 0:
            raise ValueError("max_raw_parameters must be positive.")
        if settings.overflow_policy not in {"error", "truncate"}:
            raise ValueError("overflow_policy must be 'error' or 'truncate'.")
        if settings.unsupported_model_policy not in {"zeros", "error"}:
            raise ValueError("unsupported_model_policy must be 'zeros' or 'error'.")
        if not 0.0 <= settings.near_bound_fraction <= MAX_NEAR_BOUND_FRACTION:
            raise ValueError("near_bound_fraction must lie in [0, 0.5].")
        return settings


@dataclass(frozen=True)
class ScalarKernelParameter:
    """One free scalar in the stable ``kernel.theta`` order."""

    index: int
    name: str
    element_index: int
    theta: float
    lower_bound: float
    upper_bound: float
    normalized: float
    bound_available: bool
    clipped: bool
    role: Role


@dataclass(frozen=True)
class GPHyperparameterFeatureBundle:
    """One immutable set of all policy-visible GP features."""

    summary: np.ndarray
    change: np.ndarray
    raw: np.ndarray
    raw_mask: np.ndarray
    raw_roles: np.ndarray
    parameters: tuple[ScalarKernelParameter, ...] = ()
    state_key: tuple[int, str] | None = None


@dataclass
class GPHyperparameterDiagnostics:
    """Non-policy-visible extraction diagnostics."""

    extraction_calls: int = 0
    cache_hits: int = 0
    extraction_failures: int = 0
    non_gp_states: int = 0
    unfitted_gp_states: int = 0
    fallback_bound_count: int = 0
    clipping_count: int = 0
    truncation_count: int = 0
    original_parameter_count: int = 0
    role_counts: dict[str, int] = field(default_factory=lambda: dict.fromkeys(GP_HP_ROLE_NAMES, 0))


def classify_kernel_parameter_role(name: str) -> Role:
    """Classify a kernel parameter conservatively from its normalized name."""
    normalized = "".join(character for character in name.lower() if character.isalnum() or character == "_")
    compact = normalized.replace("_", "")
    if "lengthscale" in compact:
        return "lengthscale"
    if any(token in compact for token in ("noiselevel", "noise", "white")):
        return "noise"
    if any(token in compact for token in ("constantvalue", "variance", "amplitude", "outputscale", "signal")):
        return "signal"
    return "other"


def _normalize_theta(theta: float, lower: float, upper: float) -> tuple[float, bool, bool]:
    """Normalize one log-theta value and report bound/final clipping metadata."""
    bound_available = bool(np.isfinite(lower) and np.isfinite(upper) and upper > lower)
    if bound_available:
        unbounded = 2.0 * (theta - lower) / (upper - lower) - 1.0
    else:
        unbounded = float(np.tanh(theta / 4.0)) if np.isfinite(theta) else 0.0
    clipped = bool(np.isfinite(unbounded) and not -1.0 <= unbounded <= 1.0)
    normalized = float(np.clip(unbounded, -1.0, 1.0)) if np.isfinite(unbounded) else 0.0
    return normalized, bound_available, clipped


def extract_free_kernel_parameters(kernel: Any) -> tuple[ScalarKernelParameter, ...]:
    """Extract all nonfixed scalar kernel parameters in ``theta`` order."""
    theta = np.asarray(kernel.theta, dtype=float).reshape(-1)
    bounds = np.asarray(kernel.bounds, dtype=float)
    if bounds.size == 0:
        bounds = np.empty((0, 2), dtype=float)
    bounds = bounds.reshape(-1, 2)
    extracted: list[ScalarKernelParameter] = []
    scalar_index = 0
    for hyperparameter in kernel.hyperparameters:
        if bool(hyperparameter.fixed):
            continue
        n_elements = int(hyperparameter.n_elements)
        role = classify_kernel_parameter_role(str(hyperparameter.name))
        for element_index in range(n_elements):
            if scalar_index >= len(theta) or scalar_index >= len(bounds):
                raise ValueError("Kernel metadata ended before kernel.theta.")
            lower, upper = (float(value) for value in bounds[scalar_index])
            normalized, bound_available, clipped = _normalize_theta(float(theta[scalar_index]), lower, upper)
            extracted.append(
                ScalarKernelParameter(
                    index=scalar_index,
                    name=str(hyperparameter.name),
                    element_index=element_index,
                    theta=float(theta[scalar_index]),
                    lower_bound=lower,
                    upper_bound=upper,
                    normalized=normalized,
                    bound_available=bound_available,
                    clipped=clipped,
                    role=role,
                )
            )
            scalar_index += 1
    if len(theta) != scalar_index or len(bounds) != scalar_index:
        raise ValueError(
            "Kernel free-parameter invariant failed: "
            f"theta={len(theta)}, extracted={scalar_index}, bounds={len(bounds)}."
        )
    return tuple(extracted)


def _scaled_count(count: int) -> float:
    return float(np.tanh(np.log1p(count) / 4.0))


def _effective_dimension_fraction(theta: np.ndarray) -> float:
    if theta.size == 0:
        return 0.0
    log_weights = -2.0 * theta
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    denominator = float(np.square(weights).sum())
    if denominator <= 0.0:
        return 0.0
    effective = float(np.square(weights.sum()) / denominator)
    return float(np.clip(effective / theta.size, 0.0, 1.0))


def summarize_parameters(parameters: tuple[ScalarKernelParameter, ...], near_bound_fraction: float) -> np.ndarray:
    """Create the fixed 24-value GP summary."""
    result = np.zeros(GP_HP_SUMMARY_DIM, dtype=np.float32)
    result[GP_HP_SUMMARY_INDEX["available"]] = 1.0
    result[GP_HP_SUMMARY_INDEX["is_gp"]] = 1.0
    result[GP_HP_SUMMARY_INDEX["n_free_parameters_scaled"]] = _scaled_count(len(parameters))
    by_role = {
        role: tuple(parameter for parameter in parameters if parameter.role == role) for role in GP_HP_ROLE_NAMES
    }
    lengthscales = by_role["lengthscale"]
    lengthscale_group_sizes: dict[str, int] = {}
    for parameter in lengthscales:
        lengthscale_group_sizes[parameter.name] = lengthscale_group_sizes.get(parameter.name, 0) + 1
    result[GP_HP_SUMMARY_INDEX["uses_ard"]] = float(any(size > 1 for size in lengthscale_group_sizes.values()))
    result[GP_HP_SUMMARY_INDEX["n_lengthscales_scaled"]] = _scaled_count(len(lengthscales))
    lower_threshold = -1.0 + 2.0 * near_bound_fraction
    upper_threshold = 1.0 - 2.0 * near_bound_fraction

    if lengthscales:
        normalized = np.asarray([parameter.normalized for parameter in lengthscales], dtype=float)
        physical_theta = np.asarray([parameter.theta for parameter in lengthscales], dtype=float)
        q10, median, q90 = np.quantile(normalized, [0.1, 0.5, 0.9])
        assignments = {
            "lengthscale_available": 1.0,
            "lengthscale_mean": np.mean(normalized),
            "lengthscale_std": np.std(normalized),
            "lengthscale_min": np.min(normalized),
            "lengthscale_max": np.max(normalized),
            "lengthscale_q10": q10,
            "lengthscale_median": median,
            "lengthscale_q90": q90,
            "lengthscale_anisotropy": np.clip((q90 - q10) / 2.0, 0.0, 1.0),
            "lengthscale_effective_dimension_fraction": _effective_dimension_fraction(physical_theta),
            "lengthscale_lower_bound_fraction": np.mean(normalized <= lower_threshold),
            "lengthscale_upper_bound_fraction": np.mean(normalized >= upper_threshold),
        }
        for name, value in assignments.items():
            result[GP_HP_SUMMARY_INDEX[name]] = float(value)

    role_levels: dict[str, float] = {}
    for role in ("signal", "noise"):
        role_parameters = by_role[role]
        if role_parameters:
            role_levels[role] = float(np.mean([parameter.normalized for parameter in role_parameters]))
            result[GP_HP_SUMMARY_INDEX[f"{role}_available"]] = 1.0
            result[GP_HP_SUMMARY_INDEX[f"{role}_level"]] = role_levels[role]
    if "noise" in role_levels and "signal" in role_levels:
        result[GP_HP_SUMMARY_INDEX["noise_minus_signal"]] = (role_levels["noise"] - role_levels["signal"]) / 2.0
    if parameters:
        normalized = np.asarray([parameter.normalized for parameter in parameters], dtype=float)
        result[GP_HP_SUMMARY_INDEX["all_parameter_lower_bound_fraction"]] = float(
            np.mean(normalized <= lower_threshold)
        )
        result[GP_HP_SUMMARY_INDEX["all_parameter_upper_bound_fraction"]] = float(
            np.mean(normalized >= upper_threshold)
        )
    return result


def _change_features(
    current: tuple[ScalarKernelParameter, ...],
    current_summary: np.ndarray,
    previous: tuple[ScalarKernelParameter, ...] | None,
    previous_summary: np.ndarray | None,
) -> np.ndarray:
    result = np.zeros(GP_HP_CHANGE_DIM, dtype=np.float32)
    if previous is None or previous_summary is None:
        return result
    current_identity = tuple((parameter.name, parameter.element_index) for parameter in current)
    previous_identity = tuple((parameter.name, parameter.element_index) for parameter in previous)
    if current_identity != previous_identity:
        return result
    result[GP_HP_CHANGE_INDEX["previous_available"]] = 1.0
    delta = np.asarray([parameter.normalized for parameter in current]) - np.asarray(
        [parameter.normalized for parameter in previous]
    )
    result[GP_HP_CHANGE_INDEX["theta_delta_l2"]] = float(np.clip(np.sqrt(np.mean(np.square(delta))) / 2.0, 0, 1))
    result[GP_HP_CHANGE_INDEX["theta_delta_max"]] = float(np.clip(np.max(np.abs(delta)) / 2.0, 0, 1))
    pairs = (
        ("lengthscale_mean_delta", "lengthscale_mean", 2.0),
        ("lengthscale_anisotropy_delta", "lengthscale_anisotropy", 1.0),
        ("effective_dimension_delta", "lengthscale_effective_dimension_fraction", 1.0),
        ("signal_delta", "signal_level", 2.0),
        ("noise_delta", "noise_level", 2.0),
    )
    for output_name, summary_name, scale in pairs:
        result[GP_HP_CHANGE_INDEX[output_name]] = float(
            np.clip(
                (
                    current_summary[GP_HP_SUMMARY_INDEX[summary_name]]
                    - previous_summary[GP_HP_SUMMARY_INDEX[summary_name]]
                )
                / scale,
                -1.0,
                1.0,
            )
        )
    return result


class GPHyperparameterFeatureProvider:
    """Inspect one SMAC surrogate and provide cached fixed-shape features."""

    def __init__(self, settings: GPHyperparameterSettings | None = None) -> None:
        self.settings = settings or GPHyperparameterSettings()
        self.diagnostics = GPHyperparameterDiagnostics()
        self._last_state_key: tuple[int, str] | None = None
        self._last_parameters: tuple[ScalarKernelParameter, ...] | None = None
        self._last_summary: np.ndarray | None = None
        self._cached_bundle: GPHyperparameterFeatureBundle | None = None

    def reset(self) -> None:
        """Clear episode-local delta and extraction-cache state."""
        self._last_state_key = None
        self._last_parameters = None
        self._last_summary = None
        self._cached_bundle = None

    def _zeros(self, *, is_gp: bool = False) -> GPHyperparameterFeatureBundle:
        summary = np.zeros(GP_HP_SUMMARY_DIM, dtype=np.float32)
        summary[GP_HP_SUMMARY_INDEX["is_gp"]] = float(is_gp)
        maximum = self.settings.max_raw_parameters
        return GPHyperparameterFeatureBundle(
            summary=summary,
            change=np.zeros(GP_HP_CHANGE_DIM, dtype=np.float32),
            raw=np.zeros(maximum, dtype=np.float32),
            raw_mask=np.zeros(maximum, dtype=np.float32),
            raw_roles=np.zeros((maximum, len(GP_HP_ROLE_NAMES)), dtype=np.float32),
        )

    @staticmethod
    def _locate_model(smbo: SMBO) -> Any | None:
        return getattr(getattr(smbo.intensifier, "config_selector", None), "_model", None)

    @staticmethod
    def _is_gp_model(model: Any) -> bool:
        module = type(model).__module__.lower()
        name = type(model).__name__.lower()
        return "gaussian_process" in module or "gaussianprocess" in name or hasattr(model, "_gp")

    @staticmethod
    def _locate_kernel(model: Any, *, fitted: bool) -> Any | None:
        gp = getattr(model, "_gp", None)
        fitted_kernel = getattr(gp, "kernel_", None)
        if fitted and fitted_kernel is not None:
            return fitted_kernel
        return getattr(model, "_kernel", None) or getattr(gp, "kernel", None)

    def _handle_malformed(self, message: str, error: Exception | None = None) -> GPHyperparameterFeatureBundle:
        self.diagnostics.extraction_failures += 1
        if self.settings.strict_kernel_validation or self.settings.unsupported_model_policy == "error":
            raise RuntimeError(message) from error
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        return self._zeros(is_gp=True)

    def features(self, smbo: SMBO) -> GPHyperparameterFeatureBundle:
        """Return one immutable bundle without fitting or modifying the model."""
        self.diagnostics.extraction_calls += 1
        model = self._locate_model(smbo)
        if model is None or not self._is_gp_model(model):
            self.diagnostics.non_gp_states += 1
            return self._zeros()
        fitted = bool(getattr(model, "_is_trained", False))
        if not fitted:
            self.diagnostics.unfitted_gp_states += 1
            return self._zeros(is_gp=True)
        kernel = self._locate_kernel(model, fitted=True)
        if kernel is None or not all(
            hasattr(kernel, attribute) for attribute in ("theta", "bounds", "hyperparameters")
        ):
            return self._handle_malformed("Fitted GP kernel metadata is unavailable or unsupported.")
        try:
            parameters = extract_free_kernel_parameters(kernel)
        except (TypeError, ValueError, AttributeError) as error:
            return self._handle_malformed(f"Malformed fitted GP kernel metadata: {error}", error)
        theta_bytes = np.asarray([parameter.theta for parameter in parameters], dtype=np.float64).tobytes()
        fingerprint = hashlib.sha256(theta_bytes).hexdigest()
        state_key = (int(smbo.runhistory.finished), fingerprint)
        if self._cached_bundle is not None and self._cached_bundle.state_key == state_key:
            self.diagnostics.cache_hits += 1
            return self._cached_bundle

        summary = summarize_parameters(parameters, self.settings.near_bound_fraction)
        change = _change_features(parameters, summary, self._last_parameters, self._last_summary)
        maximum = self.settings.max_raw_parameters
        original_count = len(parameters)
        self.diagnostics.original_parameter_count = max(self.diagnostics.original_parameter_count, original_count)
        if original_count > maximum:
            if self.settings.overflow_policy == "error":
                raise OverflowError(
                    f"GP has {original_count} free scalar hyperparameters, exceeding configured maximum {maximum}."
                )
            warnings.warn(
                f"Truncating {original_count} GP hyperparameters to the stable first {maximum} entries.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.diagnostics.truncation_count += original_count - maximum
        visible = parameters[:maximum]
        raw = np.zeros(maximum, dtype=np.float32)
        mask = np.zeros(maximum, dtype=np.float32)
        roles = np.zeros((maximum, len(GP_HP_ROLE_NAMES)), dtype=np.float32)
        for index, parameter in enumerate(visible):
            raw[index] = parameter.normalized
            mask[index] = 1.0
            roles[index, GP_HP_ROLE_INDEX[parameter.role]] = 1.0
        self.diagnostics.fallback_bound_count += sum(not parameter.bound_available for parameter in parameters)
        self.diagnostics.clipping_count += sum(parameter.clipped for parameter in parameters)
        for parameter in parameters:
            self.diagnostics.role_counts[parameter.role] += 1
        bundle = GPHyperparameterFeatureBundle(
            summary=summary,
            change=change,
            raw=raw,
            raw_mask=mask,
            raw_roles=roles,
            parameters=parameters,
            state_key=state_key,
        )
        self._last_state_key = state_key
        self._last_parameters = parameters
        self._last_summary = summary.copy()
        self._cached_bundle = bundle
        return bundle

    def observation_spaces(self) -> dict[str, Box]:
        """Return Gymnasium spaces for every selectable GP key."""
        maximum = self.settings.max_raw_parameters
        return {
            "gp_hp_summary": Box(low=GP_HP_SUMMARY_LOW, high=GP_HP_SUMMARY_HIGH, dtype=np.float32),
            "gp_hp_change": Box(low=GP_HP_CHANGE_LOW, high=GP_HP_CHANGE_HIGH, dtype=np.float32),
            "gp_hp_raw": Box(low=-1.0, high=1.0, shape=(maximum,), dtype=np.float32),
            "gp_hp_raw_mask": Box(low=0.0, high=1.0, shape=(maximum,), dtype=np.float32),
            "gp_hp_raw_roles": Box(
                low=0.0,
                high=1.0,
                shape=(maximum, len(GP_HP_ROLE_NAMES)),
                dtype=np.float32,
            ),
        }

    @staticmethod
    def value(bundle: GPHyperparameterFeatureBundle, key: str) -> np.ndarray:
        """Select one ordinary observation key from a shared bundle."""
        attribute = key.removeprefix("gp_hp_")
        return np.asarray(getattr(bundle, attribute), dtype=np.float32)


__all__ = [
    "GP_HP_CHANGE_DIM",
    "GP_HP_CHANGE_INDEX",
    "GP_HP_CHANGE_NAMES",
    "GP_HP_OBSERVATION_NAMES",
    "GP_HP_ROLE_INDEX",
    "GP_HP_ROLE_NAMES",
    "GP_HP_SUMMARY_DIM",
    "GP_HP_SUMMARY_INDEX",
    "GP_HP_SUMMARY_NAMES",
    "GPHyperparameterFeatureBundle",
    "GPHyperparameterFeatureProvider",
    "GPHyperparameterSettings",
    "ScalarKernelParameter",
    "classify_kernel_parameter_role",
    "extract_free_kernel_parameters",
    "summarize_parameters",
]
