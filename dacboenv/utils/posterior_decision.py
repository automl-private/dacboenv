"""Posterior decision modes for acquisition-function selection.

The modes in this module share one surrogate model and one incumbent value.
They are alternatives for choosing the next candidate, not quantities whose
raw values should be compared across modes.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from scipy.stats import norm
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)

POSTERIOR_MEAN: Final = "posterior_mean"
PROBABILITY_IMPROVEMENT: Final = "probability_improvement"
EXPECTED_IMPROVEMENT: Final = "expected_improvement"
LOWER_CONFIDENCE_BOUND: Final = "lower_confidence_bound"
MAXIMUM_VARIANCE: Final = "maximum_variance"

POSTERIOR_MODE_NAMES: Final[tuple[str, ...]] = (
    POSTERIOR_MEAN,
    PROBABILITY_IMPROVEMENT,
    EXPECTED_IMPROVEMENT,
    LOWER_CONFIDENCE_BOUND,
    MAXIMUM_VARIANCE,
)
MEDIAN_QUANTILE: Final = 0.5


class PosteriorModeAcquisition(AbstractAcquisitionFunction):
    """Select candidates using one of five deterministic posterior operations.

    The operations all assume minimization:

    - posterior mean maximizes ``-mean``;
    - PI and EI use a fixed improvement offset ``xi``;
    - the lower confidence bound minimizes the configured posterior quantile;
    - maximum variance maximizes predictive variance.

    Parameters
    ----------
    mode : str, optional
        Initial decision mode. Defaults to expected improvement, matching
        SMAC's native black-box facade.
    xi : float, optional
        Fixed PI/EI improvement offset. Defaults to zero.
    lower_quantile : float, optional
        Posterior quantile used by the lower-confidence-bound mode. It must be
        strictly between zero and one half. Defaults to ``0.1``.
    """

    def __init__(
        self,
        mode: str = EXPECTED_IMPROVEMENT,
        xi: float = 0.0,
        lower_quantile: float = 0.1,
    ) -> None:
        super().__init__()
        if not np.isfinite(xi):
            raise ValueError(f"xi must be finite, got {xi}.")
        if not 0.0 < lower_quantile < MEDIAN_QUANTILE:
            raise ValueError(
                "lower_quantile must be strictly between 0 and 0.5, "
                f"got {lower_quantile}."
            )

        self._mode = self._validate_mode(mode)
        self._mode_names = POSTERIOR_MODE_NAMES
        self._xi = float(xi)
        self._lower_quantile = float(lower_quantile)
        self._eta: float | None = None

    @staticmethod
    def _validate_mode(mode: str) -> str:
        if mode not in POSTERIOR_MODE_NAMES:
            raise ValueError(
                f"Unknown posterior decision mode {mode!r}; "
                f"expected one of {POSTERIOR_MODE_NAMES}."
            )
        return mode

    @property
    def name(self) -> str:
        """Return a stable name independent of the currently selected mode."""
        return "Posterior Decision Mode"

    @property
    def meta(self) -> dict[str, Any]:
        """Return acquisition metadata, including the active operation."""
        meta = super().meta
        meta.update(
            {
                "mode": self._mode,
                "xi": self._xi,
                "lower_quantile": self._lower_quantile,
                "mode_names": POSTERIOR_MODE_NAMES,
            }
        )
        return meta

    @property
    def mode(self) -> str:
        """Return the active posterior decision mode."""
        return self._mode

    @mode.setter
    def mode(self, mode: str) -> None:
        """Select a posterior decision mode."""
        self._mode = self._validate_mode(mode)

    @property
    def mode_names(self) -> tuple[str, ...]:
        """Return action-index order for the five posterior modes."""
        return self._mode_names

    @property
    def selected_mode(self) -> str:
        """Return the active posterior operation for observation and logging."""
        return self._mode

    @property
    def current_action_index(self) -> int:
        """Return the active operation's categorical action index."""
        return self._mode_names.index(self._mode)

    @property
    def normalized_action(self) -> float:
        """Return the active action index normalized to ``[0, 1]``."""
        return self.current_action_index / (len(self._mode_names) - 1)

    @property
    def current_control_value(self) -> float:
        """Return the normalized operation index for generic observations."""
        return self.normalized_action

    @property
    def lower_quantile(self) -> float:
        """Return the quantile used by lower-confidence-bound decisions."""
        return self._lower_quantile

    def _update(self, **kwargs: Any) -> None:
        if "eta" not in kwargs:
            raise ValueError("Posterior decision modes require the current incumbent value `eta`.")
        self._eta = float(np.asarray(kwargs["eta"]).item())

    def score_predictions(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Score posterior predictions independently under every mode.

        The returned values have different units and ranges. Consumers may
        maximize values within each dictionary entry, but must not compare raw
        values between entries. Use :meth:`best_candidate_indices` when only
        the consequence of each action is needed.
        """
        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update(eta=<float>) before scoring."
            )

        mean_array = np.asarray(mean, dtype=float)
        variance_array = np.asarray(variance, dtype=float)
        if mean_array.shape != variance_array.shape:
            raise ValueError(
                "mean and variance must have the same shape, "
                f"got {mean_array.shape} and {variance_array.shape}."
            )

        variance_array = np.maximum(variance_array, 0.0)
        std = np.sqrt(variance_array)
        improvement = self._eta - mean_array - self._xi
        nonzero_std = std > 0.0

        z = np.zeros_like(improvement)
        np.divide(improvement, std, out=z, where=nonzero_std)

        probability_improvement = np.empty_like(improvement)
        probability_improvement[nonzero_std] = norm.cdf(z[nonzero_std])
        probability_improvement[~nonzero_std] = (
            improvement[~nonzero_std] > 0.0
        ).astype(float)

        expected_improvement = np.maximum(improvement, 0.0)
        expected_improvement[nonzero_std] = (
            improvement[nonzero_std] * norm.cdf(z[nonzero_std])
            + std[nonzero_std] * norm.pdf(z[nonzero_std])
        )

        quantile_z = float(norm.ppf(self._lower_quantile))
        return {
            POSTERIOR_MEAN: -mean_array,
            PROBABILITY_IMPROVEMENT: probability_improvement,
            EXPECTED_IMPROVEMENT: expected_improvement,
            LOWER_CONFIDENCE_BOUND: -(mean_array + quantile_z * std),
            MAXIMUM_VARIANCE: variance_array,
        }

    def best_candidate_indices(
        self,
        mean: np.ndarray,
        variance: np.ndarray,
    ) -> dict[str, int]:
        """Return each operation's best candidate without cross-mode comparison."""
        scores = self.score_predictions(mean, variance)
        if np.asarray(mean).size == 0:
            return {}

        best: dict[str, int] = {}
        for mode in POSTERIOR_MODE_NAMES:
            mode_scores = np.asarray(scores[mode], dtype=float).reshape(-1)
            finite_scores = np.where(np.isfinite(mode_scores), mode_scores, -np.inf)
            best[mode] = int(np.argmax(finite_scores))
        return best

    def _compute(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        if self._model is None:
            raise ValueError("Posterior decision modes require a fitted surrogate model.")

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        mean, variance = self._model.predict_marginalized(X)
        scores = self.score_predictions(mean, variance)
        return np.asarray(scores[self._mode], dtype=float).reshape(-1, 1)
