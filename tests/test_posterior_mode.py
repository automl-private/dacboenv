"""Tests for deterministic posterior-mode acquisition selection."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dacboenv.env.action import PosteriorModeActionSpace
from dacboenv.utils.posterior_decision import (
    EXPECTED_IMPROVEMENT,
    LOWER_CONFIDENCE_BOUND,
    MAXIMUM_VARIANCE,
    POSTERIOR_MEAN,
    POSTERIOR_MODE_NAMES,
    PROBABILITY_IMPROVEMENT,
    PosteriorModeAcquisition,
)
from scipy.stats import norm
from smac.acquisition.function import EI


class PredictionModel:
    """Minimal SMAC-model stand-in returning fixed posterior predictions."""

    def __init__(self, mean: np.ndarray, variance: np.ndarray) -> None:
        self.mean = mean
        self.variance = variance

    def predict_marginalized(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return one posterior row per requested candidate."""
        assert X.shape[0] == self.mean.shape[0]
        return self.mean.copy(), self.variance.copy()


def make_smac(
    acquisition_function: object,
    *,
    previous_entries: int = 7,
) -> SimpleNamespace:
    """Build the selector surface consumed by the action space."""
    selector = SimpleNamespace(
        _acquisition_function=acquisition_function,
        _previous_entries=previous_entries,
    )
    return SimpleNamespace(
        _intensifier=SimpleNamespace(_config_selector=selector),
    )


def test_posterior_modes_match_closed_form_operations() -> None:
    """Each action implements its named posterior decision rule."""
    mean = np.asarray([[-1.0], [0.2], [1.5]])
    variance = np.asarray([[0.04], [1.0], [4.0]])
    eta = 0.5
    acquisition = PosteriorModeAcquisition()
    acquisition.update(
        model=PredictionModel(mean, variance),
        eta=eta,
    )

    scores = acquisition.score_predictions(mean, variance)
    std = np.sqrt(variance)
    improvement = eta - mean
    z = improvement / std

    np.testing.assert_allclose(scores[POSTERIOR_MEAN], -mean)
    np.testing.assert_allclose(
        scores[PROBABILITY_IMPROVEMENT],
        norm.cdf(z),
    )
    np.testing.assert_allclose(
        scores[EXPECTED_IMPROVEMENT],
        improvement * norm.cdf(z) + std * norm.pdf(z),
    )
    np.testing.assert_allclose(
        scores[LOWER_CONFIDENCE_BOUND],
        -(mean + norm.ppf(0.1) * std),
    )
    np.testing.assert_allclose(scores[MAXIMUM_VARIANCE], variance)


@pytest.mark.parametrize(("mode_index", "mode"), list(enumerate(POSTERIOR_MODE_NAMES)))
def test_selected_mode_returns_only_its_own_scores(
    mode_index: int,
    mode: str,
) -> None:
    """SMAC sees the score for the selected operation, never a cross-mode aggregate."""
    mean = np.asarray([[-0.25], [0.5]])
    variance = np.asarray([[0.1], [2.0]])
    model = PredictionModel(mean, variance)
    acquisition = PosteriorModeAcquisition()
    acquisition.update(model=model, eta=0.0)
    smac = make_smac(acquisition)
    action_space = PosteriorModeActionSpace(smac)

    action_space.update_optimizer(mode_index)

    expected = acquisition.score_predictions(mean, variance)[mode]
    actual = acquisition._compute(np.zeros((2, 1)))
    np.testing.assert_allclose(actual, expected)
    assert action_space.selected_mode == mode
    assert action_space.current_action_index == mode_index
    assert action_space.normalized_action == pytest.approx(mode_index / 4)


def test_mode_switch_preserves_acquisition_and_refreshes_selector() -> None:
    """Switching mode keeps one acquisition object and invalidates SMAC's update cache."""
    acquisition = PosteriorModeAcquisition()
    smac = make_smac(acquisition, previous_entries=12)
    selector = smac._intensifier._config_selector
    action_space = PosteriorModeActionSpace(smac)

    assert action_space.mode_names == POSTERIOR_MODE_NAMES
    assert action_space.current_action_index == 2
    assert action_space.normalized_action == pytest.approx(0.5)
    assert acquisition._mode_names == POSTERIOR_MODE_NAMES
    assert acquisition.selected_mode == EXPECTED_IMPROVEMENT
    assert acquisition.current_action_index == 2
    assert acquisition.current_control_value == pytest.approx(0.5)

    action_space.update_optimizer(0)

    assert selector._acquisition_function is acquisition
    assert acquisition.mode == POSTERIOR_MEAN
    assert selector._previous_entries == -1

    # Reapplying the already active operation does not cause a needless refit.
    selector._previous_entries = 12
    action_space.update_optimizer(0)
    assert selector._previous_entries == 12


def test_best_candidate_indices_rank_within_each_mode() -> None:
    """Action consequences are based on within-mode rankings only."""
    mean = np.asarray([[-2.0], [0.0], [1.0]])
    variance = np.asarray([[0.01], [9.0], [1.0]])
    acquisition = PosteriorModeAcquisition()
    acquisition.update(model=PredictionModel(mean, variance), eta=-1.0)

    best = acquisition.best_candidate_indices(mean, variance)

    assert set(best) == set(POSTERIOR_MODE_NAMES)
    assert best[POSTERIOR_MEAN] == 0
    assert best[MAXIMUM_VARIANCE] == 1
    for mode, scores in acquisition.score_predictions(mean, variance).items():
        assert best[mode] == int(np.argmax(scores.reshape(-1)))


def test_zero_variance_rules_are_finite_and_deterministic() -> None:
    """Degenerate posterior predictions do not introduce NaNs."""
    mean = np.asarray([[-1.0], [0.0], [1.0]])
    variance = np.zeros_like(mean)
    acquisition = PosteriorModeAcquisition()
    acquisition.update(model=PredictionModel(mean, variance), eta=0.0)

    scores = acquisition.score_predictions(mean, variance)

    assert all(np.isfinite(values).all() for values in scores.values())
    np.testing.assert_array_equal(
        scores[PROBABILITY_IMPROVEMENT].reshape(-1),
        np.asarray([1.0, 0.0, 0.0]),
    )
    np.testing.assert_array_equal(
        scores[EXPECTED_IMPROVEMENT].reshape(-1),
        np.asarray([1.0, 0.0, 0.0]),
    )


@pytest.mark.parametrize("lower_quantile", [0.0, 0.5, 1.0, np.nan])
def test_lower_confidence_quantile_is_strictly_lower(
    lower_quantile: float,
) -> None:
    """Selection-mode LCB accepts only proper lower posterior quantiles."""
    with pytest.raises(ValueError, match="lower_quantile"):
        PosteriorModeAcquisition(lower_quantile=lower_quantile)


def test_action_space_rejects_invalid_setup_and_actions() -> None:
    """Configuration errors fail before a training rollout starts."""
    with pytest.raises(TypeError, match="requires PosteriorModeAcquisition"):
        PosteriorModeActionSpace(make_smac(EI()))

    action_space = PosteriorModeActionSpace(make_smac(PosteriorModeAcquisition()))
    with pytest.raises(ValueError, match="outside"):
        action_space.update_optimizer(5)


def test_acquisition_requires_eta_and_matching_prediction_shapes() -> None:
    """Malformed SMAC updates and posterior outputs fail clearly."""
    acquisition = PosteriorModeAcquisition()
    with pytest.raises(ValueError, match="eta"):
        acquisition.score_predictions(np.zeros((1, 1)), np.ones((1, 1)))

    acquisition._eta = 0.0
    with pytest.raises(ValueError, match="same shape"):
        acquisition.score_predictions(np.zeros((2, 1)), np.ones((2,)))
