from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dacboenv.env.action import AcqParameterActionSpace, PosteriorQuantileActionSpace
from dacboenv.utils.confidence_bound import LCB, UCB
from gymnasium.spaces import Discrete
from scipy.stats import norm
from smac.acquisition.function import EI


def _smbo_with(acquisition_function):
    selector = SimpleNamespace(_acquisition_function=acquisition_function)
    return SimpleNamespace(_intensifier=SimpleNamespace(_config_selector=selector))


@pytest.mark.parametrize(
    ("acquisition_function", "expected_quantiles"),
    [
        (LCB(update_beta=False), (0.5, 0.25, 0.1, 0.025, 0.005)),
        (UCB(update_beta=False), (0.5, 0.75, 0.9, 0.975, 0.995)),
    ],
)
def test_confidence_bound_quantile_defaults_are_five_categorical_actions(
    acquisition_function,
    expected_quantiles,
) -> None:
    action_space = PosteriorQuantileActionSpace(_smbo_with(acquisition_function))

    assert isinstance(action_space.space, Discrete)
    assert action_space.space.n == 5
    assert action_space.quantile_levels == expected_quantiles
    assert action_space._param_levels == list(expected_quantiles)
    assert action_space.bound_type == acquisition_function.bound_type
    assert action_space.current_quantile == pytest.approx(action_space.current_control_value)


@pytest.mark.parametrize(
    ("acquisition_function", "quantile"),
    [
        (LCB(update_beta=False), 0.1),
        (UCB(update_beta=False), 0.9),
    ],
)
def test_quantile_action_maps_to_equivalent_confidence_beta(
    acquisition_function,
    quantile,
) -> None:
    action_space = PosteriorQuantileActionSpace(
        _smbo_with(acquisition_function),
        quantile_levels=[quantile],
    )

    action_space.update_optimizer(0)

    expected_beta = norm.ppf(quantile) ** 2 / acquisition_function._nu
    assert acquisition_function._beta == pytest.approx(expected_beta)
    assert acquisition_function._posterior_quantile == pytest.approx(quantile)
    assert action_space.current_quantile == pytest.approx(quantile)


@pytest.mark.parametrize(
    ("acquisition_function", "quantile", "expected"),
    [
        (LCB(update_beta=False), 0.1, -(10.0 + norm.ppf(0.1) * 2.0)),
        (UCB(update_beta=False), 0.9, -(10.0 + norm.ppf(0.9) * 2.0)),
    ],
)
def test_quantile_action_selects_the_requested_posterior_quantile(
    acquisition_function,
    quantile,
    expected,
) -> None:
    action_space = PosteriorQuantileActionSpace(
        _smbo_with(acquisition_function),
        quantile_levels=[quantile],
    )
    action_space.update_optimizer(0)
    acquisition_function._model = SimpleNamespace(
        predict_marginalized=lambda _x: (np.array([[10.0]]), np.array([[4.0]])),
    )
    acquisition_function._num_data = 1

    value = acquisition_function._compute(np.array([[0.0]]))

    assert value.item() == pytest.approx(expected)


def _two_point_action_margins(
    acquisition_function: LCB | UCB,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Return choices, uncertain-point margins, and beta in action order."""
    action_space = PosteriorQuantileActionSpace(_smbo_with(acquisition_function))
    acquisition_function._model = SimpleNamespace(
        # A has the better mean and low uncertainty; B has a worse mean and
        # high uncertainty. The model returns variance, not standard deviation.
        predict_marginalized=lambda _x: (
            np.asarray([[0.0], [1.0]]),
            np.asarray([[0.01], [4.0]]),
        ),
    )
    acquisition_function._num_data = 1

    choices: list[int] = []
    margins: list[float] = []
    beta: list[float] = []
    for action_index in range(action_space.space.n):
        action_space.update_optimizer(action_index)
        scores = acquisition_function._compute(np.asarray([[0.0], [1.0]])).reshape(-1)
        choices.append(int(np.argmax(scores)))
        margins.append(float(scores[1] - scores[0]))
        beta.append(float(acquisition_function._beta))
    return choices, np.asarray(margins), np.asarray(beta)


def test_lcb_actions_increasingly_favor_the_uncertain_point() -> None:
    """Lower cost quantiles provide the intended exploration ordering."""
    choices, uncertain_point_margins, beta = _two_point_action_margins(LCB(update_beta=False))

    assert choices == [0, 1, 1, 1, 1]
    assert np.all(np.diff(uncertain_point_margins) > 0.0)
    assert np.all(np.diff(beta) > 0.0)


def test_ucb_actions_increasingly_avoid_the_uncertain_point() -> None:
    """An upper cost bound is risk-averse, rather than exploratory, in minimization."""
    choices, uncertain_point_margins, beta = _two_point_action_margins(UCB(update_beta=False))

    assert choices == [0, 0, 0, 0, 0]
    assert np.all(np.diff(uncertain_point_margins) < 0.0)
    assert np.all(np.diff(beta) > 0.0)


def test_median_quantile_is_pure_posterior_mean() -> None:
    acquisition_function = LCB(update_beta=False)
    action_space = PosteriorQuantileActionSpace(
        _smbo_with(acquisition_function),
        quantile_levels=[0.5],
    )

    action_space.update_optimizer(0)

    assert acquisition_function._beta == pytest.approx(0.0)


def test_quantile_mapping_accounts_for_confidence_bound_nu() -> None:
    acquisition_function = LCB(beta=0.25, nu=4.0, update_beta=False)
    action_space = PosteriorQuantileActionSpace(
        _smbo_with(acquisition_function),
        quantile_levels=[0.1],
    )

    assert action_space.current_control_value == pytest.approx(norm.cdf(-1.0))

    action_space.update_optimizer(0)

    assert acquisition_function._beta == pytest.approx(norm.ppf(0.1) ** 2 / 4.0)


@pytest.mark.parametrize(
    ("acquisition_function", "quantile_levels", "message"),
    [
        (LCB(update_beta=False), [], "must not be empty"),
        (LCB(update_beta=False), [0.0], "strictly between 0 and 1"),
        (LCB(update_beta=False), [1.0], "strictly between 0 and 1"),
        (LCB(update_beta=False), [np.nan], "must be finite"),
        (LCB(update_beta=False), [0.1, 0.1], "must be unique"),
        (LCB(update_beta=False), [0.75], "must be <= 0.5"),
        (UCB(update_beta=False), [0.25], "must be >= 0.5"),
    ],
)
def test_quantile_action_rejects_invalid_levels(
    acquisition_function,
    quantile_levels,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        PosteriorQuantileActionSpace(
            _smbo_with(acquisition_function),
            quantile_levels=quantile_levels,
        )


def test_quantile_action_requires_fixed_beta() -> None:
    with pytest.raises(ValueError, match="update_beta=False"):
        PosteriorQuantileActionSpace(_smbo_with(LCB(update_beta=True)))


def test_quantile_action_requires_positive_nu() -> None:
    with pytest.raises(ValueError, match="finite positive nu"):
        PosteriorQuantileActionSpace(_smbo_with(LCB(nu=0.0, update_beta=False)))


def test_quantile_action_requires_finite_beta() -> None:
    with pytest.raises(ValueError, match="finite beta"):
        PosteriorQuantileActionSpace(_smbo_with(LCB(beta=np.nan, update_beta=False)))


def test_quantile_action_requires_confidence_bound_acquisition() -> None:
    with pytest.raises(TypeError, match="requires an LCB or UCB"):
        PosteriorQuantileActionSpace(_smbo_with(EI()))


def test_quantile_action_rejects_out_of_range_index() -> None:
    action_space = PosteriorQuantileActionSpace(_smbo_with(LCB(update_beta=False)))

    with pytest.raises(ValueError, match="outside Discrete"):
        action_space.update_optimizer(5)


def test_legacy_parameter_action_can_control_lcb_beta() -> None:
    acquisition_function = LCB(update_beta=False)
    action_space = AcqParameterActionSpace(
        _smbo_with(acquisition_function),
        bounds=(-1, 1),
        adjustment_type="continuous",
    )

    action_space.update_optimizer(np.array([1.0], dtype=np.float32))

    assert acquisition_function._beta == pytest.approx(10.0)
