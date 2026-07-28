"""Structured-observation contracts for the new acquisition controllers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ConfigSpace import Configuration, ConfigurationSpace, Float
from dacboenv.env.action import (
    PosteriorModeActionSpace,
    PosteriorQuantileActionSpace,
)
from dacboenv.env.observation import (
    ACTION_FEATURE_INDEX,
    AF_ACTION_FEATURE_INDEX,
    GLOBAL_STATE_INDEX,
    ObservationSpace,
)
from dacboenv.utils.confidence_bound import LCB
from dacboenv.utils.posterior_decision import (
    MAXIMUM_VARIANCE,
    POSTERIOR_MODE_NAMES,
    PosteriorModeAcquisition,
)
from scipy.stats import norm
from smac import BlackBoxFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


def objective(config: Configuration, seed: int = 0) -> float:  # noqa: ARG001
    """Return a deterministic one-dimensional objective."""
    return float((config["x"] - 0.75) ** 2)


def make_smbo(
    tmp_path: Path,
    acquisition_function: LCB | PosteriorModeAcquisition,
) -> object:
    """Build a fitted sequential SMAC optimizer."""
    configspace = ConfigurationSpace(
        seed=7,
        space={"x": Float("x", (-5.0, 5.0))},
    )
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=8,
        seed=7,
        output_directory=tmp_path,
    )
    facade = BlackBoxFacade(
        scenario,
        objective,
        initial_design=BlackBoxFacade.get_initial_design(
            scenario,
            n_configs=2,
        ),
        acquisition_function=acquisition_function,
        overwrite=True,
        logging_level=False,
    )
    smbo = facade.optimizer
    while len(smbo.runhistory.get_configs()) < 2:
        tell_one(smbo)
    return smbo


def tell_one(smbo: object) -> None:
    """Evaluate and tell one configuration."""
    trial_info = smbo.ask()
    smbo.tell(
        trial_info,
        TrialValue(cost=objective(trial_info.config), time=0.0),
    )


def test_quantile_actions_receive_matching_candidate_rows(
    tmp_path: Path,
) -> None:
    """Each posterior quantile has one finite, correctly identified row."""
    initial_beta = float(norm.ppf(0.1) ** 2)
    smbo = make_smbo(
        tmp_path / "quantile-smac",
        LCB(beta=initial_beta, update_beta=False),
    )
    action_space = PosteriorQuantileActionSpace(smbo)
    observation_space = ObservationSpace(
        smbo,
        keys=["global_state", "action_features"],
        action_space=action_space,
    )

    initial = observation_space.get_initial_observation()

    assert observation_space.space.contains(initial)
    assert initial["action_features"].shape == (5, 4)
    np.testing.assert_allclose(
        initial["action_features"][:, ACTION_FEATURE_INDEX["control_value"]],
        action_space.quantile_levels,
    )
    assert np.isfinite(initial["action_features"]).all()

    action_space.update_optimizer(4)
    tell_one(smbo)
    updated = observation_space.get_observation()

    assert observation_space.space.contains(updated)
    assert updated["global_state"][
        GLOBAL_STATE_INDEX["previous_control"]
    ] == pytest.approx(0.005)


def test_af_selection_rows_use_one_hot_identity_and_comparable_features(
    tmp_path: Path,
) -> None:
    """Mode rows expose consequences without raw cross-family AF values."""
    smbo = make_smbo(
        tmp_path / "mode-smac",
        PosteriorModeAcquisition(),
    )
    action_space = PosteriorModeActionSpace(smbo)
    observation_space = ObservationSpace(
        smbo,
        keys=["global_state", "af_action_features"],
        action_space=action_space,
    )

    initial = observation_space.get_initial_observation()
    features = initial["af_action_features"]

    assert observation_space.space.contains(initial)
    assert features.shape == (5, 10)
    assert np.isfinite(features).all()
    one_hot = features[:, : len(POSTERIOR_MODE_NAMES)]
    np.testing.assert_array_equal(one_hot, np.eye(5, dtype=np.float32))
    assert np.all(
        (features[:, AF_ACTION_FEATURE_INDEX["normalized_ei_rank"]] >= 0.0)
        & (features[:, AF_ACTION_FEATURE_INDEX["normalized_ei_rank"]] <= 1.0)
    )

    action_space.update_optimizer(POSTERIOR_MODE_NAMES.index(MAXIMUM_VARIANCE))
    tell_one(smbo)
    updated = observation_space.get_observation()

    assert observation_space.space.contains(updated)
    assert action_space.selected_mode == MAXIMUM_VARIANCE
    assert updated["global_state"][
        GLOBAL_STATE_INDEX["previous_control"]
    ] == pytest.approx(1.0)
