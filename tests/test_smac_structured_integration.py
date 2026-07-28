"""Integration contract for structured observations against a real SMAC loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from ConfigSpace import Categorical, Configuration, ConfigurationSpace, Float, Integer
from dacboenv.env.observation import (
    ACTION_FEATURE_INDEX,
    GLOBAL_STATE_INDEX,
    WEI_ALPHA_LEVELS,
    ObservationSpace,
)
from dacboenv.env.reward import DACBOReward
from dacboenv.utils.weighted_expected_improvement import WEI
from smac import BlackBoxFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


def objective(config: Configuration, seed: int = 0) -> float:  # noqa: ARG001
    """Return a deterministic mixed-space test objective."""
    categorical_penalty = 0.5 if config["kind"] == "b" else 0.0
    return float((config["x"] - 0.75) ** 2 + 0.1 * config["n"] + categorical_penalty)


def tell_one(smbo: object) -> None:
    """Evaluate and tell one configuration."""
    trial_info = smbo.ask()
    smbo.tell(
        trial_info,
        TrialValue(cost=objective(trial_info.config), time=0.0),
    )


def test_real_smac_structured_transition(tmp_path: Path) -> None:
    """Structured state is finite, contained, and updated after a real tell."""
    configspace = ConfigurationSpace(
        seed=7,
        space={
            "x": Float("x", (-5.0, 5.0)),
            "n": Integer("n", (1, 5)),
            "kind": Categorical("kind", ["a", "b"]),
        },
    )
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=12,
        seed=7,
        output_directory=tmp_path,
    )
    facade = BlackBoxFacade(
        scenario,
        objective,
        initial_design=BlackBoxFacade.get_initial_design(scenario, n_configs=3),
        acquisition_function=WEI(),
        overwrite=True,
        logging_level=False,
    )
    smbo = facade.optimizer
    initial_design_size = len(smbo.intensifier.config_selector._initial_design_configs)
    while len(smbo.runhistory.get_configs()) < initial_design_size:
        tell_one(smbo)

    observation_space = ObservationSpace(
        smbo,
        keys=["global_state", "action_features"],
    )
    initial_observation = observation_space.get_initial_observation()

    assert set(initial_observation) == {"global_state", "action_features"}
    assert observation_space.space.contains(initial_observation)
    assert initial_observation["global_state"].shape == (13,)
    assert initial_observation["action_features"].shape == (5, 4)
    np.testing.assert_array_equal(
        initial_observation["action_features"][:, ACTION_FEATURE_INDEX["alpha"]],
        WEI_ALPHA_LEVELS,
    )
    assert np.isfinite(initial_observation["action_features"]).all()
    assert np.any(initial_observation["action_features"][:, ACTION_FEATURE_INDEX["normalized_uncertainty"]] > 0)

    reward = DACBOReward(smbo, keys=["reference_free_improvement"])
    tell_one(smbo)
    next_observation = observation_space.get_observation()
    transition_reward = reward.get_reward()

    assert observation_space.space.contains(next_observation)
    assert np.isfinite(transition_reward)
    assert transition_reward >= 0.0
    assert next_observation["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] == np.float32(
        smbo.runhistory.finished / scenario.n_trials
    )
    assert reward.get_reward() == transition_reward
