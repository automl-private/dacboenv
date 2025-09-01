"""RL Environment for DACBO."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, SupportsFloat, TypeVar

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete
from smac.acquisition.function import EI, PI

from dacboenv.utils.confidence_bound import UCB
from dacboenv.utils.observation import ObservationSpace
from dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class DACBOEnv(gym.Env):
    """Gymnasium environment for DACBO.

    Parameters
    ----------
    smac_kwargs : dict
        Arguments for configuring the SMAC optimizer.
    scenario_kwargs : dict
        Arguments for configuring the SMAC scenario.
    target_function : Callable
        The black-box function to optimize.

    Observation Space
    -----------------
    incumbent_changes : int
        Number of times the incumbent solution has changed.
    trials_passed : int
        Number of optimization trials completed.
    trials_left : int
        Number of trials remaining.
    ubr : float
        Upper bound regret.
    modelfit_mse : float
        Model fit measured as mean squared error.

    Action Space
    ------------
    acquisition_function : int
        Discrete selection among EI, PI, UCB, WEI.
    ei_pi_xi : float
        Parameter for EI/PI acquisition functions.
    ucb_beta : float
        Parameter for UCB acquisition function (log scale).
    wei_alpha : float
        Parameter for WEI acquisition function.

    Methods:
    -------
    step(action)
        Executes one optimization step using the selected acquisition function and parameters.
    reset(seed=None, options=None)
        Resets the environment and optimizer state.
    """

    _acquisition_functions = {0: EI, 1: PI, 2: UCB, 3: WEI}
    _acquisition_function_parameters = {0: "ei_pi_xi", 1: "ei_pi_xi", 2: "ucb_beta", 3: "wei_alpha"}
    _acquisition_function_attrs = {0: "_xi", 1: "_xi", 2: "_beta", 3: "_alpha"}

    def __init__(self, smac_instance: SMBO):
        super().__init__()

        self._smac_instance = smac_instance
        self._n_trials = self._smac_instance._scenario.n_trials
        self._observation_space = ObservationSpace()

        self.observation_space = self._observation_space.observation_space

        self.action_space = Dict(
            {
                "acquisition_function": Discrete(len(DACBOEnv._acquisition_functions)),
                "ei_pi_xi": Box(low=-10_000.0, high=10_000.0, dtype=np.float32),
                "ucb_beta": Box(low=-10.0, high=5.0, dtype=np.float32),  # Log scale
                "wei_alpha": Box(low=0.0, high=1.0, dtype=np.float32),
            }
        )

    @staticmethod
    def update_optimizer(optimizer: SMBO, action: ActType):
        # Update optimizer / Take environment step

        acquisition_function_id = action["acquisition_function"]

        acquisition_function = DACBOEnv._acquisition_functions[acquisition_function_id]
        acquisition_function_parameter = action[DACBOEnv._acquisition_function_parameters[acquisition_function_id]]

        if acquisition_function_id == 2:  # Log scale for UCB beta
            acquisition_function_parameter = 10**acquisition_function_parameter

        optimizer.update_acquisition_function(acquisition_function())
        setattr(
            optimizer._intensifier._config_selector._acquisition_function,
            DACBOEnv._acquisition_function_attrs[acquisition_function_id],
            acquisition_function_parameter,
        )

    def get_observation(self, optimizer: SMBO) -> ObsType:
        obs = self._observation_space.get_observation(optimizer)

        incumbent_cost = optimizer.intensifier.trajectory[-1].costs

        # ParEgo
        # TODO: get_reward()

        # if len(optimizer.intensifier.trajectory) < 2:
        #     last_incumbent_cost = np.inf # TODO: Sensible?
        # else:
        #     last_incumbent = optimizer.intensifier.trajectory[-2].config_ids[-1]
        #     last_incumbent_cost = optimizer.runhistory.get_min_cost(last_incumbent)

        # incumbent_improvement = last_incumbent_cost - incumbent_cost

        reward = incumbent_cost  # XXX: Reward == current incumbent cost

        return obs, reward

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Update optimizer
        DACBOEnv.update_optimizer(self._smac_instance, action)

        # BO step
        trial_info = self._smac_instance.ask()
        trial_value = self._smac_instance._runner.run_wrapper(trial_info)
        self._smac_instance.tell(trial_info, trial_value)

        # Compute observation
        obs, reward = self.get_observation(self._smac_instance)

        return obs, reward, self._smac_instance.budget_exhausted, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        # XXX: Going to be used to actually reset optimization?
        return (0, 0, self._n_trials, np.nan, np.nan), {}
