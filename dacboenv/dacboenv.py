"""RL Environment for DACBO."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
)

import gymnasium as gym
import numpy as np

from dacboenv.env.action import AbstractActionSpace, AcqFunctionActionSpace, AcqParameterActionSpace
from dacboenv.env.observation import ObservationSpace
from dacboenv.env.reward import DACBOReward

if TYPE_CHECKING:
    from smac.facade.abstract_facade import AbstractFacade

ObsType = dict[str, Any]
ActType = int | float | list[float]

# each seed 1
THRESHOLD_2_1_0 = -92.64999983345098
THRESHOLD_2_8_0 = -133.59630637351353
THRESHOLD_2_20_0 = 183.90958853932779


class DACBOEnv(gym.Env):
    """Gymnasium environment for Dynamic Algorithm Configuration in Bayesian Optimization (DACBO).

    This environment wraps a SMAC optimizer and offers a reinforcement learning interface for
    dynamically adjusting acquisition functions / parameters during Bayesian optimization.

    Parameters
    ----------
    smac_instance_factory : Callable[[], AbstractFacade]
        Function returning the SMAC instance. Called for each new episode.
    observation_keys : list[str], optional
        Which observations to compute at each step.
    action_mode : str, optional
        Action mode, either "parameter" (default) or "function".
    reward_keys : list[str], optional
        Which rewards to compute at each step.
    rho : float, optional
        ParEGO scalarization parameter.

    Observation Space
    ----------
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
    ----------
    acquisition_function : int
        Discrete selection among EI, PI, UCB, WEI.
    ei_pi_xi : float
        Parameter for EI/PI acquisition functions.
    ucb_beta : float
        Parameter for UCB acquisition function (log scale).
    wei_alpha : float
        Parameter for WEI acquisition function.

    Methods
    -------
    step(action)
        Executes one optimization step using the selected acquisition function and parameters.
    reset(seed=None, options=None)
        Resets the environment and optimizer state.
    update_optimizer(action)
        Updates the SMAC optimizer with the given action.
    get_observation()
        Computes the current observation and reward from the optimizer.
    get_reward()
        Computes the current reward from the optimizer.
    """

    def __init__(
        self,
        smac_instance_factory: Callable[[int], AbstractFacade],
        observation_keys: list[str] | None = None,
        action_mode: str = "parameter",
        reward_keys: list[str] | None = None,
        rho: float = 0.05,
        seed: int = -1,
    ) -> None:
        """Initialize the DACBOEnv environment.

        Parameters
        ----------
        smac_instance_factory : Callable[[], AbstractFacade]
            Function returning the SMAC instance. Called for each new episode.
        observation_keys : list[str], optional
            Which observations to compute at each step.
        action_mode : str, optional
            Action mode, either "parameter" (default) or "function".
        reward_keys : list[str], optional
            Which rewards to compute at each step.
        rho : float, optional
            ParEGO scalarization parameter.
        """
        super().__init__()

        self._smac_instance_factory = smac_instance_factory
        self._seed = seed
        self._solver = self._smac_instance_factory(self._seed)
        self._smac_instance = self._solver.optimizer
        self._n_trials = self._smac_instance._scenario.n_trials
        self._action_mode = action_mode
        self._action_space: AbstractActionSpace
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._rho = rho

        # Create seed generator for resetting for new episodes
        self._seeder = np.random.default_rng(self._seed)

        if self._smac_instance._scenario.count_objectives() != 1:
            raise NotImplementedError("Multi-objective not supported.")

        self._observation_space = ObservationSpace(self._smac_instance, self._observation_keys)
        self.observation_space = self._observation_space.space

        if self._action_mode == "parameter":
            self._action_space = AcqParameterActionSpace(self._smac_instance)
        elif self._action_mode == "function":
            self._action_space = AcqFunctionActionSpace(self._smac_instance)
        else:
            raise ValueError("Invalid action mode given")

        self.action_space = self._action_space.space
        self.action_space.seed(self._seed)

        self._reward = DACBOReward(self._smac_instance, self._reward_keys, self._rho)

        self._episode_reward = 0.0
        self._episode_length = 0

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer with the given action.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Raises
        ------
        ValueError
            If the action type is invalid.
        """
        self._action_space.update_optimizer(action)

    def get_observation(self) -> ObsType:
        """Compute the current observation from the optimizer.

        Returns
        -------
        obs : dict[str, Any]
            Dictionary of observation values.
        """
        return self._observation_space.get_observation()

    def get_reward(self) -> float:
        """Compute the current reward from the optimizer.

        Returns
        -------
        reward : float
            The current reward signal.
        """
        return self._reward.get_reward()

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one optimization step using the selected acquisition function and parameters.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Returns
        -------
        obs : dict
            The new observation after taking the action.
        reward : float
            The reward for the action taken.
        terminated : bool
            Whether the episode has terminated (budget exhausted).
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (empty).
        """
        self.update_optimizer(action)

        # BO step
        trial_info = self._smac_instance.ask()
        _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
        self._smac_instance.tell(trial_info, trial_value)

        # Compute observation + reward
        obs = self.get_observation()
        # reward = self.get_reward()

        curr_incumbent = self._smac_instance.runhistory.get_min_cost(self._smac_instance.intensifier.get_incumbent())

        if os.environ["FID"] == "1":
            threshold = THRESHOLD_2_1_0
        elif os.environ["FID"] == "8":
            threshold = THRESHOLD_2_8_0
        elif os.environ["FID"] == "20":
            threshold = THRESHOLD_2_20_0
        else:
            threshold = float("-inf")

        budget = self._smac_instance._scenario.n_trials
        init_des_size = len(self._smac_instance.intensifier.config_selector._initial_design_configs)

        b = budget - init_des_size

        reward = -1 / b if curr_incumbent >= threshold else 0

        self._episode_reward += reward
        self._episode_length += 1

        done = self._smac_instance.remaining_trials <= 0 or reward == 1
        info = {}

        if done:
            info["episode"] = {"r": self._episode_reward, "l": self._episode_length}
            self._episode_reward = 0
            self._episode_length = 0

        return obs, reward, done, False, info

    def reset(
        self,
        *,
        seed: int | None = None,  # noqa: ARG002
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : tuple
            The initial observation.
        info : dict
            Additional information (empty).
        """
        # Reset SMAC instance
        del self._smac_instance
        del self._solver

        new_seed = int(self._seeder.integers(low=0, high=2**32 - 1))

        self._solver = self._smac_instance_factory(new_seed)  # Update seed for new episode
        self._smac_instance = self._solver.optimizer

        self._observation_space._smac_instance = self._smac_instance
        self._action_space._smac_instance = self._smac_instance
        self._reward._smac_instance = self._smac_instance

        if hasattr(self._action_space, "_last"):
            self._action_space._last = 0

        super().reset(seed=new_seed)

        # Work off new initial design
        for _ in self._smac_instance.intensifier.config_selector._initial_design_configs:
            trial_info = self._smac_instance.ask()
            _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
            self._smac_instance.tell(trial_info, trial_value)

        initial_obs = (
            np.atleast_1d(self._observation_space._observation_types[0].default).astype(np.float32)
            if len(self._observation_space._observation_types) == 1
            else {
                obs.name: np.atleast_1d(obs.default).astype(np.float32)
                for obs in self._observation_space._observation_types
            }
        )

        return initial_obs, {}
