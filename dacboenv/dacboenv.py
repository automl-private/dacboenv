"""RL Environment for DACBO."""

from __future__ import annotations

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
        smac_instance_factory: Callable[[], AbstractFacade],
        observation_keys: list[str] | None = None,
        action_mode: str = "parameter",
        reward_keys: list[str] | None = None,
        rho: float = 0.05,
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
        self._solver = self._smac_instance_factory()
        self._smac_instance = self._solver.optimizer
        self._n_trials = self._smac_instance._scenario.n_trials
        self._action_mode = action_mode
        self._action_space: AbstractActionSpace
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._rho = rho

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

        self._reward = DACBOReward(self._smac_instance, self._reward_keys, self._rho)

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
        reward = self.get_reward()

        return obs, reward, self._smac_instance.remaining_trials <= 0, False, {}

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

        self._solver = self._smac_instance_factory()
        self._smac_instance = self._solver.optimizer

        # XXX: The initial design configs remain the same as the initial design
        # is part of the facade

        self._observation_space._smac_instance = self._smac_instance
        self._action_space._smac_instance = self._smac_instance
        self._reward._smac_instance = self._smac_instance

        super().reset(seed=self._smac_instance._scenario.seed)

        initial_obs = {
            obs.name: np.atleast_1d(obs.default).astype(np.float32)
            for obs in self._observation_space._observation_types
        }

        return initial_obs, {}
