"""RL Environment for DACBO."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
    TypeVar,
)

import gymnasium as gym
import numpy as np

from dacboenv.utils.action import ActionSpace, FunctionAction, ParameterAction
from dacboenv.utils.observation import ObservationSpace
from dacboenv.utils.reward import DACBOReward

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

ObsType = dict[str, Any]
ActType = TypeVar("ActType")


class DACBOEnv(gym.Env):
    """Gymnasium environment for Dynamic Algorithm Configuration in Bayesian Optimization (DACBO).

    This environment wraps a SMAC optimizer and offers a reinforcement learning interface for
    dynamically adjusting acquisition functions / parameters during Bayesian optimization.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC optimizer instance.
    action_mode : str, optional
        Action mode, either "parameter" (default) or "function".

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
    ----------
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

    def __init__(self, smac_instance: SMBO, action_mode: str = "parameter"):
        """Initialize the DACBOEnv environment.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.
        action_mode : str, optional
            Action mode, either "parameter" (default) or "function".
        """
        super().__init__()

        self._smac_instance = smac_instance
        self._n_trials = self._smac_instance._scenario.n_trials
        self._action_mode = action_mode

        self._observation_space = ObservationSpace(self._smac_instance)
        self.observation_space = self._observation_space.space

        self._action_space = ActionSpace(self._smac_instance, self._action_mode)
        self.action_space = self._action_space.space

        self._reward = DACBOReward(self._smac_instance)

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer with the given action.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Raises
        ----------
        ValueError
            If the action type is invalid.
        """
        if isinstance(self._action_space._action, ParameterAction):
            action_array = np.array(action, dtype=np.float32)
            action_val = action_array[0]

            if self._action_space._action.log:
                action_val **= 10

            setattr(
                self._smac_instance._intensifier._config_selector._acquisition_function,
                self._action_space._action.attr,
                action_val,
            )

        elif isinstance(self._action_space._action, FunctionAction):
            function_idx = int(np.array(action).item())
            self._smac_instance.update_acquisition_function(ActionSpace._ACQUISITION_FUNCTIONS[function_idx]())
        else:
            raise ValueError("Invalid action type")

    def get_observation(self) -> ObsType:
        """Compute the current observation from the optimizer.

        Returns
        ----------
        obs : dict[str, Any]
            Dictionary of observation values.
        """
        return self._observation_space.get_observation()

    def get_reward(self) -> float:
        """Compute the current reward from the optimizer.

        Returns
        ----------
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
        ----------
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
        trial_value = self._smac_instance._runner.run_wrapper(trial_info)
        self._smac_instance.tell(trial_info, trial_value)

        # Compute observation + reward
        obs = self.get_observation()
        reward = self.get_reward()

        return obs, reward, self._smac_instance.budget_exhausted, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
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
        ----------
        obs : tuple
            The initial observation.
        info : dict
            Additional information (empty).
        """
        super().reset(seed=seed)

        initial_obs = {obs.name: obs.default for obs in self._observation_space._observation_types}

        # XXX: Going to be used to actually reset optimization?
        return initial_obs, {}
