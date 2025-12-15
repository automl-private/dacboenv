"""RL Environment for DACBO."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
)

import gymnasium as gym
import numpy as np
from carps.utils.loggingutils import get_logger

from dacboenv.env.action import AbstractActionSpace, AcqParameterActionSpace
from dacboenv.env.instance import InstanceSelector, RoundRobinInstanceSelector
from dacboenv.env.observation import ObservationSpace
from dacboenv.env.reward import DACBOReward
from dacboenv.utils.carps_optimizer import build_carps_optimizer
from dacboenv.utils.math import safe_log10
from dacboenv.utils.reference_performance import ReferencePerformance

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer
    from omegaconf import DictConfig
    from smac.facade.abstract_facade import AbstractFacade
    from smac.main.smbo import SMBO

ObsType = dict[str, Any]
ActType = int | float | list[float]

logger = get_logger("dacboenv")


class DACBOEnv(gym.Env):
    """Gymnasium environment for Dynamic Algorithm Configuration in Bayesian Optimization (DACBO).

    This environment wraps a SMAC optimizer and offers a reinforcement learning interface for
    dynamically adjusting acquisition functions / parameters during Bayesian optimization.

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

    def __init__(  # noqa: PLR0913
        self,
        task_ids: list[str],
        optimizer_cfg: DictConfig | None = None,
        observation_keys: list[str] | None = None,
        action_space_class: type[AbstractActionSpace] = AcqParameterActionSpace,
        action_space_kwargs: dict[str, Any] | None = None,
        reward_keys: list[str] | None = None,
        rho: float = 0.05,
        seed: int | None = None,
        reference_performance_fn: str = "reference_performance/reference_performance.parquet",
        inner_seeds: list[int] | None = None,
        terminate_after_reference_performance_reached: bool = False,  # noqa: FBT001, FBT002
        instance_selector: InstanceSelector | None = None,
    ) -> None:
        """Initialize the DACBOEnv environment.

        Parameters
        ----------
        task_ids : list[str], optional
            The carps task ids that BO should run on.
        optimizer_cfg : DictCOnfig, optional
            The carps (SMAC) optimizer config. Defaults to `SMAC3-BlackBoxFacade` which is the standard blackbox
            facade with a GP.
        observation_keys : list[str], optional
            Which observations to compute at each step.
        action_space_class : type[AbstractActionSpace], optional
            Which action space, either parameter control or acquisition function selection.
        action_space_kwargs : dict[str, Any], optional
            Keyword arguments for the action space class.
        reward_keys : list[str], optional
            Which rewards to compute at each step. If nothing provided, will be `incumbent_cost`. Beware,
            this might not make sense for DAC as the tasks live on different scales.
        rho : float, optional
            ParEGO scalarization parameter.
        inner_seeds : list[int], optional
            The seeds that the inner BO will run on.
        terminate_after_reference_performance_reached : bool, optional
            Terminate episode after a certain reference performance on a task/seed has been reached. Defaults to False.
        """
        if reward_keys is None:
            reward_keys = ["incumbent_cost"]
        if action_space_kwargs is None:
            action_space_kwargs = {
                # SMAC's default acquisition function is EI, thus we adjust xi, thus those are sensible default bounds
                "bounds": (-10, 10)
            }
        super().__init__()

        self._seed = seed
        # Create seed generator for resetting for new episodes
        self._seeder = np.random.default_rng(self._seed)

        self._optimizer_cfg = optimizer_cfg
        self._action_space_class = action_space_class
        self._action_space_kwargs = action_space_kwargs
        self._action_space: AbstractActionSpace
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._rho = rho
        self.task_ids = task_ids
        self.reference_performance_fn = reference_performance_fn
        self._inner_seeds: list[int] = (
            inner_seeds if inner_seeds else list(self._seeder.integers(low=344, high=46483, size=3))
        )
        self._terminate_after_reference_performance_reached = terminate_after_reference_performance_reached

        self.reference_performance_optimizer_id = "SMAC3-BlackBoxFacade"
        if self._terminate_after_reference_performance_reached:
            self._reference_performance = ReferencePerformance(
                optimizer_id=self.reference_performance_optimizer_id,
                task_ids=self.task_ids,
                seeds=self._inner_seeds,
                reference_performance_fn=self.reference_performance_fn,
            )

        self.instance_selector = (
            instance_selector
            if instance_selector
            else RoundRobinInstanceSelector(task_ids=self.task_ids, seeds=self._inner_seeds)
        )

        self._carps_solver: Optimizer
        self._smac_facade: AbstractFacade
        self._smac_instance: SMBO
        self._n_trials = None

        self._episode_reward = 0.0
        self._episode_length = 0

        self.current_task_id = ""
        self.current_seed = -1
        self.current_threshold: float | None = None

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
        return self._dacbo_observation_space.get_observation()

    def get_reward(self) -> float:
        """Compute the current reward from the optimizer.

        Returns
        -------
        reward : float
            The current reward signal.
        """
        return self._reward.get_reward()

    def get_next_instance(self) -> tuple[int, str]:
        """Get the next instance.

        Returns
        -------
        tuple[int,str]
            (seed,task_id)
        """
        return self.instance_selector.select_instance()  # type: ignore[return-value]

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
            Whether the episode has terminated (reference performance reached).
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

        self._episode_reward += reward
        self._episode_length += 1

        terminated = False
        if self._terminate_after_reference_performance_reached:
            curr_incumbent = self.get_incumbent_cost()
            threshold = self._reference_performance.query_cost(  # type: ignore[attr-defined]
                optimizer_id=self.reference_performance_optimizer_id,
                task_id=self.current_task_id,
                seed=self.current_seed,
            )
            self.current_threshold = threshold
            distance = abs(curr_incumbent - threshold)
            log_distance = safe_log10(distance)
            logger.info(f"Current: {curr_incumbent:.4f}, threshold: {threshold:.4f}, log distance: {log_distance:.4f}")
            terminated = curr_incumbent < threshold  # We minimize

        truncated = self._smac_instance.remaining_trials <= 0

        info = {}
        if terminated or truncated:
            info["episode"] = {"r": self._episode_reward, "l": self._episode_length}
            self._episode_reward = 0
            self._episode_length = 0

        return obs, reward, terminated, truncated, info

    def get_incumbent_cost(self) -> float:
        """Get the current incumbent cost.

        Returns
        -------
        float
            Minimum cost found so far on this target function (not necessarily the reward).
        """
        return self._smac_instance.runhistory.get_min_cost(self._smac_instance.intensifier.get_incumbent())

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
        -------
        obs : tuple
            The initial observation.
        info : dict
            Additional information (empty).
        """
        # Reset SMAC instance
        if hasattr(self, "_carps_solver"):
            del self._carps_solver
        if hasattr(self, "_smac_instance"):
            del self._smac_instance

        # Get next instance which is a combo of task id and seed
        self.instance = self.get_next_instance()
        seed, task_id = self.instance
        if seed is None:
            seed = int(self._seeder.integers(low=0, high=2**32 - 1))

        # Build carps optimizer (wrapper around smac) with appropriate objective function
        optimizer_id = "SMAC3-BlackBoxFacade" if self._optimizer_cfg is None else None
        self._carps_solver = build_carps_optimizer(
            optimizer_id=optimizer_id,
            task_id=task_id,
            seed=seed,
            optimizer_cfg=self._optimizer_cfg,
        )
        # Get the smac instance
        self._smac_facade = self._carps_solver.solver
        self._smac_instance = self._carps_solver.solver.optimizer

        if self._smac_instance._scenario.count_objectives() != 1:
            raise NotImplementedError("Multi-objective not supported.")

        # Setup observation space
        self._dacbo_observation_space = ObservationSpace(self._smac_instance, self._observation_keys)
        self.observation_space = self._dacbo_observation_space.space  # gym observation space

        # Setup action space
        self._action_space = self._action_space_class(smac_instance=self._smac_instance, **self._action_space_kwargs)
        self.action_space = self._action_space.space  # gym action space
        self.action_space.seed(seed)  # Seed with current seed

        # Setup reward
        self._reward = DACBOReward(self._smac_instance, self._reward_keys, self._rho)

        super().reset(seed=seed)
        self.current_seed = seed
        self.current_task_id = task_id

        initial_obs = {
            obs.name: np.atleast_1d(obs.default).astype(np.float32)
            for obs in self._dacbo_observation_space._observation_types
        }

        return initial_obs, {}
