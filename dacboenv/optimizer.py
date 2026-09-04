"""SMAC3 Optimizer including an RL agent."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from carps.optimizers.smac20 import SMAC3Optimizer
from hydra.utils import get_class
from omegaconf import DictConfig

from dacboenv.env.action import (
    PosteriorModeActionSpace,
    PosteriorQuantileActionSpace,
    WEIDiscreteActionSpace,
    WEITempoRLActionSpace,
)
from dacboenv.policy.random import RandomPolicy
from dacboenv.utils.carps_optimizer import task_ids_equivalent
from dacboenv.utils.loggingutils import dump_logs, get_logger

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.trials import TrialInfo, TrialValue
    from smac.facade.abstract_facade import AbstractFacade

    from dacboenv.dacboenv import DACBOEnv
    from dacboenv.env.observations.types import ObsType
    from dacboenv.policy.abstract_policy import Policy

logger = get_logger("OptPolicy")

# Generated evaluation launchers use this capability marker to reject stale
# installations that still try to convert NoOpPolicy's None action to int.
SUPPORTS_NOOP_POLICY_ACTION = True


class DACBOEnvOptimizer(SMAC3Optimizer):
    """SMAC3 optimizer wrapper that integrates the DACBOEnv RL environment.

    This optimizer wraps a SMAC3 optimizer and manages a DACBOEnv instance
    for DAC using an RL agent.

    Parameters
    ----------
    task : Task
        The optimization task.
    dacboenv : DACBOEnv
        DACBO Env. Contains the SMAC configuration.
    loggers : list, optional
        List of logger instances.
    expects_multiple_objectives : bool, optional
        Whether the optimizer expects multiple objectives.
    expects_fidelities : bool, optional
        Whether the optimizer expects fidelity parameters.
    frequency : int, optional
        Frequency (in trials) with which to take environment steps.

    Attributes
    ----------
    _dacboenv : DACBOEnv
        The DACBOEnv RL environment instance.
    _model : Callable
        The RL policy or model for selecting actions.
    _state : Any
        The current observation/state from the environment.
    _frequency : int
        Frequency for RL env steps updates.

    Methods
    -------
    _setup_optimizer()
        Sets up the underlying SMAC optimizer and initializes DACBOEnv and RL model.
    ask()
        Requests a new configuration and updates the optimizer according to the policy.
    tell(trial_info, trial_value)
        Updates the optimizer and environment with the result of a trial.
    """

    def __init__(
        self,
        task: Task,
        dacboenv: DACBOEnv,
        seed: int | None = None,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
        policy_class: type[Policy] | str = RandomPolicy,
        policy_kwargs: dict[str, Any] | None = None,
        log_observations: bool = False,  # noqa: FBT001, FBT002
    ) -> None:
        """Initialize the DACBOEnvOptimizer.

        Parameters
        ----------
        task : Any
            The optimization task.
        dacboenv : DACBOEnv
            DACBO Env. Contains the SMAC configuration.
        loggers : list, optional
            List of logger instances.
        expects_multiple_objectives : bool, optional
            Whether the optimizer expects multiple objectives.
        expects_fidelities : bool, optional
            Whether the optimizer expects fidelity parameters.
        observation_keys : list[str], optional
            Which observations to compute at each step.
        action_mode : str, optional
            Action mode, either "parameter" (default) or "function".
        reward_keys : list[str], optional
            Which rewards to compute at each step.
        policy_class : type[Policy] | str, optional
            The class of the policy for the agent to use.
            If none is given, act randomly.
        policy_kwargs : dict[str, Any], optional
            Keyword arguments to pass to the policy class constructor.
        rho : float, optional
            ParEGO scalarization parameter.
        frequency : int, optional
            Frequency (in trials) with which to take environment steps.
        log_observations : bool, optional
            Whether to log observations. Could be many.
        """
        smac_cfg = dacboenv._optimizer_cfg.smac_cfg if dacboenv._optimizer_cfg is not None else DictConfig({})
        super().__init__(
            task=task,
            smac_cfg=smac_cfg,
            loggers=loggers,
            expects_fidelities=expects_fidelities,
            expects_multiple_objectives=expects_multiple_objectives,
        )

        self.configspace = self.task.input_space.configuration_space
        self._solver: AbstractFacade | None = None

        self._seed = seed
        self._dacboenv: DACBOEnv = dacboenv
        self._state: ObsType

        self._policy_class = policy_class if isinstance(policy_class, type | partial) else get_class(policy_class)
        self._policy_kwargs = policy_kwargs if policy_kwargs is not None else {}

        self._log_observations = log_observations

        self._obsfile = "DACBOEnvLogs.jsonl"
        self._actionfile = "DACBOEnvActions.jsonl"

        self._skip_duration: int = 0

    def _setup_optimizer(self) -> AbstractFacade:
        """Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns
        -------
        AbstractFacade
            Instance of a SMAC facade.
        """
        self._state, _ = self._dacboenv.reset()
        # Increase the trial counter because the DACBOEnv works off initial design
        # during reset. The initial design counts to the overall trials.
        self.trial_counter += self._dacboenv.get_n_finished_trials()
        if self._seed != self._dacboenv._smac_instance._scenario.seed:
            raise ValueError(f"Seeds not the same: {self._seed} != {self._dacboenv._smac_instance._scenario.seed}")
        inner_task_name = self._dacboenv._carps_solver.task.name
        if not task_ids_equivalent(inner_task_name, self.task.name):
            raise ValueError(f"Tasks not the same: {self._dacboenv._carps_solver.task.name} != {self.task.name}")
        selected_instances = self._dacboenv.instance_selector.instances
        instances_match = (
            len(selected_instances) == 1
            and selected_instances[0][0] == self._seed
            and task_ids_equivalent(selected_instances[0][1], self.task.name)
        )
        if not instances_match:
            raise ValueError(
                "Inner seed and task id not matching: "
                f"{self._dacboenv.instance_selector.instances} != {[(self._seed, self.task.name)]}"
            )
        self._policy = self._policy_class(self._dacboenv, **self._policy_kwargs)
        self._policy.set_seed(self._seed)
        return self._dacboenv._smac_facade

    def _structured_action_values(self) -> list[Any] | None:
        """Return human-readable values for the current structured action."""
        action = getattr(self, "action", None)
        action_space = self._dacboenv._action_space
        if action is None:
            return None
        if isinstance(action_space, WEITempoRLActionSpace):
            return [
                action_space._step_durations[action[0]],
                action_space._param_levels[action[1]],
            ]

        action_idx = int(np.asarray(action).item())
        if isinstance(action_space, WEIDiscreteActionSpace):
            action_value: Any = action_space._param_levels[action_idx]
        elif isinstance(action_space, PosteriorQuantileActionSpace):
            action_value = action_space.quantile_levels[action_idx]
        elif isinstance(action_space, PosteriorModeActionSpace):
            action_value = action_space.selected_mode
        else:
            return None
        return [self._dacboenv.interaction_frequency, action_value]

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        Returns
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        # Don't update during initial design
        if len(self.solver.runhistory) >= len(self.solver.intensifier.config_selector._initial_design_configs):
            assert self._policy is not None, "Policy must be initialized before calling ask."
            if self._skip_duration == 0:
                self.action = self._policy(self._state)

                # NoOpPolicy
                if self.action is not None:
                    if isinstance(self._dacboenv._action_space, WEITempoRLActionSpace):
                        self._skip_duration = self._dacboenv._action_space._step_durations[self.action[0]] - 1
                        self._param_level = self.action[1]
                        logger.info(f"Apply action {self._param_level} for {self._skip_duration + 1} steps.")
                    elif self._dacboenv.interaction_frequency > 1:
                        self._skip_duration = self._dacboenv.interaction_frequency - 1
                        self._param_level = self.action
                        logger.info(f"Apply action {self._param_level} for {self._skip_duration + 1} steps.")

                    self._dacboenv.update_optimizer(self.action)
            else:
                self._skip_duration -= 1

            # Log actions
            logs = {
                "action_type": self._dacboenv._action_space._action.name,
                "n_trials": len(self.solver.runhistory),
            }
            if isinstance(self.action, np.ndarray):
                action_array = np.asarray(self.action)
                action_to_log = action_array.item() if action_array.size == 1 else action_array.tolist()
            else:
                action_to_log = self.action
            action_values = self._structured_action_values()
            if action_values is not None:
                logs["action_values"] = action_values
            logs["action"] = action_to_log
            dump_logs(logs, self._actionfile)

        return super().ask()

    def tell(self, trial_info: TrialInfo, trial_value: TrialValue) -> None:
        """Tell the optimizer a new trial.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        super().tell(trial_info, trial_value)

        obs = self._dacboenv.get_observation()
        rew = self._dacboenv.get_reward()

        logs = {
            "observation": {
                k: v.item()
                if hasattr(v, "item") and np.ndim(v) == 0
                else v[0].item()
                if isinstance(v, np.ndarray) and len(v) == 1
                else v.tolist()
                if isinstance(v, np.ndarray)
                else v
                for k, v in obs.items()
            },
            "reward": rew,
            "n_trials": len(self.solver.runhistory),
        }
        dump_logs(logs, self._obsfile)
        self._state = obs
