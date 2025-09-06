"""SMAC3 Optimizer including an RL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from carps.optimizers.smac20 import SMAC3Optimizer

from dacboenv.dacboenv import ActType, DACBOEnv, ObsType
from dacboenv.env.policy import Policy, RandomPolicy

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.trials import TrialInfo, TrialValue
    from omegaconf import DictConfig
    from smac.facade.abstract_facade import AbstractFacade

from dacboenv.utils.loggingutils import dump_logs


class DACBOEnvOptimizer(SMAC3Optimizer):
    """SMAC3 optimizer wrapper that integrates the DACBOEnv RL environment.

    This optimizer wraps a SMAC3 optimizer and manages a DACBOEnv instance
    for DAC using an RL agent.

    Parameters
    ----------
    task : Any
        The optimization task.
    smac_cfg : Any
        SMAC configuration object.
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

    def __init__(  # noqa: PLR0913
        self,
        task: Task,
        smac_cfg: DictConfig,
        loggers: list[AbstractLogger] | None = None,
        expects_multiple_objectives: bool = False,  # noqa: FBT001, FBT002
        expects_fidelities: bool = False,  # noqa: FBT001, FBT002
        observation_keys: list[str] | None = None,
        action_mode: str = "parameter",
        reward_keys: list[str] | None = None,
        policy: Policy | None = None,
        rho: float = 0.05,
        frequency: int = 1,
    ) -> None:
        """Initialize the DACBOEnvOptimizer.

        Parameters
        ----------
        task : Any
            The optimization task.
        smac_cfg : Any
            SMAC configuration object.
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
        policy : Policy, optional
            Policy for the agent to use. If none is given, act randomly.
        rho : float, optional
            ParEGO scalarization parameter.
        frequency : int, optional
            Frequency (in trials) with which to take environment steps.
        """
        super().__init__(task, smac_cfg, loggers, expects_multiple_objectives, expects_fidelities)

        self._dacboenv: DACBOEnv
        self._state: ObsType
        self._observation_keys = observation_keys
        self._action_mode = action_mode
        self._reward_keys = reward_keys
        self._rho = rho
        self._frequency = frequency
        self._model = policy

        self._obs_flag = False
        self._obsfile = "DACBOEnvLogs.jsonl"
        self._actionfile = "DACBOEnvActions.jsonl"

    def _setup_optimizer(self) -> AbstractFacade:
        """Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns
        -------
        SMAC4AC
            Instance of a SMAC facade.
        """
        solver = super()._setup_optimizer()

        self._dacboenv = DACBOEnv(
            solver.optimizer, self._observation_keys, self._action_mode, self._reward_keys, self._rho
        )
        self._state, _ = self._dacboenv.reset(seed=solver.optimizer._scenario.seed)

        if self._model is None:
            self._model = RandomPolicy(self._dacboenv)

        return solver

    def ask(self) -> TrialInfo:
        """Ask the optimizer for a new trial to evaluate.

        Returns
        -------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        # Don't update during initial design
        if len(self.solver.runhistory) % self._frequency == 0 and len(self.solver.runhistory) > len(
            self.solver.intensifier.config_selector._initial_design_configs
        ):
            assert self._model is not None, "Model must be initialized before calling ask."
            action: ActType = self._model(self._state)
            # print(action)
            self._dacboenv.update_optimizer(action)
            self._obs_flag = True

            logs = {
                "action": np.array(action).item(),
                "action_type": self._dacboenv._action_space._action.name,
                "n_trials": len(self.solver.runhistory),
            }
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

        # TODO: Reactivate
        # if self._obs_flag:  # Compute obs only when needed (optimizer was updated)
        obs = self._dacboenv.get_observation()
        # print(obs)
        # rew = self._dacboenv.get_reward()
        full_reward = self._dacboenv._reward._get_full_reward()
        reward = self._dacboenv._reward._parego(list(full_reward.values()))

        logs = {
            "observation": {k: v.item() if hasattr(v, "item") else v for k, v in obs.items()},
            "full_reward": full_reward,
            "reward": reward,
            "n_trials": len(self.solver.runhistory),
        }
        dump_logs(logs, self._obsfile)

        self._state = obs
        self._obs_flag = False
