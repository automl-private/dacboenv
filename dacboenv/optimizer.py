"""SMAC3 Optimizer including an RL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from carps.optimizers.smac20 import SMAC3Optimizer

from dacboenv.dacboenv import ActType, DACBOEnv, ObsType
from dacboenv.utils.policy import Policy, RandomPolicy

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.task import Task
    from carps.utils.trials import TrialInfo, TrialValue
    from omegaconf import DictConfig
    from smac.facade.abstract_facade import AbstractFacade


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
        rho : float, optional
            ParEGO scalarization parameter.
        frequency : int, optional
            Frequency (in trials) with which to take environment steps.
        """
        super().__init__(task, smac_cfg, loggers, expects_multiple_objectives, expects_fidelities)

        self._dacboenv: DACBOEnv
        self._model: Policy
        self._state: ObsType
        self._observation_keys = observation_keys
        self._action_mode = action_mode
        self._reward_keys = reward_keys
        self._rho = rho
        self._frequency = frequency

        self._obs_flag = False

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

        # Dummy model: Sample random action
        self._model = RandomPolicy(self._dacboenv)  # TODO: Insert policy here
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
            action: ActType = self._model(self._state)
            print(action)
            self._dacboenv.update_optimizer(action)
            self._obs_flag = True
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

        if self._obs_flag:  # Compute obs only when needed (optimizer was updated)
            obs = self._dacboenv.get_observation()
            print(obs)
            self._dacboenv.get_reward()
            self._state = obs
            self._obs_flag = False
