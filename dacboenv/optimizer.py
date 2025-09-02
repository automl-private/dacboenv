"""SMAC3 Optimizer including an RL agent."""

from __future__ import annotations

from carps.optimizers.smac20 import SMAC3Optimizer

from dacboenv.dacboenv import DACBOEnv


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
        Frequency (in trials) at which to update the RL policy.

    Attributes:
    ----------
    _dacboenv : DACBOEnv
        The DACBOEnv RL environment instance.
    _model : Callable
        The RL policy or model for selecting actions.
    _state : Any
        The current observation/state from the environment.
    _frequency : int
        Frequency for RL policy updates.

    Methods:
    ----------
    _setup_optimizer()
        Sets up the underlying SMAC optimizer and initializes DACBOEnv and RL model.
    ask()
        Requests a new configuration and updates the optimizer according to the policy.
    tell(trial_info, trial_value)
        Updates the optimizer and environment with the result of a trial.
    """

    def __init__(
        self, task, smac_cfg, loggers=None, expects_multiple_objectives=False, expects_fidelities=False, frequency=1
    ):
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
        frequency : int, optional
            Frequency (in trials) at which to update the RL policy.
        """
        super().__init__(task, smac_cfg, loggers, expects_multiple_objectives, expects_fidelities)

        self._dacboenv = None
        self._model = None
        self._state = None
        self._frequency = frequency

    def _setup_optimizer(self):
        """Setup SMAC.

        Retrieve defaults and instantiate SMAC.

        Returns:
        ----------
        SMAC4AC
            Instance of a SMAC facade.
        """
        solver = super()._setup_optimizer()

        self._dacboenv = DACBOEnv(smac_instance=solver.optimizer)
        self._state, _ = self._dacboenv.reset(seed=solver.optimizer._scenario.seed)

        # Dummy model: Sample random action
        self._model = lambda obs: self._dacboenv.action_space.sample()  # TODO: Insert policy here
        return solver

    def ask(self):
        """Ask the optimizer for a new trial to evaluate.

        Returns:
        ----------
        TrialInfo
            trial info (config, seed, instance, budget)
        """
        # Don't update during initial design
        if len(self.solver.runhistory) % self._frequency == 0 and len(self.solver.runhistory) > len(
            self.solver.intensifier.config_selector._initial_design_configs
        ):
            action = self._model(self._state)
            self._dacboenv.update_optimizer(action)
        return super().ask()

    def tell(self, trial_info, trial_value):
        """Tell the optimizer a new trial.

        Parameters
        ----------
        trial_info : TrialInfo
            trial info (config, seed, instance, budget)
        trial_value : TrialValue
            trial value (cost, time, ...)
        """
        super().tell(trial_info, trial_value)
        # TODO: Compute only when needed (update_optimizer was called)
        obs = self._dacboenv.get_observation()
        self._state = obs
