from __future__ import annotations

from carps.optimizers.smac20 import SMAC3Optimizer

from dacboenv.dacboenv import DACBOEnv


class DACBOEnvOptimizer(SMAC3Optimizer):
    def __init__(
        self, task, smac_cfg, loggers=None, expects_multiple_objectives=False, expects_fidelities=False, frequency=10
    ):
        super().__init__(task, smac_cfg, loggers, expects_multiple_objectives, expects_fidelities)

        self._dacboenv = None
        self._model = None
        self._state = None
        self._frequency = frequency

    def _setup_optimizer(self):
        solver = super()._setup_optimizer()

        self._dacboenv = DACBOEnv(smac_instance=solver.optimizer)
        self._state, _ = self._dacboenv.reset(seed=solver.optimizer._scenario.seed)

        # Dummy model: Sample random action
        self._model = lambda obs: self._dacboenv.action_space.sample()  # TODO: Insert policy here
        return solver

    def ask(self):
        # Don't update during initial design
        if len(self.solver.runhistory) % self._frequency == 0 and len(self.solver.runhistory) > len(
            self.solver.intensifier.config_selector._initial_design_configs
        ):
            action = self._model(self._state)
            DACBOEnv.update_optimizer(self.solver.optimizer, action)
        return super().ask()

    def tell(self, trial_info, trial_value):
        super().tell(trial_info, trial_value)
        obs, _ = self._dacboenv.get_observation(self.solver.optimizer)
        self._state = obs
