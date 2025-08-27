from smac.callback import Callback
from dacboenv.dacboenv import DACBOEnv
import smac

class DACBOEnvCallback(Callback):

    def __init__(self, frequency):
        super().__init__()
        self._frequency = frequency

        self._dacboenv = None
        self._state = None
        self._model = None
    
    def on_start(self, smbo: smac.main.smbo.SMBO) -> None:
        self._dacboenv = DACBOEnv(smac_instance=smbo)
        self._state, _ = self._dacboenv.reset(smbo._scenario.seed)

        # Dummy model: Sample random action
        self._model = lambda obs: self._dacboenv.action_space.sample()
        
        return super().on_start(smbo)

    def on_ask_start(self, smbo):

        # Take environment step
        action = self._model(self._state)

        return super().on_ask_start(smbo)  
    
    def on_tell_end(self, smbo, info, value):

        # Compute new observation and reward
        # Update state

        return super().on_tell_end(smbo, info, value)