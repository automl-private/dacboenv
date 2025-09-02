"""Action utilities for DACBOEnv."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from smac.acquisition.function import EI, PI

from dacboenv.utils.confidence_bound import UCB
from dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
    from smac.main.smbo import SMBO


@dataclass
class ParameterAction:
    """Represents a parameter action for a fixed acquisition function.

    Attributes
    ----------
    attr : str
        Name of the function object's attribute.
    space : Space
        Gymnasium space for the parameter's value range and type.
    log : bool, optional
        Whether the parameter is interpreted in log scale.
    """

    attr: str
    space: Space
    log: bool = False


@dataclass
class FunctionAction:
    """Represents an action for selecting an acquisition function.

    Attributes
    ----------
    space : Space
        Gymnasium space for the discrete selection of acquisition functions.
    """

    space: Space


class ActionSpace:
    """Manages action spaces for acquisition function selection and parameter control.

    Depending on the mode, provides a Gymnasium space for either selecting
    an acquisition function or tuning its parameters.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC instance.
    mode : str, optional
        Action mode, either "parameter" (default) or "function".

    Attributes
    ----------
    action_space : Space
        The Gymnasium space for the current action mode.

    Raises
    ------
    ValueError
        If the acquisition function or mode is invalid.
    """

    _ACQUISITION_FUNCTIONS: ClassVar[dict[int, AbstractAcquisitionFunction]] = {0: EI, 1: PI, 2: UCB, 3: WEI}
    _PARAMETERS: ClassVar[dict[AbstractAcquisitionFunction, ParameterAction]] = {
        EI: ParameterAction("_xi", Box(low=-10_000.0, high=10_000.0, dtype=np.float32)),
        PI: ParameterAction("_xi", Box(low=-10_000.0, high=10_000.0, dtype=np.float32)),
        UCB: ParameterAction("_beta", Box(low=-10.0, high=5.0, dtype=np.float32), log=True),
        WEI: ParameterAction("_alpha", Box(low=0.0, high=1.0, dtype=np.float32)),
    }

    def __init__(self, smac_instance: SMBO, mode: str = "parameter") -> None:
        """Initialize the ActionSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.
        mode : str, optional
            Action mode, either "parameter" (default) or "function".

        Raises
        ------
        ValueError
            If the acquisition function or mode is invalid.
        """
        self._mode = mode
        self._smac_instance = smac_instance
        self._action: ParameterAction | FunctionAction

        if self._mode == "parameter":
            acquisition_function = self._smac_instance._intensifier._config_selector._acquisition_function
            if type(acquisition_function) not in ActionSpace._PARAMETERS:
                raise ValueError("Invalid acquisition function")

            self._action = ActionSpace._PARAMETERS[type(acquisition_function)]
        elif self._mode == "function":
            self._action = FunctionAction(Discrete(len(ActionSpace._ACQUISITION_FUNCTIONS)))
        else:
            raise ValueError("Invalid action mode given")

        self._action_space = self._action.space

    @property
    def space(self) -> Space:
        """Returns the Gymnasium space for the current action mode.

        Returns
        -------
        Space
            The action space.
        """
        return self._action_space
