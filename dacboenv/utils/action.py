"""Action utilities for DACBOEnv."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, TypeVar

import numpy as np
from gymnasium.spaces import Box, Discrete, Space
from smac.acquisition.function import EI, PI

from dacboenv.utils.confidence_bound import UCB
from dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
    from smac.main.smbo import SMBO

ActType = TypeVar("ActType")


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


ActionType = ParameterAction | FunctionAction


class AbstractActionSpace:
    """Manages action spaces the DACBOenv.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC instance.

    Attributes
    ----------
    _smac_instance : SMBO
        Reference to the associated SMAC instance.
    _action : ActionType
        The action object defining the action space.
    _action_space : Space
        The Gymnasium space for the current action configuration.
    """

    def __init__(self, smac_instance: SMBO) -> None:
        """Initialize the ActionSpace.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance.

        """
        self._smac_instance = smac_instance
        self._action = self._create_action()
        self._action_space = self._action.space

    @abstractmethod
    def _create_action(self) -> ActionType:
        """Create the appropriate action object.

        Returns
        -------
        ActionType
            The action object.
        """
        raise NotImplementedError

    @abstractmethod
    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer based on the chosen action.

        Parameters
        ----------
        action : ActType
            The action according to a policy.
        """
        raise NotImplementedError

    @property
    def space(self) -> Space:
        """Returns the Gymnasium space for the action.

        Returns
        -------
        Space
            The action space.
        """
        return self._action_space


class AcqParameterActionSpace(AbstractActionSpace):
    """Action space for tuning parameters of the current acquisition function.

    Attributes
    ----------
    _PARAMETERS : ClassVar[dict[type[AbstractAcquisitionFunction], ParameterAction]]
        Mapping of acquisition function classes to their corresponding parameter actions.
    """

    _PARAMETERS: ClassVar[dict[AbstractAcquisitionFunction, ParameterAction]] = {
        EI: ParameterAction("_xi", Box(low=-10_000.0, high=10_000.0, dtype=np.float32)),
        PI: ParameterAction("_xi", Box(low=-10_000.0, high=10_000.0, dtype=np.float32)),
        UCB: ParameterAction("_beta", Box(low=-10.0, high=5.0, dtype=np.float32), log=True),
        WEI: ParameterAction("_alpha", Box(low=0.0, high=1.0, dtype=np.float32)),
    }

    def _create_action(self) -> ActionType:
        """Create a ParameterAction for the current acquisition function.

        Returns
        -------
        ParameterAction
            The parameter action object for the selected acquisition function.

        Raises
        ------
        ValueError
            If the acquisition function of the SMAC instance is unsupported.
        """
        acquisition_function = self._smac_instance._intensifier._config_selector._acquisition_function
        if type(acquisition_function) not in self._PARAMETERS:
            raise ValueError("Invalid acquisition function")
        return self._PARAMETERS[type(acquisition_function)]

    def update_optimizer(self, action: ActType) -> None:
        """Update the acquisition function parameter value.

        Parameters
        ----------
        action : ActType
            A single numeric action value for the parameter.
        """
        action_val = np.array(action).item()

        if self._action_space._action.log:
            action_val **= 10

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action_space._action.attr,
            action_val,
        )


class AcqFunctionActionSpace(AbstractActionSpace):
    """Action space for selecting an acquisition function.

    Attributes
    ----------
    _ACQUISITION_FUNCTIONS : ClassVar[dict[int, type[AbstractAcquisitionFunction]]]
        Mapping of integer IDs to available acquisition function classes.
    """

    _ACQUISITION_FUNCTIONS: ClassVar[dict[int, AbstractAcquisitionFunction]] = {0: EI, 1: PI, 2: UCB, 3: WEI}

    def _create_action(self) -> ActionType:
        """Create a FunctionAction representing the discrete selection of acquisition functions.

        Returns
        -------
        FunctionAction
            The FunctionAction object for acquisition function selection.
        """
        return FunctionAction(Discrete(len(self._ACQUISITION_FUNCTIONS)))

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer to use the selected acquisition function.

        Parameters
        ----------
        action : ActType
            Integer index representing the selected acquisition function.
        """
        function_idx = int(np.array(action).item())
        self._smac_instance.update_acquisition_function(self._ACQUISITION_FUNCTIONS[function_idx]())
