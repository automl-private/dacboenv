"""Action utilities for DACBOEnv."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Space
from scipy.stats import norm
from smac.acquisition.function import EI, PI
from smac.main.config_selector import ConfigSelector
from smac.main.smbo import SMBO

from dacboenv.utils.confidence_bound import LCB, UCB
from dacboenv.utils.posterior_decision import (
    POSTERIOR_MODE_NAMES,
    PosteriorModeAcquisition,
)
from dacboenv.utils.weighted_expected_improvement import WEI

if TYPE_CHECKING:
    from smac.acquisition.function.abstract_acquisition_function import AbstractAcquisitionFunction
    from smac.main.smbo import SMBO

    from dacboenv.dacboenv import ActType


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
    name : str
        String representation of the action.
    """

    attr: str
    space: Space
    log: bool = False
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"ParameterAction:{self.attr}"


@dataclass
class FunctionAction:
    """Represents an action for selecting an acquisition function.

    Attributes
    ----------
    space : Space
        Gymnasium space for the discrete selection of acquisition functions.
    name : str
        String representation of the action.
    """

    space: Space
    name: str = field(init=False, default="FunctionAction")


ActionType = ParameterAction | FunctionAction

_POSTERIOR_MEDIAN_QUANTILE = 0.5


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


class WEITempoRLActionSpace(AbstractActionSpace):
    """TempoRL Action Space for WEI.

    The first action is the skip duration, the second the action to hold.
    This might prevent wild oscillating actions/parameter values.
    """

    def __init__(
        self, smac_instance: SMBO, step_durations: list[int] | None, param_levels: list[float] | None = None
    ) -> None:
        self._step_durations = list(step_durations) if step_durations is not None else [1, 5, 10]
        if len(self._step_durations) == 0:
            raise ValueError("step_durations must not be empty.")
        if any(not isinstance(duration, int) or isinstance(duration, bool) for duration in self._step_durations):
            raise TypeError(f"All step_durations must be integers, got {self._step_durations}")
        if any(duration <= 0 for duration in self._step_durations):
            raise ValueError(f"All step_durations must be > 0, got {self._step_durations}")
        self._param_levels = list(param_levels) if param_levels is not None else [0.0, 0.25, 0.5, 0.75, 1]
        if len(self._param_levels) == 0:
            raise ValueError("param_levels must not be empty.")
        parameter_levels = np.asarray(self._param_levels, dtype=float)
        if not np.isfinite(parameter_levels).all():
            raise ValueError(f"All param_levels must be finite, got {self._param_levels}")
        if np.any((parameter_levels < 0.0) | (parameter_levels > 1.0)):
            raise ValueError(f"WEI param_levels must be in [0, 1], got {self._param_levels}")
        super().__init__(smac_instance)

    def _create_action(self) -> ParameterAction | FunctionAction:
        nvec = [len(self._step_durations), len(self._param_levels)]
        return ParameterAction(attr="_alpha", space=MultiDiscrete(nvec=nvec), log=False)

    def update_optimizer(self, action: ActType) -> None:
        """Update the acquisition function parameter value.

        Parameters
        ----------
        action : ActType
            A single numeric action value for the parameter.
        """
        assert isinstance(action, Sequence | np.ndarray)
        assert isinstance(self._smac_instance._intensifier._config_selector, ConfigSelector)

        param_level_idx = int(action[1])
        param_val = self._param_levels[param_level_idx]

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action.attr,  # type: ignore[union-attr]
            param_val,
        )


class WEIDiscreteActionSpace(AbstractActionSpace):
    """Choose one value from an explicit, discrete set of WEI alpha levels.

    Unlike ``AcqParameterActionSpace(adjustment_type="bucket")``, action
    indices are mapped through ``param_levels`` instead of being interpreted
    as integer offsets. This keeps the public Gymnasium action space at
    ``Discrete(len(param_levels))`` while allowing non-equidistant parameter
    values.
    """

    def __init__(self, smac_instance: SMBO, param_levels: list[float] | None = None) -> None:
        """Initialize the exact discrete WEI action space.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC instance whose WEI acquisition function is controlled.
        param_levels : list[float] | None, optional
            Alpha values addressed by action indices. Defaults to
            ``[0, 0.25, 0.5, 0.75, 1]``.
        """
        self._param_levels = list(param_levels) if param_levels is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
        if len(self._param_levels) == 0:
            raise ValueError("param_levels must not be empty.")
        parameter_levels = np.asarray(self._param_levels, dtype=float)
        if not np.isfinite(parameter_levels).all():
            raise ValueError(f"All param_levels must be finite, got {self._param_levels}")
        if np.any((parameter_levels < 0.0) | (parameter_levels > 1.0)):
            raise ValueError(f"WEI param_levels must be in [0, 1], got {self._param_levels}")
        super().__init__(smac_instance)

    def _create_action(self) -> ParameterAction:
        """Create one categorical action for each configured alpha."""
        return ParameterAction(attr="_alpha", space=Discrete(n=len(self._param_levels)), log=False)

    def update_optimizer(self, action: ActType) -> None:
        """Set WEI alpha to the value addressed by ``action``."""
        assert isinstance(self._smac_instance._intensifier._config_selector, ConfigSelector)

        action_idx = int(np.asarray(action).item())
        if not self.space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} is outside {self.space}.")
        param_val = self._param_levels[action_idx]

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action.attr,
            param_val,
        )


class PosteriorQuantileActionSpace(AbstractActionSpace):
    """Choose an exact posterior quantile for a confidence-bound acquisition.

    SMAC maximizes acquisition values while DACBOEnv minimizes objective
    values.  For :class:`LCB`, a lower posterior quantile ``q <= 0.5`` is
    therefore represented by

    ``mu + Phi^-1(q) sigma = mu - sqrt(beta) sigma``.

    For :class:`UCB`, the mirrored upper posterior quantile ``q >= 0.5`` is
    represented by

    ``mu + Phi^-1(q) sigma = mu + sqrt(beta) sigma``.

    In both cases ``beta = Phi^-1(q) ** 2 / nu``; the acquisition-function
    class supplies the orientation. LCB is the optimistic/exploratory
    orientation for minimization. UCB is the
    pessimistic/uncertainty-averse orientation.
    """

    DEFAULT_LCB_QUANTILES: ClassVar[tuple[float, ...]] = (0.5, 0.25, 0.1, 0.025, 0.005)
    DEFAULT_UCB_QUANTILES: ClassVar[tuple[float, ...]] = (0.5, 0.75, 0.9, 0.975, 0.995)

    def __init__(
        self,
        smac_instance: SMBO,
        quantile_levels: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        """Initialize a categorical posterior-quantile action space.

        Parameters
        ----------
        smac_instance : SMBO
            SMAC instance whose confidence-bound acquisition is controlled.
        quantile_levels : list[float] | tuple[float, ...] | None, optional
            Exact posterior quantiles addressed by action indices. Defaults
            to five lower quantiles for LCB and their mirrored upper
            quantiles for UCB.
        """
        acquisition_function = smac_instance._intensifier._config_selector._acquisition_function
        if not isinstance(acquisition_function, LCB | UCB):
            raise TypeError(
                "PosteriorQuantileActionSpace requires an LCB or UCB "
                f"acquisition function, got {type(acquisition_function).__name__}."
            )
        if acquisition_function._update_beta:
            raise ValueError(
                "Posterior-quantile control requires `update_beta=False`; "
                "otherwise SMAC's confidence schedule would overwrite the selected quantile."
            )

        if quantile_levels is None:
            if isinstance(acquisition_function, LCB):
                defaults = self.DEFAULT_LCB_QUANTILES
            else:
                defaults = self.DEFAULT_UCB_QUANTILES
            quantile_levels = list(defaults)

        self._quantile_levels = self._validate_quantile_levels(
            quantile_levels=quantile_levels,
            acquisition_function=acquisition_function,
        )
        self._param_levels = self._quantile_levels
        self._bound_type = acquisition_function.bound_type
        nu = float(acquisition_function._nu)
        if not np.isfinite(nu) or nu <= 0.0:
            raise ValueError(f"Posterior-quantile control requires a finite positive nu, got {nu}.")
        beta = float(acquisition_function._beta)
        if not np.isfinite(beta):
            raise ValueError(f"Posterior-quantile control requires a finite beta, got {beta}.")
        beta = max(beta, 0.0)
        signed_kappa = np.sqrt(nu * beta)
        if isinstance(acquisition_function, LCB):
            signed_kappa = -signed_kappa
        self._current_quantile = float(norm.cdf(signed_kappa))
        acquisition_function._posterior_quantile = self._current_quantile
        super().__init__(smac_instance)

    @staticmethod
    def _validate_quantile_levels(
        quantile_levels: list[float] | tuple[float, ...],
        acquisition_function: LCB | UCB,
    ) -> list[float]:
        """Validate and copy configured posterior quantiles."""
        if len(quantile_levels) == 0:
            raise ValueError("quantile_levels must not be empty.")

        levels = np.asarray(quantile_levels, dtype=float)
        if levels.ndim != 1:
            raise ValueError(f"quantile_levels must be one-dimensional, got shape {levels.shape}.")
        if not np.isfinite(levels).all():
            raise ValueError(f"All quantile_levels must be finite, got {list(quantile_levels)}.")
        if np.any((levels <= 0.0) | (levels >= 1.0)):
            raise ValueError(f"All quantile_levels must lie strictly between 0 and 1, got {levels.tolist()}.")
        if len(np.unique(levels)) != len(levels):
            raise ValueError(f"quantile_levels must be unique, got {levels.tolist()}.")

        if isinstance(acquisition_function, LCB) and np.any(levels > _POSTERIOR_MEDIAN_QUANTILE):
            raise ValueError(
                "LCB represents lower posterior quantiles for minimization; "
                f"all quantile_levels must be <= 0.5, got {levels.tolist()}."
            )
        if isinstance(acquisition_function, UCB) and np.any(levels < _POSTERIOR_MEDIAN_QUANTILE):
            raise ValueError(
                "UCB represents upper posterior quantiles for minimization; "
                f"all quantile_levels must be >= 0.5, got {levels.tolist()}."
            )

        return levels.tolist()

    def _create_action(self) -> ParameterAction:
        """Create one categorical action per configured posterior quantile."""
        return ParameterAction(
            attr="_posterior_quantile",
            space=Discrete(n=len(self._quantile_levels)),
            log=False,
        )

    @property
    def quantile_levels(self) -> tuple[float, ...]:
        """Configured quantiles in action-index order."""
        return tuple(self._quantile_levels)

    @property
    def current_quantile(self) -> float:
        """Currently installed posterior quantile."""
        return self._current_quantile

    @property
    def current_control_value(self) -> float:
        """Currently installed posterior quantile for generic observations."""
        return self._current_quantile

    @property
    def bound_type(self) -> str:
        """Confidence-bound orientation, either ``"LCB"`` or ``"UCB"``."""
        return self._bound_type

    def update_optimizer(self, action: ActType) -> None:
        """Set confidence-bound beta to the quantile addressed by ``action``."""
        action_idx = int(np.asarray(action).item())
        if not self.space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} is outside {self.space}.")

        quantile = self._quantile_levels[action_idx]
        acquisition_function = self._smac_instance._intensifier._config_selector._acquisition_function
        beta = float(norm.ppf(quantile) ** 2 / acquisition_function._nu)
        acquisition_function._beta = beta
        acquisition_function._posterior_quantile = quantile
        self._current_quantile = quantile


ConfidenceBoundQuantileActionSpace = PosteriorQuantileActionSpace


class AcqParameterActionSpace(AbstractActionSpace):
    """Action space for tuning parameters of the current acquisition function.

    Attributes
    ----------
    _PARAMETERS : ClassVar[dict[type[AbstractAcquisitionFunction], ParameterAction]]
        Mapping of acquisition function classes to their corresponding parameter actions.
    """

    _ATTRIBUTE_MAP: ClassVar[dict[type[AbstractAcquisitionFunction], str]] = {
        EI: "_xi",
        PI: "_xi",
        LCB: "_beta",
        UCB: "_beta",
        WEI: "_alpha",
    }
    _LOG: ClassVar[dict[type[AbstractAcquisitionFunction], bool]] = {
        EI: False,
        PI: False,
        LCB: True,
        UCB: True,
        WEI: False,
    }

    def __init__(
        self,
        smac_instance: SMBO,
        bounds: tuple[int, int] | tuple[float, float],
        adjustment_type: str = "continuous",
        step_size: float = 0.5,
    ) -> None:
        """Initialize action space.

        Parameters
        ----------
        smac_instance : SMBO
            The smac instance.
        bounds : tuple[int, int] | tuple[float, float]
            The action space bounds (low, high). If the acquisition function hyperparameter should be adjusted in log
            space, it is assumed that the bounds already are in log space.
            For EI and PI, usually the bounds are (-10, 10). For UCB: -6 to 3 in log10 space
            (for continuous and bucket).
        adjustment_type : str, optional
            The adjustment, by default "continuous". Can be continuous, bucket or step.
            For bucket, we have discrete choices with bounds as bounds.
            For step, the lower bound is interpreted as the
            decrease (but put a negative number as everything is just added), the upper as increase, and there will be
            a do nothing action.
        step_size : float, optional
            If the adjustment type is step, we have as actions: decrease, do nothing, increase. For the amount of
            decrease/increase we need to specify the step size.
        """
        self._last: float = 0.0
        self._adjustment_type = adjustment_type
        self._bounds = bounds
        self._step_size = step_size
        super().__init__(smac_instance)

    def _create_action(self) -> ParameterAction:
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
        if isinstance(acquisition_function, LCB | UCB) and acquisition_function._update_beta:
            raise ValueError(
                "For LCB/UCB we can only adjust beta and for this, `_update_beta` must be set to False. "
                "If you mean to adjust nu, please add this in the code."
            )

        attribute = self._ATTRIBUTE_MAP[type(acquisition_function)]
        is_log = self._LOG[type(acquisition_function)]

        if self._adjustment_type in {"continuous", "continuousstep"}:
            dacbo_action_space = ParameterAction(
                attr=attribute, space=Box(low=self._bounds[0], high=self._bounds[1], dtype=np.float32), log=is_log
            )
        elif self._adjustment_type == "step":
            dacbo_action_space = ParameterAction(attr=attribute, space=Discrete(n=3), log=is_log)
        elif self._adjustment_type == "bucket":
            if not isinstance(self._bounds[0], int):
                raise ValueError(
                    "Expected self._bounds[0] to be int for 'bucket' adjustment type, "
                    f"got {type(self._bounds[0]).__name__}"
                )
            if not isinstance(self._bounds[1], int):
                raise ValueError(
                    "Expected self._bounds[1] to be int for 'bucket' adjustment type, "
                    f"got {type(self._bounds[1]).__name__}"
                )
            n = abs(self._bounds[0]) + self._bounds[1] + 1
            dacbo_action_space = ParameterAction(attr=attribute, space=Discrete(n=n), log=is_log)
        else:
            raise ValueError(f"Unknown adjustment type: {self._adjustment_type}.")

        return dacbo_action_space

    def update_optimizer(self, action: ActType) -> None:
        """Update the acquisition function parameter value.

        Parameters
        ----------
        action : ActType
            A single numeric action value for the parameter.
        """
        action_val = np.array(action).item()

        if self._adjustment_type == "continuous":
            param_val = action_val
        elif self._adjustment_type == "continuousstep":
            self._last = np.clip(self._last + action_val, self._bounds[0], self._bounds[1])
            param_val = self._last
        elif self._adjustment_type == "step":
            if action_val == 0:
                self._last -= self._step_size
            elif action_val == 1:
                self._last = self._last
            elif action_val == 2:  # noqa: PLR2004
                self._last += self._step_size

            self._last = np.clip(self._last, self._bounds[0], self._bounds[1])
            param_val = self._last
        elif self._adjustment_type == "bucket":
            param_val = action_val + self._bounds[0]  # that value probably is below 0 so basically the offset

        if self._action.log:  # type: ignore[union-attr]
            param_val = 10**param_val

        setattr(
            self._smac_instance._intensifier._config_selector._acquisition_function,
            self._action.attr,  # type: ignore[union-attr]
            param_val,
        )


class AcqFunctionActionSpace(AbstractActionSpace):
    """Action space for selecting an acquisition function.

    Attributes
    ----------
    _acq_fun_dict : ClassVar[dict[int, type[AbstractAcquisitionFunction]]]
        Mapping of integer IDs to available acquisition function classes.
    """

    def __init__(
        self, smac_instance: SMBO, acquisition_functions: list[AbstractAcquisitionFunction] | None = None
    ) -> None:
        """Initialize discrete acquisition function choice space.

        Parameters
        ----------
        smac_instance : SMBO
            The smac instance.
        acquisition_functions : list[AbstractAcquisitionFunction] | None, optional
            List of acquisition function classes, by default None. If None, will be [EI, PI, UCB].
        """
        _afs = [EI, PI, UCB] if acquisition_functions is None else acquisition_functions
        self._acq_fun_dict = dict(enumerate(_afs))
        super().__init__(smac_instance)

    def _create_action(self) -> FunctionAction:
        """Create a FunctionAction representing the discrete selection of acquisition functions.

        Returns
        -------
        FunctionAction
            The FunctionAction object for acquisition function selection.
        """
        return FunctionAction(Discrete(len(self._acq_fun_dict)))

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer to use the selected acquisition function.

        Parameters
        ----------
        action : ActType
            Integer index representing the selected acquisition function.
        """
        function_idx = int(np.array(action).item())
        self._smac_instance.update_acquisition_function(acquisition_function=self._acq_fun_dict[function_idx]())


class PosteriorModeActionSpace(AbstractActionSpace):
    """Choose one of five deterministic operations on a shared posterior.

    Unlike :class:`AcqFunctionActionSpace`, this action space does not replace
    SMAC's acquisition object. It changes the active operation of one
    :class:`~dacboenv.utils.posterior_decision.PosteriorModeAcquisition`, so
    the model, incumbent update, and acquisition maximizer remain synchronized.
    """

    def __init__(self, smac_instance: SMBO) -> None:
        self._mode_names = POSTERIOR_MODE_NAMES
        acquisition_function = smac_instance._intensifier._config_selector._acquisition_function
        if not isinstance(acquisition_function, PosteriorModeAcquisition):
            raise TypeError(
                "PosteriorModeActionSpace requires PosteriorModeAcquisition, "
                f"got {type(acquisition_function).__name__}."
            )
        super().__init__(smac_instance)

    def _create_action(self) -> FunctionAction:
        """Create one categorical action for each posterior operation."""
        return FunctionAction(Discrete(n=len(self._mode_names)))

    @property
    def mode_names(self) -> tuple[str, ...]:
        """Posterior operations in action-index order."""
        return self._mode_names

    @property
    def selected_mode(self) -> str:
        """Currently selected posterior operation."""
        acquisition_function = self._smac_instance._intensifier._config_selector._acquisition_function
        assert isinstance(acquisition_function, PosteriorModeAcquisition)
        return acquisition_function.mode

    @property
    def current_action_index(self) -> int:
        """Index of the currently selected posterior operation."""
        return self._mode_names.index(self.selected_mode)

    @property
    def normalized_action(self) -> float:
        """Current action index normalized to ``[0, 1]``."""
        return self.current_action_index / (len(self._mode_names) - 1)

    @property
    def current_control_value(self) -> float:
        """Current normalized action index for generic observations."""
        return self.normalized_action

    def update_optimizer(self, action: ActType) -> None:
        """Select the posterior operation addressed by ``action``."""
        action_idx = int(np.asarray(action).item())
        if not self.space.contains(action_idx):
            raise ValueError(f"Action index {action_idx} is outside {self.space}.")

        selector = self._smac_instance._intensifier._config_selector
        acquisition_function = selector._acquisition_function
        assert isinstance(acquisition_function, PosteriorModeAcquisition)
        selected_mode = self._mode_names[action_idx]
        if acquisition_function.mode != selected_mode:
            acquisition_function.mode = selected_mode
            # ConfigSelector caches acquisition updates by runhistory size. A
            # mode switch must receive eta/model before the next maximize even
            # when no new observation has arrived since the previous ask.
            selector._previous_entries = -1
