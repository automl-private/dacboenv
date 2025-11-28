"""Policy utilities for DACBOEnv."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import ActType, DACBOEnv, ObsType

from hydra.utils import get_class


class Policy:
    """Abstract base class for DACBOEnv policies.

    A policy defines a mapping from observations to actions within
    the DACBO environment.
    """

    def __init__(self, env: DACBOEnv) -> None:
        """Initialize the policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        """
        self._env = env

    @abstractmethod
    def __call__(self, obs: ObsType) -> ActType:
        """Select an action given the current observation.

        Parameters
        ----------
        obs : ObsType
            The current environment observation.

        Returns
        -------
        ActType
            The selected action.
        """
        raise NotImplementedError


class RandomPolicy(Policy):
    """Policy that samples actions uniformly at random."""

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Select an action by sampling uniformly from the action space.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            A randomly sampled action.
        """
        return self._env.action_space.sample()


class StaticParameterPolicy(Policy):
    """Policy that always returns a fixed parameter value."""

    def __init__(self, env: DACBOEnv, par_val: float) -> None:
        """Initialize the static parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        par_val : float
            Fixed parameter value to return for every action.
        """
        super().__init__(env)
        self._par_val = par_val

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the fixed parameter value.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            The fixed parameter value.
        """
        return self._par_val


class LinearParameterPolicy(Policy):
    """Policy that interpolates linearly between two parameter values
    across the optimization budget.
    """

    def __init__(self, env: DACBOEnv, high_to_low: bool, low: float, high: float) -> None:  # noqa: FBT001
        """Initialize the linear parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        high_to_low : bool
            Whether to start from the high value and decrease to the low value.
        low : float
            Lower bound of the parameter value.
        high : float
            Upper bound of the parameter value.
        """
        super().__init__(env)
        self._high_to_low = high_to_low
        self._low = low
        self._high = high

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return a linearly interpolated parameter value based on the optimization progress.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            Interpolated parameter value depending on the number of completed trials.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)

        weight = trials / budget

        if self._high_to_low:
            return (1 - weight) * self._high + weight * self._low
        return weight * self._high + (1 - weight) * self._low


class JumpParameterPolicy(Policy):
    """Policy that switches from a low to a high parameter value
    after a fraction of the optimization budget.
    """

    def __init__(self, env: DACBOEnv, low: float, high: float, jump: float) -> None:
        """Initialize the jump parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        low : float
            Parameter value before the jump.
        high : float
            Parameter value after the jump.
        jump : float
            Fraction of the optimization budget at which to switch
            from ``low`` to ``high``.
        """
        super().__init__(env)
        self._low = low
        self._high = high
        self._jump = jump

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the parameter value based on progress and jump threshold.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            ``low`` before the jump threshold, otherwise ``high``.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)

        if trials < self._jump * budget:
            return self._low
        return self._high


class JumpFunctionPolicy(Policy):
    """Policy that switches the optimizer's acquisition function
    after a fraction of the optimization budget.
    """

    def __init__(self, env: DACBOEnv, low: int, high: int, jump: float) -> None:
        """Initialize the jump function policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        low : int
            Acquisition function index before the jump.
        high : int
            Acquisition function index after the jump.
        jump : float
            Fraction of the optimization budget at which to switch
            from ``low`` to ``high``.
        """
        super().__init__(env)
        self._low = low
        self._high = high
        self._jump = jump

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the acquisition function index based on progress and jump threshold.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            ``low`` before the jump threshold, otherwise ``high``.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)

        if trials < self._jump * budget:
            return self._low
        return self._high


class ModelPolicy(Policy):
    """Policy that uses a pre-trained RL model to select actions."""

    def __init__(
        self, env: DACBOEnv, model: BaseAlgorithm | str, model_class: type[BaseAlgorithm] | str | None = None
    ) -> None:
        """Initialize the jump parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        model : BaseAlgorithm | str
            The RL model instance or path to a saved model.
        model_class : type[BaseAlgorithm] | str | None, optional
            The class of the RL model, required if loading from a path.
        """
        super().__init__(env)

        if isinstance(model, str):
            assert model_class is not None, "If model is loaded from path, model_class must be provided."
            model_class = model_class if isinstance(model_class, type) else get_class(model_class)
            self._model = model_class.load(model)
        else:
            self._model = model

    def __call__(self, obs: ObsType | None = None) -> ActType:
        """Call the model for the action to take.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation.

        Returns
        -------
        ActType
            Action predicted by the model
        """
        return self._model.predict(obs)[0]
