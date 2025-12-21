"""Policy utilities for DACBOEnv."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from dacboenv.env.reward import get_initial_design_size
from dacboenv.utils.math import sigmoid

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import ActType, DACBOEnv, ObsType

import numpy as np
from hydra.utils import get_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class Policy:
    """Abstract base class for DACBOEnv policies.

    A policy defines a mapping from observations to actions within
    the DACBO environment.
    """

    def __init__(self, env: DACBOEnv, **kwargs: Any) -> None:
        """Initialize the policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        **kwargs : Any
            Keyword arguments from child classes.
        """
        self._env = env
        self._init_kwargs = kwargs.copy()

    def get_init_kwargs(self) -> dict:
        """Get kwargs from initialization.

        Requirement is that each child class passes their kwargs to super.
        """
        return self._init_kwargs

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

    def set_seed(self, seed: int | None) -> None:
        """Set seed for stochastic policies.

        Parameters
        ----------
        seed : int | None
            Seed
        """


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

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the action space.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        self._env.action_space.seed(seed=seed)


class DefaultPolicy(Policy):
    """Default policy that does nothing."""

    def __call__(self, obs: ObsType | None = None) -> None:  # noqa: ARG002
        """Returns None.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            None.
        """
        return


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
        super().__init__(env, par_val=par_val)
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
        super().__init__(env, high_to_low=high_to_low, low=low, high=high)
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

    def __init__(self, env: DACBOEnv, low: float, high: float, jump: float, seed: int | None = None) -> None:
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
        super().__init__(env, low=low, high=high, jump=jump, seed=seed)
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
        n_finished = smac.runhistory.finished
        n_initial_design = get_initial_design_size(smac)
        n_smbo = smac._scenario.n_trials
        n_model_based = n_smbo - n_initial_design

        if (n_finished - n_initial_design) < self._jump * n_model_based:
            return self._low
        return self._high


class PiecewiseParameterPolicy(Policy):
    """Policy that sets the parameter based on a fixed piecewise-linear function."""

    def __init__(self, env: DACBOEnv, splits: np.ndarray) -> None:
        """Initialize the jump parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        splits : np.ndarray
            y values of the splits between the linear sections.
        """
        super().__init__(env, splits=splits)
        self._splitsy = splits

    def __call__(self, obs: ObsType | None = None) -> ActType:  # noqa: ARG002
        """Return the parameter value based on progress and piecewise function.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation (unused). Default is None.

        Returns
        -------
        ActType
            Parameter value based on piecewise-linear interpolation.
        """
        smac = self._env._smac_instance
        budget = smac._scenario.n_trials
        trials = len(smac.runhistory)
        splitsx = np.linspace(0, 1, len(self._splitsy), dtype=float) * budget
        val = np.interp(trials, splitsx, self._splitsy)

        return val**2  # XXX: Circumvent squareroot


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
        super().__init__(env, low=low, high=high, jump=jump)
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


class PerceptronPolicy(Policy):
    r"""Perceptron policy.

    Simple form of:
    $\varsigma(\theta^\mathrm{T} s + b)$
    With $\theta$ and $b$ the weights and bias, and $s$ the observation vector.
    Will be squished to an output range of $[0,1]$ due to the sigmoid.
    """

    def __init__(
        self,
        env: DACBOEnv,
        weights: list[float] | None = None,
        theta: list[float] | None = None,
        bias: float | None = None,
        seed: int | None = None,
    ) -> None:
        r"""Init.

        If theta and bias not given, will be sampled uniformly $\sim \mathcal{U}(0,1)$, with
        the seed given in the environment.

        Parameters
        ----------
        env : DACBOEnv
            The DACBO env.
        weights : list[float], optional
            The vector [theta, bias]. Has priority over individual theta and bias definitions.
        theta : list[float], optional
            The weight vector (1d). Can be given instead of weights.
        bias : float, optional
            The bias (scalar). Can be given instead of weights.
        """
        super().__init__(env, weights=weights, theta=theta, bias=bias, seed=seed)

        self._seed = seed
        rng = np.random.default_rng(seed=self._seed)

        if weights is not None:
            theta = weights[:-1]
            bias = weights[-1]
        if theta is None:
            n_obs = env.observation_space.shape[0]
            theta = rng.uniform(size=n_obs)
        if bias is None:
            bias = rng.uniform()

        self._theta = np.array(theta)
        self._bias = bias

    def __call__(self, obs: dict[str, Any]) -> int | float | list[float]:
        """Apply policy.

        Parameters
        ----------
        obs : dict[str, Any]
            The current observation.

        Returns
        -------
        int | float | list[float]
            The action in the interval [0,1].
        """
        obs_arr = np.array(list(obs.values()))
        signal = self._theta.T @ obs_arr + self._bias
        return float(sigmoid(signal))


class ModelPolicy(Policy):
    """Policy that uses a pre-trained RL model to select actions."""

    def __init__(
        self,
        env: DACBOEnv,
        model: BaseAlgorithm | str,
        model_class: type[BaseAlgorithm] | str | None = None,
        normalization_wrapper: str | None = None,
    ) -> None:
        """Initialize the model parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        model : BaseAlgorithm | str
            The RL model instance or path to a saved model.
        model_class : type[BaseAlgorithm] | str | None, optional
            The class of the RL model, required if loading from a path.
        normalization_wrapper : str | None, optional
            Path to a saved VecNormalize wrapper, if applicable.
        """
        super().__init__(env, model=model, model_class=model_class, normalization_wrapper=normalization_wrapper)

        vec_env = DummyVecEnv([lambda: env])

        if normalization_wrapper is not None:
            vec_env = VecNormalize.load(normalization_wrapper, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        self._vec_env = vec_env

        if isinstance(model, str):
            assert model_class is not None, "If model is loaded from path, model_class must be provided."
            model_class = model_class if isinstance(model_class, type) else get_class(model_class)
            self._model = model_class.load(model, env=self._vec_env)
        else:
            self._model = model
            self._model.set_env(self._vec_env)

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
        if isinstance(self._vec_env, VecNormalize):
            obs = self._vec_env.normalize_obs(obs)
        return self._model.predict(obs, deterministic=True)[0]

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the model.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        self._model.set_random_seed(seed=seed)
