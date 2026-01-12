"""Policy utilities for DACBOEnv."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from ConfigSpace import ConfigurationSpace, Float
from hydra.utils import get_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from torch import nn

from dacboenv.dacboenv import DACBOEnv, ObsType
from dacboenv.env.reward import get_initial_design_size
from dacboenv.policy.abstract_policy import AbstractPolicy
from dacboenv.utils.math import sigmoid

if TYPE_CHECKING:
    from typing import Any, TypeAlias

    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import ActType, DACBOEnv, ObsType

Policy: TypeAlias = AbstractPolicy


class LinearParameterPolicy(AbstractPolicy):
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


class JumpParameterPolicy(AbstractPolicy):
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


class PiecewiseParameterPolicy(AbstractPolicy):
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


class JumpFunctionPolicy(AbstractPolicy):
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


class PerceptronPolicy(AbstractPolicy):
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


def get_nweights_alpharulenet() -> int:
    """Get the numer of weights of AlphaRuleNet.

    Returns
    -------
    int
        The number of weights.
    """
    return AlphaRuleNet.n_weights


# TODO add torch no grad
class AlphaRuleNet(nn.Module):
    """Alpha Rule Net.

    Inspired by an analytical form of SAWEI.
    """

    n_weights = 57

    def __init__(self, delta_alpha: float = 0.1, k: float = 10.0, weights: list[float] | None = None):
        """Initialize.

        Parameters
        ----------
        env : DACBOEnv
            The DACBO environment.
        delta_alpha : float, optional
            The amount of adjustment of alpha, same in SAWEI paper, by default 0.1
        k : float, optional
            How steep the comparison between the acq fun values of PI and EI should be, by default 10
        weights : list[float] | None, optional
            The weight vector, by default None. If None, initialize to the approximation of SAWEI rule.
        """
        super().__init__()
        self.delta_alpha = delta_alpha
        self.k = k

        # 5 inputs (R, v_PI, v_EI, alpha_prev, R_scale)
        self.fc1 = nn.Linear(5, 8)
        self.fc2 = nn.Linear(8, 1)

        if weights is None:
            # Ensures very rough SAWEI behavior
            # Default preinitialization (as before)
            nn.init.constant_(self.fc2.bias, 0.0)
            self.fc1.weight.data.zero_()
            self.fc1.bias.data.zero_()
            # Neurons 0-3 approximate tanh(k*(v_EI - v_PI))
            self.fc1.weight.data[0, 1] = -k  # v_PI
            self.fc1.weight.data[1, 2] = k  # v_EI
            self.fc1.weight.data[2, 1] = -0.5 * k
            self.fc1.weight.data[2, 2] = 0.5 * k
            self.fc1.weight.data[3, 1] = -0.5 * k
            self.fc1.weight.data[3, 2] = 0.5 * k
            # Neurons 4-7 approximate Gaussian gate using R / R_scale
            self.fc1.weight.data[4, 0] = -1.0
            self.fc1.weight.data[4, 4] = 1.0
            self.fc1.weight.data[5, 0] = 1.0
            self.fc1.weight.data[5, 4] = -1.0
            self.fc1.weight.data[6, 0] = -0.5
            self.fc1.weight.data[6, 4] = 0.5
            self.fc1.weight.data[7, 0] = 0.5
            self.fc1.weight.data[7, 4] = -0.5
            # Output weights
            self.fc2.weight.data.fill_(delta_alpha * 0.125)
        else:
            # Flatten all parameters into a single vector
            params = list(self.fc1.parameters()) + list(self.fc2.parameters())
            # Count total elements
            total_params = sum(p.numel() for p in params)
            assert len(weights) == total_params, f"Expected {total_params} floats, got {len(weights)}"
            # Copy values
            offset = 0
            for p in params:
                n = p.numel()
                p.data.copy_(torch.tensor(weights[offset : offset + n], dtype=p.dtype).view_as(p))
                offset += n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference.

        Parameters
        ----------
        x : torch.Tensor
            Input (obs: R, v_PI, v_EI, alpha_prev, R_scale)

        Returns
        -------
        torch.Tensor
            Output (alpha).
        """
        x[:, 0:1]
        x[:, 1:2]
        x[:, 2:3]
        alpha_prev = x[:, 3:4]
        x[:, 4:5]

        # Hidden layer
        h = torch.tanh(self.fc1(x))

        # Linear output
        delta_alpha_out = self.fc2(h)

        # Update alpha
        alpha_new = alpha_prev + delta_alpha_out
        return torch.clamp(alpha_new, 0.0, 1.0)

    @classmethod
    def alpha_rule_init_weights(cls: type[AlphaRuleNet], k: float = 10, delta_alpha: float = 0.1) -> torch.Tensor:
        """
        Construct the flattened parameter vector that reproduces the default
        (hand-crafted) initialization of AlphaRuleNet when `weights=None`.

        This function generates the exact 57-element weight vector corresponding
        to the analytical SAWEI-inspired initialization used in the model:
        - fc1.weight: 8x5 matrix encoding comparisons between PI, EI, and scaled R
        - fc1.bias: all zeros
        - fc2.weight: uniform weights equal to delta_alpha / 8
        - fc2.bias: zero

        The returned vector matches the internal PyTorch parameter ordering:
            [fc1.weight, fc1.bias, fc2.weight, fc2.bias]

        Parameters
        ----------
        k : float
            Steepness parameter controlling sensitivity to the difference
            between acquisition values (e.g., EI vs PI).
        delta_alpha : float
            Step size scaling factor applied to the output of the network.

        Returns
        -------
        torch.Tensor
            A 1D tensor of length 57 containing the initialized network parameters.
            This tensor can be passed directly as the `weights` argument when
            constructing an `AlphaRuleNet`.
        """
        weights: list[float] = []

        # ---- fc1.weight (8 x 5) ----
        fc1_weight: list[list[float]] = [
            [0, -k, 0, 0, 0],
            [0, 0, k, 0, 0],
            [0, -0.5 * k, 0.5 * k, 0, 0],
            [0, -0.5 * k, 0.5 * k, 0, 0],
            [-1, 0, 0, 0, 1],
            [1, 0, 0, 0, -1],
            [-0.5, 0, 0, 0, 0.5],
            [0.5, 0, 0, 0, -0.5],
        ]

        for row in fc1_weight:
            weights.extend(row)

        # ---- fc1.bias (8) ----
        weights.extend([0.0] * 8)

        # ---- fc2.weight (1 x 8) ----
        weights.extend([delta_alpha * 0.125] * 8)

        # ---- fc2.bias (1) ----
        weights.append(0.0)

        return torch.tensor(weights, dtype=torch.float32)


class AlphaRulePolicy(AbstractPolicy):
    """AlphaRulePolicy.

    Expects the sawei observations of ubr_difference, acq_value_PI, acq_value_EI, previous_param.
    Interface to DACBOEnv.
    """

    def __init__(
        self,
        env: DACBOEnv,
        alpha_start: float = 0.5,
        delta_alpha: float = 0.1,
        k: float = 10,
        weights: list[float] | None = None,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        env : DACBOEnv
            The DACBO environment.
        alpha_start : float, optional
            The start value of alpha, by default 0.5. This is basically EI.
        delta_alpha : float, optional
            The amount of adjustment of alpha, same in SAWEI paper, by default 0.1
        k : float, optional
            How steep the comparison between the acq fun values of PI and EI should be, by default 10
        weights : list[float] | None, optional
            The weight vector, by default None. If None, initialize to the approximation of SAWEI rule.
        """
        super().__init__(env)
        self.delta_alpha = delta_alpha
        self.alpha_start = alpha_start
        self.k = k
        self.weights = weights

        self.net = AlphaRuleNet(delta_alpha=delta_alpha, k=k, weights=self.weights)
        self._ubr_diffs: list[float] = []

    def __call__(self, obs: dict[str, Any]) -> int | float | list[float] | None:
        """Infer action based on observations.

        Calculate/track the scale of the difference of the UBR here.

        Parameters
        ----------
        obs : dict[str, Any]
            The observations.

        Returns
        -------
        int | float | list[float] | None
            The action, the WEI alpha parameter.
        """
        self._ubr_diffs.append(obs["ubr_difference"])
        ubr_diff_std = np.std(self._ubr_diffs)
        if np.isnan(ubr_diff_std):
            ubr_diff_std = 1
        previous_param = float(obs["previous_param"]) if obs["previous_param"] is not None else self.alpha_start
        x_list = [
            float(obs["ubr_difference"]),
            float(obs["acq_value_PI"]),
            float(obs["acq_value_EI"]),
            previous_param,
            float(ubr_diff_std),
        ]
        x = torch.tensor([x_list], dtype=torch.float32)  # batch
        y = torch.squeeze(self.net(x).detach().cpu())
        return float(y)

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the model.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        torch.manual_seed(seed)

    @classmethod
    def get_alpharulenet_configspace(
        cls, weight_bounds: tuple[float, float], k: float = 10, delta_alpha: float = 0.1
    ) -> ConfigurationSpace:
        """Get configuration space for AlphaRuleNet policy.

        Parameters
        ----------
        weight_bounds : tuple[float,float]
            The weight bounds.

        Returns
        -------
        ConfigurationSpace
            The configuration space, contaings n_obs + 1 hyperparameters (weight vector and bias).
        """
        n_hps = AlphaRuleNet.n_weights
        defaults = AlphaRuleNet.alpha_rule_init_weights(k=k, delta_alpha=delta_alpha)
        configspace = ConfigurationSpace()
        configspace.add([Float(name=f"w{i}", bounds=weight_bounds, default=float(defaults[i])) for i in range(n_hps)])
        return configspace


class ModelPolicy(AbstractPolicy):
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
