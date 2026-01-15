"""Perceptron policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from dacboenv.dacboenv import DACBOEnv
from dacboenv.policy.abstract_policy import AbstractPolicy
from dacboenv.utils.math import sigmoid

if TYPE_CHECKING:
    from dacboenv.dacboenv import DACBOEnv, ObsType


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

    def __call__(self, obs: ObsType) -> int | float | list[float]:
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
