"""Reward utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.metrics import auc

from dacboenv.utils.parego import ParEGO

if TYPE_CHECKING:
    from smac.main.smbo import SMBO


@dataclass
class RewardType:
    """Represents a single reward type for the DACBO environment.

    Attributes
    ----------
    name : str
        Name of the reward.
    compute : Callable[[SMBO], Any]
        Function to compute the reward value from a SMAC instance.
    """

    name: str
    compute: Callable[[SMBO], Any]


# Multi-objective: Handle incumbent cost

auc_reward = RewardType(
    "trajectory_auc",
    lambda smbo: -auc([t.trial for t in smbo.intensifier.trajectory], costs)
    if len(costs := [t.costs[-1] - smbo.intensifier.trajectory[0].costs[-1] for t in smbo.intensifier.trajectory]) > 1
    else 0,
)
incumbent_cost_reward = RewardType(
    "incumbent_cost", lambda smbo: -smbo.intensifier.trajectory[-1].costs[-1]
)  # Minimize cost
incumbent_improvement_reward = RewardType(
    "incumbent_improvement",
    lambda smbo: abs(smbo.intensifier.trajectory[-1].costs[-1] - smbo.intensifier.trajectory[-2].costs[-1])
    if len(smbo.intensifier.trajectory) > 1 and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
sqrt_incumbent_improvement_reward = RewardType(
    "sqrt_incumbent_improvement",
    lambda smbo: np.sqrt(abs(smbo.intensifier.trajectory[-1].costs[-1] - smbo.intensifier.trajectory[-2].costs[-1]))
    if len(smbo.intensifier.trajectory) > 1 and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
auc_reward_alt = RewardType(
    "trajectory_auc_alt",
    lambda smbo: -auc(
        range(len(smbo.runhistory)),
        np.minimum.accumulate([t.cost - smbo.intensifier.trajectory[0].costs[-1] for t in smbo.runhistory.values()]),
    )
    if len(smbo.runhistory) > 1
    else 0,
)

ALL_REWARDS = [incumbent_improvement_reward]  # [incumbent_cost_reward, incumbent_improvement_reward, auc_reward]


class DACBOReward:
    """Manages a collection of reward types and computes (possibly multi-objective) rewards.

    Supports scalarization of multiple reward objectives using ParEGO.

    Parameters
    ----------
    smac_instance : SMBO
        The SMAC optimizer instance.
    keys : list[str], optional
        List of reward names to include. If None, all available rewards are used.
    rho : float, optional
        ParEGO scalarization parameter (default: 0.05).

    Attributes
    ----------
    _reward_types : list[RewardType]
        The selected reward types.
    _parego : ParEGO
        ParEGO scalarization utility.

    Methods
    -------
    get_reward() -> float
        Computes the (scalarized) reward from the selected reward types.
    """

    _REWARD_MAP: ClassVar[dict[str, RewardType]] = {rew.name: rew for rew in ALL_REWARDS}

    def __init__(self, smac_instance: SMBO, keys: list[str] | None = None, rho: float = 0.05) -> None:
        """Initialize the DACBOReward.

        Parameters
        ----------
        smac_instance : SMBO
            The SMAC optimizer instance.
        keys : list[str], optional
            List of reward names to include. If None, all available rewards are used.
        rho : float, optional
            ParEGO scalarization parameter (default: 0.05).

        Raises
        ------
        ValueError
            If any provided keys are not valid reward names.
        """
        self._smac_instance = smac_instance
        self._rho = rho

        # Default to all possible keys if not provided
        self._keys = keys if keys is not None else list(self._REWARD_MAP.keys())

        # Check for invalid keys
        invalid_keys = set(self._keys) - set(self._REWARD_MAP.keys())
        if invalid_keys:
            raise ValueError(f"Invalid reward keys: {invalid_keys}")

        self._reward_types = [self._REWARD_MAP[key] for key in self._keys]

        self._parego = ParEGO(len(self._reward_types), self._smac_instance._scenario.seed, self._rho)

    def _get_full_reward(self) -> dict[str, float]:
        """Compute all sub-rewards from the selected reward types.

        Returns
        -------
        dict[str, float]
            All sub-rewards.
        """
        if len(self._reward_types) == 1:
            return self._reward_types[0].compute(self._smac_instance)
        return {rew.name: rew.compute(self._smac_instance) for rew in self._reward_types}

    def get_reward(self) -> float:
        """Compute the (scalarized) reward from the selected reward types.

        Returns
        -------
        float
            The computed reward value.
        """
        # Multi-objective using ParEGO
        return self._parego([rew.compute(self._smac_instance) for rew in self._reward_types])
