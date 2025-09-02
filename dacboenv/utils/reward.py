"""Reward utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

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


def _sum_costs(costs: float | list[float]) -> float:
    """Sum costs for single- or multi-objective cases.

    Parameters
    ----------
    costs : float or list of float
        Cost(s) to sum.

    Returns
    ----------
    float
        The summed cost.
    """
    return costs if isinstance(costs, float) else sum(costs)  # TODO: Also ParEGO?


incumbent_cost_reward = RewardType(
    "incumbent_cost", lambda smbo: abs(_sum_costs(smbo.intensifier.trajectory[-1].costs))
)
incumbent_improvement_reward = RewardType(
    "incumbent_improvement",
    lambda smbo: 0
    if smbo.intensifier.trajectory[-1].trial != len(smbo.runhistory)
    else abs(_sum_costs(smbo.intensifier.trajectory[-1].costs) - _sum_costs(smbo.intensifier.trajectory[-2].costs))
    if len(smbo.intensifier.trajectory) > 1
    else abs(_sum_costs(smbo.intensifier.trajectory[-1].costs)),
)
cum_cost_reward = RewardType(
    "cum_cost",
    lambda smbo: -sum(_sum_costs(v.cost) for v in smbo.runhistory.values()),
)

ALL_REWARDS = [incumbent_cost_reward, incumbent_improvement_reward, cum_cost_reward]


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
    ----------
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
        ----------
        ValueError
            If any provided keys are not valid reward names.
        """
        self._smac_instance = smac_instance
        self._rho = rho

        # Default to all possible keys if not provided
        self._keys = keys if keys is not None else list(DACBOReward._REWARD_MAP.keys())

        # Check for invalid keys
        invalid_keys = set(self._keys) - set(DACBOReward._REWARD_MAP.keys())
        if invalid_keys:
            raise ValueError(f"Invalid reward keys: {invalid_keys}")

        self._reward_types = [DACBOReward._REWARD_MAP[key] for key in self._keys]
        self._parego = ParEGO(len(self._reward_types), self._smac_instance._scenario.seed, self._rho)

    def get_reward(self) -> float:
        """Compute the (scalarized) reward from the selected reward types.

        Returns
        ----------
        float
            The computed reward value.
        """
        # Multi-objective using ParEGO
        return self._parego([rew.compute(self._smac_instance) for rew in self._reward_types])
