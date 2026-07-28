"""Reward utilities for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from sklearn.metrics import auc
from smac.runhistory.enumerations import StatusType

from dacboenv.utils.math import symlog
from dacboenv.utils.parego import ParEGO

if TYPE_CHECKING:
    from smac.main.smbo import SMBO

MIN_TRANSITION_POINTS = 2


@dataclass
class RewardType:
    """Represents a single reward type for the DACBO environment.

    Attributes
    ----------
    name : str
        Name of the reward.
    compute : Callable[[SMBO], Any]
        Function to compute the reward value from a SMAC instance and from
        reference_performance: float | None.
    """

    name: str
    compute: Callable[[SMBO, float | None], Any]


# Multi-objective: Handle incumbent cost

auc_reward = RewardType(
    "trajectory_auc",
    lambda smbo, reference_performance: -auc([t.trial for t in smbo.intensifier.trajectory], costs)  # noqa: ARG005
    if len(costs := [t.costs[-1] - smbo.intensifier.trajectory[0].costs[-1] for t in smbo.intensifier.trajectory]) > 1
    else 0,
)
incumbent_cost_reward = RewardType(
    "incumbent_cost",
    lambda smbo, reference_performance: -smbo.intensifier.trajectory[-1].costs[-1],  # noqa: ARG005
)  # Minimize cost
incumbent_improvement_reward = RewardType(
    "incumbent_improvement",
    lambda smbo, reference_performance: abs(  # noqa: ARG005
        smbo.intensifier.trajectory[-1].costs[-1] - smbo.intensifier.trajectory[-2].costs[-1]
    )
    if len(smbo.intensifier.trajectory) > 1 and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
sqrt_incumbent_improvement_reward = RewardType(
    "sqrt_incumbent_improvement",
    lambda smbo, reference_performance: np.sqrt(  # noqa: ARG005
        abs(smbo.intensifier.trajectory[-1].costs[-1] - smbo.intensifier.trajectory[-2].costs[-1])
    )
    if len(smbo.intensifier.trajectory) > 1 and smbo.intensifier.trajectory[-1].trial == len(smbo.runhistory)
    else 0,
)
auc_reward_alt = RewardType(
    "trajectory_auc_alt",
    lambda smbo, reference_performance: -auc(  # noqa: ARG005
        range(len(smbo.runhistory)),
        np.minimum.accumulate([t.cost - smbo.intensifier.trajectory[0].costs[-1] for t in smbo.runhistory.values()]),
    )
    if len(smbo.runhistory) > 1
    else 0,
)


def get_initial_design_size(solver: SMBO) -> int:
    """Get the size of the initial design.

    Parameters
    ----------
    solver : smac.main.smbo.SMBO
        The optimizer.

    Returns
    -------
    int
        Initial design size.
    """
    return len(solver.intensifier.config_selector._initial_design_configs)


def _get_ordered_scalar_costs(smbo: SMBO) -> np.ndarray:
    """Return single-objective costs in run-history insertion order.

    Failed/non-finite trials are represented as ``+inf`` rather than removed,
    because removing a trial would shift the transition timeline and could
    emit an earlier improvement reward more than once.
    """
    costs: list[float] = []
    for trial_value in smbo.runhistory._data.values():
        if getattr(trial_value, "status", StatusType.SUCCESS) != StatusType.SUCCESS:
            costs.append(np.inf)
            continue

        values = np.asarray(trial_value.cost, dtype=float).reshape(-1)
        if values.size == 0:
            costs.append(np.inf)
            continue

        value = float(values[0])
        costs.append(value if np.isfinite(value) else np.inf)

    return np.asarray(costs, dtype=float)


def _initial_design_location_and_scale(initial_costs: np.ndarray) -> tuple[float, float]:
    """Return robust, episode-fixed normalization for initial-design costs."""
    finite_costs = np.asarray(initial_costs, dtype=float).reshape(-1)
    finite_costs = finite_costs[np.isfinite(finite_costs)]
    if finite_costs.size == 0:
        return 0.0, 1.0

    location = float(np.median(finite_costs))
    mad = 1.4826 * float(np.median(np.abs(finite_costs - location)))
    q25, q75 = np.quantile(finite_costs, [0.25, 0.75])
    iqr = float(q75 - q25) / 1.349
    std = float(np.std(finite_costs))
    scale = max(mad, iqr, std)

    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        scale = max(abs(location), 1.0)

    return location, scale


def calc_reference_free_improvement(
    smbo: SMBO,
    reference_performance: float | None = None,  # noqa: ARG001
) -> float:
    """Return normalized potential improvement from the latest evaluation.

    For incumbent ``b_t``, initial-design median ``m_0`` and scale ``s_0``,
    the potential is ``P_t = -asinh((b_t - m_0) / s_0)`` and the reward is
    ``P_t - P_{t-1}``. The undiscounted episode return therefore telescopes to
    final transformed incumbent quality without a true optimum or reference
    optimizer.
    """
    costs = _get_ordered_scalar_costs(smbo)
    n_initial = get_initial_design_size(smbo)
    if costs.size <= n_initial or costs.size < MIN_TRANSITION_POINTS:
        return 0.0

    initial_costs = costs[: min(n_initial, costs.size)]
    if not np.isfinite(initial_costs).any():
        return 0.0

    previous_incumbent = float(np.min(costs[:-1]))
    current_incumbent = float(min(previous_incumbent, costs[-1]))
    if (
        not np.isfinite(previous_incumbent)
        or not np.isfinite(current_incumbent)
        or current_incumbent >= previous_incumbent
    ):
        return 0.0

    location, scale = _initial_design_location_and_scale(initial_costs)
    previous_potential = -np.arcsinh((previous_incumbent - location) / scale)
    current_potential = -np.arcsinh((current_incumbent - location) / scale)
    reward = float(current_potential - previous_potential)
    return reward if np.isfinite(reward) else 0.0


reference_free_improvement_reward = RewardType(
    "reference_free_improvement",
    calc_reference_free_improvement,
)


def get_reward_for_episode_finished(
    smbo: SMBO,
    reference_performance: float | None = None,  # noqa: ARG001
    scale_by_budget: bool = False,  # noqa: FBT001, FBT002
) -> float:
    """Get reward (or rather punishment: -1) as long the episode is not finished.

    Typically, the episode is finished after DACBO has reached reference performance.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.
    scale_by_budget : bool, optional
        Whether to scale by the model-based budget, by default False. If yes, return -1/b.

    Returns
    -------
    float
        Reward value: -1 if the episode is not finished, or -1 divided by the model-based budget
        if `scale_by_budget` is True.
    """
    if not scale_by_budget:
        return -1

    n_initial_design = get_initial_design_size(smbo)
    n_smbo = smbo._scenario.n_trials
    n_model_based = n_smbo - n_initial_design

    return -1 / n_model_based


episode_finished = RewardType("episode_finished", get_reward_for_episode_finished)

episode_finished_scaled = RewardType(
    "episode_finished_scaled", partial(get_reward_for_episode_finished, scale_by_budget=True)
)


def calc_symlogregret_of_reference_performance(smbo: SMBO, reference_performance: float | None = None) -> float:
    """Calculate the symmetric log regret to the reference performance.

    Parameters
    ----------
    smbo : SMBO
        The SMAC instance.
    reference_performance : float | None, optional
        The reference performance., by default None

    Returns
    -------
    float
        The symlog regret.
    """
    cost_inc = smbo.runhistory.get_min_cost(smbo.intensifier.get_incumbent())
    diff = reference_performance - cost_inc
    return symlog(diff)


symlogregret_reward = RewardType("symlogregret", calc_symlogregret_of_reference_performance)

LEGACY_REWARDS = [
    auc_reward,
    incumbent_cost_reward,
    incumbent_improvement_reward,
    sqrt_incumbent_improvement_reward,
    auc_reward_alt,
    episode_finished,
    episode_finished_scaled,
    symlogregret_reward,
]

ALL_REWARDS = [
    *LEGACY_REWARDS,
    reference_free_improvement_reward,
]


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

        self._keys = (
            keys if keys is not None else [reward.name for reward in LEGACY_REWARDS]
        )

        # Check for invalid keys
        invalid_keys = set(self._keys) - set(self._REWARD_MAP.keys())
        if invalid_keys:
            raise ValueError(f"Invalid reward keys: {invalid_keys}")

        self._reward_types = [self._REWARD_MAP[key] for key in self._keys]

        self._parego = ParEGO(len(self._reward_types), self._smac_instance._scenario.seed, self._rho)
        self._cached_key: tuple[int, float | None] | None = None
        self._cached_reward: float | None = None

    def _get_full_reward(self, reference_performance: float | None = None) -> dict[str, float]:
        """Compute all sub-rewards from the selected reward types.

        Returns
        -------
        dict[str, float]
            All sub-rewards.
        """
        return {rew.name: rew.compute(self._smac_instance, reference_performance) for rew in self._reward_types}

    def get_reward(self, reference_performance: float | None = None) -> float:
        """Compute the (scalarized) reward from the selected reward types.

        Returns
        -------
        float
            The computed reward value.
        """
        cache_key = (
            int(self._smac_instance.runhistory.finished),
            None if reference_performance is None else float(reference_performance),
        )
        if self._cached_key == cache_key and self._cached_reward is not None:
            return self._cached_reward

        full_reward = self._get_full_reward(reference_performance=reference_performance)
        if len(self._reward_types) == 1:
            reward = float(next(iter(full_reward.values())))
        else:
            # Multi-objective using ParEGO
            reward = float(self._parego(list(full_reward.values())))

        self._cached_key = cache_key
        self._cached_reward = reward
        return reward
