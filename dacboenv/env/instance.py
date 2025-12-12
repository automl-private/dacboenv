"""Instance Selection for DACBO Env."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod

import numpy as np


class InstanceSelector(ABC):
    """Instance Selector.

    One instance is represented as (task_id, seed).
    The list of instances is [(seed_0, task_id_0), (seed_0, task_id_1), ..., (seed_1, task_id_0),]

    Attributes
    ----------
    task_ids : list[str]
        List of carps task ids.
    seeds : list[int]
        List of seeds.
    idx : int
        Current instance index. Default is 0.
    rng : Generator
        Random generator.
    """

    def __init__(self, task_ids: list[str], seeds: list[int], selector_seed: int | None = None) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int]
            List of seeds.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        self.task_ids = task_ids
        self.seeds = seeds
        self.instances = list(itertools.product(self.seeds, self.task_ids))
        self.idx: int = 0
        self.rng = np.random.default_rng(seed=selector_seed)

    @abstractmethod
    def select_instance(self) -> tuple[int, str]:
        """Select next instance.

        Returns
        -------
        tuple[int,str]
            (seed, task_id)
        """


class RoundRobinInstanceSelector(InstanceSelector):
    """Round robin instance selector.

    Rotate through instances.
    """

    def select_instance(self) -> tuple[int, str]:
        """Select next instance.

        Returns
        -------
        tuple[int,str]
            (seed, task_id)
        """
        n_instances = len(self.instances)
        instance = self.instances[self.idx]
        self.idx = (self.idx + 1) % n_instances
        return instance


class RandomInstanceSelector(InstanceSelector):
    """Random instance selector."""

    def select_instance(self) -> tuple[int, str]:
        """Select next instance.

        Returns
        -------
        tuple[int,str]
            (seed, task_id)
        """
        return self.rng.choice(self.instances)
