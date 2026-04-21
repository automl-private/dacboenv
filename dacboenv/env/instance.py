"""Instance Selection for DACBO Env."""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from dataclasses_json import dataclass_json

Instance = tuple[int, str]  # (seed, task_id)
InstanceRelax = tuple[int | None, str]  # (seed, task_id). seed can be None here. Internal type.


@dataclass_json
@dataclass(frozen=True)
class InstanceSet:
    """Instance Set."""

    task_ids: list[str]
    seeds: list[int] | None


class InstanceSelector(ABC):
    """Instance Selector.

    One instance is represented as (task_id, seed).
    The list of instances is [(seed_0, task_id_0), (seed_0, task_id_1), ..., (seed_1, task_id_0),]

    Attributes
    ----------
    task_ids : list[str]
        List of carps task ids.
    seeds : list[int] | list[None]
        List of seeds.
    idx : int
        Current instance index. Default is 0.
    rng : Generator
        Random generator.
    """

    def __init__(self, task_ids: list[str], seeds: list[int] | None, selector_seed: int | None = None) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int] | None
            List of seeds. If None, randomly select a seed based on
            `selector_seed`.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        self.task_ids = task_ids
        self.seeds = seeds if seeds else [None]  # type: ignore[list-item]
        self.instances = list(itertools.product(self.seeds, self.task_ids))
        self.idx: int = 0
        self.selector_seed = selector_seed
        self.rng = np.random.default_rng(seed=selector_seed)

    @abstractmethod
    def _select_instance(self, size: int = 1) -> InstanceRelax | list[InstanceRelax]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        InstanceRelax | list[InstanceRelax]
            (seed, task_id)
            Seed can be None.
        """

    def select_instance(self, size: int = 1) -> Instance | list[Instance]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        Instance | list[Instance]
            (seed, task_id)
            Seed should not be None anymore.
        """
        _instance = self._select_instance(size=size)

        # Create actual seeds if necessary
        if size == 1:
            seed, task_id = _instance
            if seed is None:
                new_seed = int(self.rng.integers(low=0, high=2**32 - 1, size=None))
            else:
                assert isinstance(seed, int)
                new_seed = seed
            instance: Instance = (new_seed, task_id)  # type: ignore[assignment,no-redef]
            return instance
        if isinstance(_instance, list):
            if _instance[0][0] is None:
                seeds = self.rng.integers(low=0, high=2**32 - 1, size=len(_instance))
                task_ids = [inst[1] for inst in _instance]
                instance: list[Instance] = list(zip(seeds, task_ids, strict=True))  # type: ignore[no-redef]
            return instance
        raise ValueError(f"Unknown instance type: {type(_instance)}")


class RoundRobinInstanceSelector(InstanceSelector):
    """Round robin instance selector.

    Rotate through instances.
    """

    def __init__(
        self, task_ids: list[str], seeds: list[int], offset: int = 0, selector_seed: int | None = None
    ) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int]
            List of seeds.
        offset : int, 0
            An optional offset to add to the index.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        super().__init__(task_ids, seeds, selector_seed)
        self._offset = offset
        self.idx = (self.idx + self._offset) % len(self.instances)

    def _select_instance(self, size: int = 1) -> InstanceRelax | list[InstanceRelax]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        InstanceRelax | list[InstanceRelax]
            (seed, task_id)
        """
        n_instances = len(self.instances)
        if size == 1:
            instance = self.instances[self.idx]
        else:
            indexer = np.arange(self.idx, self.idx + size) % n_instances
            instance = self.instances[indexer]
        self.idx = (self.idx + size) % n_instances
        return instance


class RandomInstanceSelector(InstanceSelector):
    """Random instance selector."""

    def _select_instance(self, size: int = 1) -> InstanceRelax | list[InstanceRelax]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        InstanceRelax | list[InstanceRelax]
            (seed, task_id)
        """
        indices = np.arange(0, len(self.instances))
        if size == 1:
            idx = self.rng.choice(indices)
            return self.instances[idx]
        ids = self.rng.choice(indices, size=size)
        return self.instances[ids]
