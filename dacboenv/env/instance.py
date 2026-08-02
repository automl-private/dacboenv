"""Instance Selection for DACBO Env."""

from __future__ import annotations

import itertools
import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")

BBOB_FUNCTION_FAMILY_RANGES = (
    range(1, 6),
    range(6, 10),
    range(10, 15),
    range(15, 20),
    range(20, 25),
)
_BBOB_TASK_ID = re.compile(
    r"^bbob/(?P<dimension>\d+)/(?P<function_id>\d+)/(?P<instance_id>\d+)$",
    flags=re.IGNORECASE,
)
_YAHPO_QUALIFIERS = frozenset({"so", "mo", "momf"})
_MIN_YAHPO_TASK_ID_PARTS = 3


# TypeVar syntax retains the package's declared Python 3.10 compatibility.
def _choice(rng: np.random.Generator, values: Sequence[T]) -> T:  # noqa: UP047
    """Choose one list entry without NumPy coercing its Python type."""
    return values[int(rng.integers(len(values)))]


def _bbob_function_family(function_id: int) -> int:
    """Return the canonical BBOB function-group index."""
    for family, function_ids in enumerate(BBOB_FUNCTION_FAMILY_RANGES):
        if function_id in function_ids:
            return family
    raise ValueError(f"BBOB function id must be in [1, 24], got {function_id}.")


def _parse_bbob_task_id(task_id: str) -> tuple[int, int, int]:
    """Parse one canonical ``bbob/dimension/function/instance`` task ID."""
    match = _BBOB_TASK_ID.fullmatch(task_id)
    if match is None:
        raise ValueError(
            "Hierarchical BBOB selection requires task IDs of the form "
            f"'bbob/<dimension>/<function>/<instance>', got {task_id!r}."
        )
    dimension = int(match.group("dimension"))
    function_id = int(match.group("function_id"))
    instance_id = int(match.group("instance_id"))
    if dimension <= 0:
        raise ValueError(f"BBOB dimension must be positive, got {dimension} in {task_id!r}.")
    _bbob_function_family(function_id)
    return dimension, function_id, instance_id


def _parse_yahpo_scenario(task_id: str) -> str:
    """Return the scenario component from CARPS YAHPO task IDs."""
    parts = task_id.split("/")
    if len(parts) < _MIN_YAHPO_TASK_ID_PARTS or parts[0].lower() != "yahpo":
        raise ValueError(
            f"Hierarchical YAHPO selection requires a CARPS task ID beginning with 'yahpo/', got {task_id!r}."
        )
    scenario_index = 2 if parts[1].lower() in _YAHPO_QUALIFIERS else 1
    if len(parts) <= scenario_index or not parts[scenario_index]:
        raise ValueError(f"YAHPO task ID has no scenario component: {task_id!r}.")
    return parts[scenario_index]


class InstanceSelector(ABC):
    """Instance Selector.

    One instance is represented as (task_id, seed).
    The list of instances is [(seed_0, task_id_0), (seed_0, task_id_1), ..., (seed_1, task_id_0),]

    Attributes
    ----------
    task_ids : list[str]
        List of carps task ids.
    seeds : list[int | None]
        List of seeds.
    idx : int
        Current instance index. Default is 0.
    rng : Generator
        Random generator.
    """

    def __init__(self, task_ids: list[str], seeds: list[int | None], selector_seed: int | None = None) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int | None]
            List of seeds.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        self.task_ids = task_ids
        self.seeds = seeds
        self.instances = list(itertools.product(self.seeds, self.task_ids))
        self.idx: int = 0
        self.selector_seed = selector_seed
        self.rng = np.random.default_rng(seed=selector_seed)

    @abstractmethod
    def select_instance(self, size: int = 1) -> tuple[int | None, str] | list[tuple[int | None, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        tuple[int | None, str] | list[tuple[int | None, str]]
            (seed, task_id)
        """


class RoundRobinInstanceSelector(InstanceSelector):
    """Round robin instance selector.

    Rotate through instances.
    """

    def __init__(
        self, task_ids: list[str], seeds: list[int | None], offset: int = 0, selector_seed: int | None = None
    ) -> None:
        """Initialize instance selector.

        Parameters
        ----------
        task_ids : list[str]
            List of carps task ids.
        seeds : list[int | None]
            List of seeds.
        offset : int, 0
            An optional offset to add to the index.
        selector_seed : int | None, optional
            Selector seed, e.g., needed in random selection, by default None
        """
        super().__init__(task_ids, seeds, selector_seed)
        self._offset = offset
        self.idx = (self.idx + self._offset) % len(self.instances)

    def select_instance(self, size: int = 1) -> tuple[int | None, str] | list[tuple[int | None, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        tuple[int | None, str] | list[tuple[int | None, str]]
            (seed, task_id)
        """
        n_instances = len(self.instances)
        if size == 1:
            instance = self.instances[self.idx]
        else:
            indexer = np.arange(self.idx, self.idx + size) % n_instances
            instance = [self.instances[int(index)] for index in indexer]
        self.idx = (self.idx + size) % n_instances
        return instance


class RandomInstanceSelector(InstanceSelector):
    """Random instance selector."""

    def select_instance(self, size: int = 1) -> tuple[int | None, str] | list[tuple[int | None, str]]:
        """Select next instance.

        Parameters
        ----------
        size : int, optional
            The number of instances, by default 1.

        Returns
        -------
        tuple[int | None, str] | list[tuple[int | None, str]]
            (seed, task_id)
        """
        indices = np.arange(0, len(self.instances))
        if size == 1:
            idx = self.rng.choice(indices)
            return self.instances[idx]
        ids = self.rng.choice(indices, size=size)
        return [self.instances[int(idx)] for idx in ids]


class HierarchicalBBOBInstanceSelector(InstanceSelector):
    """Sample BBOB family, function, dimension, and task uniformly in order."""

    def __init__(self, task_ids: list[str], seeds: list[int | None], selector_seed: int | None = None) -> None:
        """Build a hierarchy from canonical BBOB task identifiers."""
        super().__init__(task_ids, seeds, selector_seed)
        tasks_by_family: dict[int, dict[int, dict[int, list[str]]]] = {}
        for task_id in task_ids:
            dimension, function_id, _instance_id = _parse_bbob_task_id(task_id)
            family = _bbob_function_family(function_id)
            tasks_by_dimension = tasks_by_family.setdefault(family, {}).setdefault(function_id, {})
            tasks_by_dimension.setdefault(dimension, []).append(task_id)

        self._tasks_by_family = tasks_by_family
        self._families = sorted(tasks_by_family)
        for tasks_by_function in self._tasks_by_family.values():
            for tasks_by_dimension in tasks_by_function.values():
                for tasks in tasks_by_dimension.values():
                    tasks.sort()

    def _select_one(self) -> tuple[int | None, str]:
        family = int(_choice(self.rng, self._families))
        tasks_by_function = self._tasks_by_family[family]
        function_id = int(_choice(self.rng, sorted(tasks_by_function)))
        tasks_by_dimension = tasks_by_function[function_id]
        dimension = int(_choice(self.rng, sorted(tasks_by_dimension)))
        task_id = str(_choice(self.rng, tasks_by_dimension[dimension]))
        seed = _choice(self.rng, self.seeds)
        return seed, task_id

    def select_instance(self, size: int = 1) -> tuple[int | None, str] | list[tuple[int | None, str]]:
        """Select one instance, or a list when ``size`` is greater than one."""
        if size == 1:
            return self._select_one()
        return [self._select_one() for _ in range(size)]


class HierarchicalYAHPOInstanceSelector(InstanceSelector):
    """Sample a YAHPO scenario uniformly, then a task within that scenario."""

    def __init__(self, task_ids: list[str], seeds: list[int | None], selector_seed: int | None = None) -> None:
        """Build a scenario hierarchy from CARPS YAHPO task identifiers."""
        super().__init__(task_ids, seeds, selector_seed)
        tasks_by_scenario: dict[str, list[str]] = {}
        for task_id in task_ids:
            tasks_by_scenario.setdefault(_parse_yahpo_scenario(task_id), []).append(task_id)

        self._tasks_by_scenario = tasks_by_scenario
        self._scenarios = sorted(tasks_by_scenario)
        for tasks in self._tasks_by_scenario.values():
            tasks.sort()

    def _select_one(self) -> tuple[int | None, str]:
        scenario = str(_choice(self.rng, self._scenarios))
        task_id = str(_choice(self.rng, self._tasks_by_scenario[scenario]))
        seed = _choice(self.rng, self.seeds)
        return seed, task_id

    def select_instance(self, size: int = 1) -> tuple[int | None, str] | list[tuple[int | None, str]]:
        """Select one instance, or a list when ``size`` is greater than one."""
        if size == 1:
            return self._select_one()
        return [self._select_one() for _ in range(size)]
