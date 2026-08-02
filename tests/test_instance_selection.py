"""Focused contracts for flat and hierarchical instance selectors."""

from __future__ import annotations

from collections import Counter

import pytest
from dacboenv.env.instance import (
    HierarchicalBBOBInstanceSelector,
    HierarchicalYAHPOInstanceSelector,
    RandomInstanceSelector,
    RoundRobinInstanceSelector,
)


def _sequence(selector: object, n_draws: int) -> list[tuple[int | None, str]]:
    return [selector.select_instance() for _ in range(n_draws)]  # type: ignore[attr-defined, misc]


BBOB_TASK_IDS = [f"bbob/{dimension}/{function_id}/0" for dimension in (2, 4) for function_id in (3, 6, 8, 13, 17, 21)]
YAHPO_TASK_IDS = [
    "yahpo/so/lcbench/1/None",
    "yahpo/so/rbv2_glmnet/1/None",
    "yahpo/so/rbv2_glmnet/2/None",
    "yahpo/so/rbv2_glmnet/3/None",
    *(f"yahpo/so/rbv2_super/{instance}/None" for instance in range(1, 8)),
]


@pytest.mark.parametrize(
    ("selector_class", "task_ids"),
    [
        (HierarchicalBBOBInstanceSelector, BBOB_TASK_IDS),
        (HierarchicalYAHPOInstanceSelector, YAHPO_TASK_IDS),
    ],
)
def test_hierarchical_selector_seed_replays_exact_sequence(
    selector_class: type,
    task_ids: list[str],
) -> None:
    """The selector seed defines the complete context-selection stream."""
    first = selector_class(task_ids, [11, 12], selector_seed=91)
    replayed = selector_class(task_ids, [11, 12], selector_seed=91)
    different = selector_class(task_ids, [11, 12], selector_seed=92)

    first_sequence = _sequence(first, 200)
    assert _sequence(replayed, 200) == first_sequence
    assert _sequence(different, 200) != first_sequence


def test_bbob_hierarchy_balances_families_functions_and_dimensions() -> None:
    """Unequal numbers of functions do not change family or dimension weight."""
    selector = HierarchicalBBOBInstanceSelector(BBOB_TASK_IDS, [None], selector_seed=7)
    selected = [task_id for _seed, task_id in _sequence(selector, 30_000)]
    parsed = [tuple(map(int, task_id.split("/")[1:])) for task_id in selected]
    family_by_function = {3: 0, 6: 1, 8: 1, 13: 2, 17: 3, 21: 4}
    family_counts = Counter(family_by_function[function_id] for _dimension, function_id, _instance in parsed)

    assert set(family_counts) == set(range(5))
    for count in family_counts.values():
        assert count / len(parsed) == pytest.approx(0.2, abs=0.015)

    family_one_functions = Counter(
        function_id for _dimension, function_id, _instance in parsed if family_by_function[function_id] == 1
    )
    assert set(family_one_functions) == {6, 8}
    assert family_one_functions[6] / sum(family_one_functions.values()) == pytest.approx(0.5, abs=0.025)

    for function_id in family_by_function:
        dimensions = Counter(
            dimension for dimension, selected_function, _instance in parsed if selected_function == function_id
        )
        assert set(dimensions) == {2, 4}
        assert dimensions[2] / sum(dimensions.values()) == pytest.approx(0.5, abs=0.035)


def test_yahpo_hierarchy_balances_scenarios_despite_unequal_task_counts() -> None:
    """Scenario probability is independent of the number of its instances."""
    selector = HierarchicalYAHPOInstanceSelector(YAHPO_TASK_IDS, [None], selector_seed=17)
    selected = [task_id for _seed, task_id in _sequence(selector, 30_000)]
    scenario_counts = Counter(task_id.split("/")[2] for task_id in selected)

    assert set(scenario_counts) == {"lcbench", "rbv2_glmnet", "rbv2_super"}
    for count in scenario_counts.values():
        assert count / len(selected) == pytest.approx(1 / 3, abs=0.015)
    assert set(selected) == set(YAHPO_TASK_IDS)


@pytest.mark.parametrize("selector_class", [RoundRobinInstanceSelector, RandomInstanceSelector])
def test_flat_selectors_return_python_lists_for_batch_selection(selector_class: type) -> None:
    """Selecting more than one item must not index a Python list by an ndarray."""
    selector = selector_class(["task-a", "task-b"], [1, 2], selector_seed=3)

    selected = selector.select_instance(size=7)

    assert isinstance(selected, list)
    assert len(selected) == 7
    assert all(instance in selector.instances for instance in selected)
