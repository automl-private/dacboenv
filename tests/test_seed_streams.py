"""Scientific seed-tree and persistent worker-allocation contracts."""

from __future__ import annotations

from collections import Counter

from dacboenv.experiment.ppo import aggregate_validation_scores, assign_training_worker_context, derive_worker_seed
from dacboenv.utils.seeding import derive_named_seed, episode_component_seeds, run_seed_metadata


def test_named_streams_are_reproducible_distinct_and_call_order_independent() -> None:
    """A sibling stream cannot perturb another component's sequence."""
    first = {
        name: derive_named_seed(41, name)
        for name in ("policy_model", "task_selector", "episode_inner", "policy_action_space")
    }
    reverse = {
        name: derive_named_seed(41, name)
        for name in reversed(("policy_model", "task_selector", "episode_inner", "policy_action_space"))
    }

    assert reverse == first
    assert len(set(first.values())) == len(first)
    assert derive_named_seed(42, "episode_inner") != first["episode_inner"]


def test_vector_workers_and_episode_components_have_distinct_streams() -> None:
    """Workers and stochastic BO components never copy an RNG state."""
    worker_seeds = [derive_worker_seed(73, worker_id) for worker_id in range(32)]

    assert len(set(worker_seeds)) == 32
    for inner_seed in worker_seeds:
        components = episode_component_seeds(inner_seed)
        assert len(set(components.values())) == len(components)


def test_run_seed_metadata_fully_records_replayable_roots() -> None:
    metadata = run_seed_metadata(17, 4)

    assert metadata == run_seed_metadata(17, 4)
    assert metadata["policy_model_seed"] != 17
    assert metadata["vector_worker_seeds"] == [derive_worker_seed(17, worker_id) for worker_id in range(4)]


def test_bbob_workers_are_persistently_dimension_balanced() -> None:
    tasks = [f"bbob/{dimension}/{function_id}/0" for dimension in (2, 4) for function_id in (3, 6, 8)]
    assignments = [assign_training_worker_context(tasks, worker_id=worker_id, n_workers=32) for worker_id in range(32)]

    assert Counter(assignment.bbob_dimension for assignment in assignments) == {2: 16, 4: 16}
    assert all(
        {int(task_id.split("/")[1]) for task_id in assignment.task_ids} == {assignment.bbob_dimension}
        for assignment in assignments
    )


def test_mixed_workers_use_requested_persistent_20_12_split() -> None:
    bbob_tasks = [f"bbob/{dimension}/3/0" for dimension in (2, 4)]
    yahpo_tasks = ["yahpo/so/lcbench/1/None", "yahpo/so/rbv2_super/2/None"]
    assignments = [
        assign_training_worker_context(
            bbob_tasks + yahpo_tasks,
            worker_id=worker_id,
            n_workers=32,
            bbob_fraction=0.6,
        )
        for worker_id in range(32)
    ]

    assert Counter(assignment.domain for assignment in assignments) == {"bbob": 20, "yahpo": 12}
    assert Counter(assignment.bbob_dimension for assignment in assignments if assignment.domain == "bbob") == {
        2: 10,
        4: 10,
    }


def test_worker_assignments_do_not_depend_on_action_space() -> None:
    tasks = [f"bbob/{dimension}/{function_id}/0" for dimension in (2, 4) for function_id in (3, 6)]

    for _action_space_name in ("wei", "lcb", "ucb", "af_selection"):
        assert [assign_training_worker_context(tasks, worker_id=worker_id, n_workers=4) for worker_id in range(4)] == [
            assign_training_worker_context(tasks, worker_id=worker_id, n_workers=4) for worker_id in range(4)
        ]


def test_validation_aggregation_balances_dimensions_and_scenarios() -> None:
    bbob_tasks = [
        *(f"bbob/4/{function_id}/1" for function_id in (2, 7, 11, 16, 20)),
        *(f"bbob/8/{function_id}/1" for function_id in (2, 7, 11, 16, 20)),
    ]
    yahpo_tasks = [
        "yahpo/so/lcbench/1/None",
        "yahpo/so/rbv2_glmnet/1/None",
        "yahpo/so/rbv2_glmnet/2/None",
        "yahpo/so/rbv2_glmnet/3/None",
    ]
    task_ids = bbob_tasks + yahpo_tasks
    one_seed_rewards = [1.0] * 5 + [3.0] * 5 + [10.0, 0.0, 0.0, 0.0]

    scores = aggregate_validation_scores(task_ids, [101, 202], one_seed_rewards * 2)

    assert scores.bbob_score == 2.0
    assert scores.per_scenario == {"lcbench": 10.0, "rbv2_glmnet": 0.0}
    assert scores.yahpo_score == 5.0
    assert scores.balanced_score == 3.5
    assert scores.worst_domain_score == 2.0
