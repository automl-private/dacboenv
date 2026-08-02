"""Cross-outer-seed evaluator result aggregation tests."""

from __future__ import annotations

from dataclasses import replace

from dacboenv.experiment.aggregate_evaluation_results import (
    aggregate_outer_seed_results,
    load_evaluation_records,
    merge_evaluation_records,
)
from dacboenv.experiment.paired_evaluator import (
    LEARNED_VALIDATION_SELECTED,
    MARGINAL_RANDOM_CONTROL,
    MODAL_STATIC_CLONE,
    STATIC_ACTION_PREFIX,
    EvaluationContext,
    EvaluationRecord,
    write_evaluation_records_csv,
)


def _record(context: EvaluationContext, method: str, regret: float, outer_seed: int | None) -> EvaluationRecord:
    return EvaluationRecord(
        **context.__dict__,
        outer_ppo_seed=outer_seed,
        method=method,
        action_family="wei",
        checkpoint_type="best" if outer_seed is not None else "none",
        final_incumbent=regret,
        final_reference_regret=regret,
        normalized_final_regret=regret,
        anytime_auc=regret,
        episode_return=1.0 - regret,
        action_histogram=(1, 0),
        deterministic_switch_rate=0.0,
        constant_policy=True,
        runtime_seconds=0.1,
        code_commit="a" * 40,
    )


def test_merge_and_cross_outer_seed_probability(tmp_path) -> None:
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="1",
        dimension=2,
        task_id="bbob/2/1/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=20,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash="a" * 64,
    )
    baseline = _record(context, f"{STATIC_ACTION_PREFIX}0", 0.5, None)
    first = _record(context, LEARNED_VALIDATION_SELECTED, 0.4, 1)
    second = _record(context, LEARNED_VALIDATION_SELECTED, 0.6, 2)
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    write_evaluation_records_csv([baseline, first], first_path)
    write_evaluation_records_csv([replace(baseline, runtime_seconds=0.2), second], second_path)

    merged = merge_evaluation_records([load_evaluation_records(first_path), load_evaluation_records(second_path)])
    summary = aggregate_outer_seed_results(
        merged,
        learned_method=LEARNED_VALIDATION_SELECTED,
        action_family="wei",
        checkpoint_type="best",
        baseline_methods=[f"{STATIC_ACTION_PREFIX}0"],
        n_resamples=20,
    )

    probability = summary["baselines"][f"{STATIC_ACTION_PREFIX}0"]["outer_seed_probability"]
    assert probability["n_outer_seeds"] == 2
    assert probability["probability_beating"] == 0.5


def test_cross_seed_controls_are_matched_to_their_source_outer_seed(tmp_path) -> None:
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="1",
        dimension=2,
        task_id="bbob/2/1/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=20,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash="a" * 64,
    )
    records = [
        _record(context, LEARNED_VALIDATION_SELECTED, 0.4, 1),
        _record(context, LEARNED_VALIDATION_SELECTED, 0.6, 2),
        _record(context, MODAL_STATIC_CLONE, 0.45, 1),
        _record(context, MODAL_STATIC_CLONE, 0.55, 2),
        _record(context, MARGINAL_RANDOM_CONTROL, 0.45, 1),
        _record(context, MARGINAL_RANDOM_CONTROL, 0.55, 2),
    ]
    path = tmp_path / "controls.csv"
    write_evaluation_records_csv(records, path)

    summary = aggregate_outer_seed_results(
        load_evaluation_records(path),
        learned_method=LEARNED_VALIDATION_SELECTED,
        action_family="wei",
        checkpoint_type="best",
        baseline_methods=[MODAL_STATIC_CLONE, MARGINAL_RANDOM_CONTROL],
        n_resamples=20,
    )

    for method in (MODAL_STATIC_CLONE, MARGINAL_RANDOM_CONTROL):
        probability = summary["baselines"][method]["outer_seed_probability"]
        assert probability["n_outer_seeds"] == 2
        assert probability["probability_beating"] == 0.5
        assert set(probability["baseline_cells_by_outer_seed"]) == {"1", "2"}
