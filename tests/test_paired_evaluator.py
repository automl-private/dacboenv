"""Synthetic contracts for the manifest-driven paired-evaluation core."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dacboenv.experiment.paired_evaluator import (
    BEST_VALIDATION_STATIC,
    DEFAULT_SMAC,
    DYNAMIC_ORACLE,
    LEARNED_FINAL,
    LEARNED_VALIDATION_SELECTED,
    MARGINAL_RANDOM_CONTROL,
    MODAL_STATIC_CLONE,
    SAWEI,
    STATIC_ACTION_PREFIX,
    TIDY_RECORD_FIELDS,
    UNIFORM_RANDOM,
    AnalysisOnlyMethodError,
    CheckpointSelectionError,
    ControlDerivationError,
    DistinctStateSubstitutionError,
    EvaluationContext,
    EvaluationMethod,
    EvaluationRecord,
    MethodCell,
    MethodRegistry,
    MethodRunnerUnavailableError,
    PairingError,
    PolicyStateSample,
    SealedManifestError,
    ValidationCheckpointEvaluation,
    aggregate_validation_score,
    derive_validation_controls,
    evaluate_registered_methods,
    hierarchical_paired_bootstrap,
    outer_seed_beat_probability,
    paired_method_comparison,
    policy_state_substitution_sensitivity,
    select_full_panel_checkpoint,
    select_validation_static_action,
    validate_paired_contexts,
    validation_budget_phase,
    write_evaluation_records_csv,
)
from dacboenv.experiment.protocol import manifest_hash


def _manifest(
    task_ids: list[str],
    inner_seeds: list[int],
    *,
    split: str = "validation",
    domain: str = "bbob",
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": f"synthetic-{domain}-{split}",
        "domain": domain,
        "split": split,
        "status": "ready",
        "runnable": True,
        "task_ids": task_ids,
        "inner_seeds": inner_seeds,
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def _context(
    task_id: str,
    inner_seed: int,
    manifest_digest: str,
    *,
    budget: int = 20,
) -> EvaluationContext:
    parts = task_id.split("/")
    if parts[0] == "bbob":
        return EvaluationContext(
            domain="bbob",
            scenario_or_function=parts[2],
            dimension=int(parts[1]),
            task_id=task_id,
            native_instance=parts[3],
            inner_seed=inner_seed,
            evaluation_budget=budget,
            reference_kind="exact",
            reference_value=0.0,
            objective_transform="identity",
            manifest_hash=manifest_digest,
        )
    return EvaluationContext(
        domain="yahpo",
        scenario_or_function=parts[2],
        dimension=None,
        task_id=task_id,
        native_instance=parts[3],
        inner_seed=inner_seed,
        evaluation_budget=budget,
        reference_kind="best_known",
        reference_value=0.1,
        objective_transform="scalar_minimization",
        manifest_hash=manifest_digest,
    )


def _record(
    context: EvaluationContext,
    *,
    method: str,
    action_family: str = "wei",
    checkpoint_type: str = "not_applicable",
    outer_seed: int | None = None,
    normalized_regret: float = 1.0,
    episode_return: float = 0.0,
    histogram: tuple[int, ...] = (1, 0, 0),
) -> EvaluationRecord:
    return EvaluationRecord(
        **context.__dict__,
        outer_ppo_seed=outer_seed,
        method=method,
        action_family=action_family,
        checkpoint_type=checkpoint_type,
        final_incumbent=context.reference_value + normalized_regret,
        final_reference_regret=normalized_regret,
        normalized_final_regret=normalized_regret,
        anytime_auc=normalized_regret,
        episode_return=episode_return,
        action_histogram=histogram,
        deterministic_switch_rate=0.0,
        constant_policy=True,
        runtime_seconds=0.01,
        code_commit="a" * 40,
    )


def _manifest_contexts(manifest: dict[str, Any]) -> list[EvaluationContext]:
    return [
        _context(task_id, seed, manifest["manifest_hash"])
        for seed in manifest["inner_seeds"]
        for task_id in manifest["task_ids"]
    ]


def test_method_registry_declares_every_requested_method_without_fake_runners() -> None:
    registry = MethodRegistry(n_static_actions=5)

    assert {
        LEARNED_VALIDATION_SELECTED,
        LEARNED_FINAL,
        MODAL_STATIC_CLONE,
        MARGINAL_RANDOM_CONTROL,
        BEST_VALIDATION_STATIC,
        UNIFORM_RANDOM,
        DEFAULT_SMAC,
        SAWEI,
        DYNAMIC_ORACLE,
        *(f"{STATIC_ACTION_PREFIX}{action}" for action in range(5)),
    }.issubset(registry.method_names)
    with pytest.raises(MethodRunnerUnavailableError, match="no runner callback"):
        registry.runner(LEARNED_FINAL)


def test_all_runners_are_resolved_before_the_first_callback() -> None:
    manifest = _manifest(["bbob/2/1/0"], [11])
    contexts = _manifest_contexts(manifest)
    registry = MethodRegistry()
    calls: list[str] = []

    def static_runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        calls.append(method.name)
        return _record(context, method=method.name)

    registry.register_runner(f"{STATIC_ACTION_PREFIX}0", static_runner)

    with pytest.raises(MethodRunnerUnavailableError, match=DEFAULT_SMAC):
        evaluate_registered_methods(
            manifest,
            contexts,
            [f"{STATIC_ACTION_PREFIX}0", DEFAULT_SMAC],
            registry,
        )
    assert calls == []


def test_registered_execution_is_exactly_paired_and_rejects_changed_context() -> None:
    manifest = _manifest(["bbob/2/1/0", "bbob/2/6/0"], [11, 12])
    contexts = _manifest_contexts(manifest)
    registry = MethodRegistry()

    def runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        return _record(
            context,
            method=method.name,
            checkpoint_type="best" if method.requires_trained_model else "not_applicable",
            outer_seed=7 if method.requires_trained_model else None,
        )

    registry.register_runner(LEARNED_VALIDATION_SELECTED, runner)
    registry.register_runner(f"{STATIC_ACTION_PREFIX}0", runner)
    records = evaluate_registered_methods(
        manifest,
        contexts,
        [LEARNED_VALIDATION_SELECTED, f"{STATIC_ACTION_PREFIX}0"],
        registry,
    )

    assert len(records) == 2 * len(contexts)
    assert len(validate_paired_contexts(records)) == len(contexts)

    bad_registry = MethodRegistry()

    def changed_context_runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        changed = replace(context, evaluation_budget=context.evaluation_budget + 1)
        return _record(changed, method=method.name)

    bad_registry.register_runner(f"{STATIC_ACTION_PREFIX}0", changed_context_runner)
    with pytest.raises(PairingError, match="changed its assigned context"):
        evaluate_registered_methods(
            manifest,
            contexts,
            [f"{STATIC_ACTION_PREFIX}0"],
            bad_registry,
        )


def test_sealed_test_execution_and_analysis_only_methods_require_explicit_flags() -> None:
    manifest = _manifest(["bbob/2/1/2"], [91], split="test")
    contexts = _manifest_contexts(manifest)
    registry = MethodRegistry()
    calls: list[str] = []

    def runner(context: EvaluationContext, method: EvaluationMethod) -> EvaluationRecord:
        calls.append(method.name)
        return _record(context, method=method.name)

    registry.register_runner(f"{STATIC_ACTION_PREFIX}0", runner)
    registry.register_runner(DYNAMIC_ORACLE, runner)

    with pytest.raises(SealedManifestError, match="sealed final-test"):
        evaluate_registered_methods(
            manifest,
            contexts,
            [f"{STATIC_ACTION_PREFIX}0"],
            registry,
        )
    assert calls == []

    records = evaluate_registered_methods(
        manifest,
        contexts,
        [f"{STATIC_ACTION_PREFIX}0"],
        registry,
        allow_sealed_test=True,
    )
    assert len(records) == 1

    validation = _manifest(["bbob/2/1/0"], [91])
    with pytest.raises(AnalysisOnlyMethodError, match="analysis-only"):
        evaluate_registered_methods(
            validation,
            _manifest_contexts(validation),
            [DYNAMIC_ORACLE],
            registry,
        )


def test_tidy_schema_and_csv_serialization(tmp_path: Path) -> None:
    manifest = _manifest(["bbob/2/1/0"], [11])
    context = _manifest_contexts(manifest)[0]
    records = [
        _record(context, method=f"{STATIC_ACTION_PREFIX}0", histogram=(2, 0, 0)),
        _record(context, method=UNIFORM_RANDOM, histogram=(1, 1, 0)),
    ]
    output_path = tmp_path / "paired.csv"

    write_evaluation_records_csv(records, output_path)

    with output_path.open(encoding="utf-8", newline="") as output:
        rows = list(csv.DictReader(output))
    assert tuple(rows[0]) == TIDY_RECORD_FIELDS
    assert json.loads(rows[0]["action_histogram"]) == [2, 0, 0]
    assert rows[0]["manifest_hash"] == manifest["manifest_hash"]


def test_pairing_rejects_budget_or_reference_convention_mismatch() -> None:
    manifest = _manifest(["bbob/2/1/0"], [11])
    context = _manifest_contexts(manifest)[0]
    method_record = _record(context, method=LEARNED_FINAL, checkpoint_type="final", outer_seed=0)
    mismatched_context = replace(context, evaluation_budget=21, objective_transform="different")
    baseline_record = _record(mismatched_context, method=f"{STATIC_ACTION_PREFIX}0")

    with pytest.raises(PairingError, match="not exactly paired"):
        validate_paired_contexts([method_record, baseline_record])


def test_modal_and_marginal_controls_are_persisted_from_validation_only(tmp_path: Path) -> None:
    manifest = _manifest(["bbob/2/1/0", "bbob/2/6/0"], [11])
    contexts = _manifest_contexts(manifest)
    records = [
        _record(
            contexts[0],
            method=LEARNED_VALIDATION_SELECTED,
            checkpoint_type="best",
            outer_seed=3,
            histogram=(2, 1, 0),
        ),
        _record(
            contexts[1],
            method=LEARNED_VALIDATION_SELECTED,
            checkpoint_type="best",
            outer_seed=3,
            histogram=(0, 3, 0),
        ),
    ]
    output_path = tmp_path / "controls" / "provenance.json"

    provenance = derive_validation_controls(records, manifest, output_path=output_path)

    assert provenance.source_action_counts == (2, 4, 0)
    assert provenance.source_action_frequencies == pytest.approx((1 / 3, 2 / 3, 0.0))
    assert provenance.modal_action == 1
    assert provenance.source_checkpoint == "best"
    assert provenance.source_validation_manifest_hash == manifest["manifest_hash"]
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["source_action_frequencies"] == pytest.approx([1 / 3, 2 / 3, 0.0])
    assert persisted["modal_control_method"] == MODAL_STATIC_CLONE
    assert persisted["marginal_control_method"] == MARGINAL_RANDOM_CONTROL


def test_best_static_action_is_selected_once_from_complete_validation(tmp_path: Path) -> None:
    manifest = _manifest(["bbob/2/1/0", "bbob/2/6/0"], [11])
    contexts = _manifest_contexts(manifest)
    records = [
        _record(context, method=f"{STATIC_ACTION_PREFIX}{action}", normalized_regret=regret)
        for action, regret in ((0, 0.8), (1, 0.2), (2, 0.5))
        for context in contexts
    ]
    output = tmp_path / "best-static.json"

    provenance = select_validation_static_action(records, manifest, n_actions=3, output_path=output)

    assert provenance.selected_action == 1
    assert provenance.source_validation_manifest_hash == manifest["manifest_hash"]
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["method"] == BEST_VALIDATION_STATIC
    assert persisted["selected_action"] == 1
    assert set(persisted["source_static_scores"]) == {"0", "1", "2"}


def test_best_static_selection_rejects_nonvalidation_and_incomplete_actions() -> None:
    training_manifest = _manifest(["bbob/2/1/0"], [11], split="train")
    training_record = _record(
        _manifest_contexts(training_manifest)[0],
        method=f"{STATIC_ACTION_PREFIX}0",
    )
    with pytest.raises(ControlDerivationError, match="only on a validation"):
        select_validation_static_action([training_record], training_manifest, n_actions=1)

    validation_manifest = _manifest(["bbob/2/1/0"], [11])
    validation_record = _record(
        _manifest_contexts(validation_manifest)[0],
        method=f"{STATIC_ACTION_PREFIX}0",
    )
    with pytest.raises(ControlDerivationError, match="every static action"):
        select_validation_static_action([validation_record], validation_manifest, n_actions=2)


@pytest.mark.parametrize("split", ["train", "test"])
def test_control_derivation_rejects_non_validation_splits(split: str) -> None:
    manifest = _manifest(["bbob/2/1/0"], [11], split=split)
    record = _record(
        _manifest_contexts(manifest)[0],
        method=LEARNED_FINAL,
        checkpoint_type="final",
        outer_seed=0,
    )

    with pytest.raises(ControlDerivationError, match="only from a validation manifest"):
        derive_validation_controls([record], manifest)


def test_paired_differences_win_tie_loss_and_effect_size() -> None:
    manifest = _manifest([f"bbob/2/{function}/0" for function in (1, 2, 3)], [11])
    contexts = _manifest_contexts(manifest)
    method_records = [
        _record(
            context,
            method=LEARNED_FINAL,
            checkpoint_type="final",
            outer_seed=0,
            normalized_regret=value,
        )
        for context, value in zip(contexts, (1.0, 1.0, 3.0), strict=True)
    ]
    baseline_records = [
        _record(context, method=f"{STATIC_ACTION_PREFIX}0", normalized_regret=2.0) for context in contexts
    ]
    records = method_records + baseline_records
    method_cell = method_records[0].method_cell
    baseline_cell = baseline_records[0].method_cell

    comparison = paired_method_comparison(records, method_cell, baseline_cell)

    assert tuple(context.task_id for context in comparison.contexts) == tuple(context.task_id for context in contexts)
    assert comparison.differences == pytest.approx((1.0, 1.0, -1.0))
    assert comparison.mean_difference == pytest.approx(1 / 3)
    assert (comparison.wins, comparison.ties, comparison.losses) == (2, 0, 1)
    assert comparison.win_loss_effect == pytest.approx(1 / 3)
    assert comparison.standardized_mean_difference == pytest.approx(1 / (2 * 3**0.5))


def test_episode_return_comparison_uses_higher_is_better_direction() -> None:
    manifest = _manifest(["bbob/2/1/0"], [11])
    context = _manifest_contexts(manifest)[0]
    method = _record(
        context,
        method=LEARNED_FINAL,
        checkpoint_type="final",
        outer_seed=0,
        episode_return=2.0,
    )
    baseline = _record(context, method=f"{STATIC_ACTION_PREFIX}0", episode_return=1.0)

    comparison = paired_method_comparison(
        [method, baseline],
        method.method_cell,
        baseline.method_cell,
        metric="episode_return",
    )

    assert comparison.differences == pytest.approx((1.0,))
    assert comparison.wins == 1


def test_outer_seed_beat_probability_preserves_each_outer_seed() -> None:
    manifest = _manifest(["bbob/2/1/0", "bbob/2/6/0"], [11])
    contexts = _manifest_contexts(manifest)
    baseline_records = [
        _record(context, method=f"{STATIC_ACTION_PREFIX}0", normalized_regret=2.0) for context in contexts
    ]
    learned_records = [
        _record(
            context,
            method=LEARNED_VALIDATION_SELECTED,
            checkpoint_type="best",
            outer_seed=outer_seed,
            normalized_regret=value,
        )
        for outer_seed, value in ((0, 1.0), (1, 3.0), (2, 2.0))
        for context in contexts
    ]

    result = outer_seed_beat_probability(
        learned_records + baseline_records,
        method=LEARNED_VALIDATION_SELECTED,
        action_family="wei",
        checkpoint_type="best",
        baseline_cell=baseline_records[0].method_cell,
    )

    assert result.n_outer_seeds == 3
    assert (result.beating_seeds, result.tying_seeds, result.losing_seeds) == (1, 1, 1)
    assert result.probability_beating == pytest.approx(1 / 3)
    assert result.mean_difference_by_seed == pytest.approx({0: 1.0, 1: -1.0, 2: 0.0})


def _bootstrap_records() -> tuple[list[EvaluationRecord], MethodCell, MethodCell]:
    contexts_and_improvements: list[tuple[EvaluationContext, float]] = []
    digest = "b" * 64
    for dimension, first, second in ((2, 1.0, 3.0), (8, 5.0, 7.0)):
        for function, improvement in ((1, first), (6, second)):
            for inner_seed in (11, 12):
                task_id = f"bbob/{dimension}/{function}/0"
                contexts_and_improvements.append((_context(task_id, inner_seed, digest), improvement))
    for scenario, improvement in (("lcbench", 6.0), ("rbv2_super", 10.0)):
        for instance in ("a", "b"):
            for inner_seed in (11, 12):
                task_id = f"yahpo/so/{scenario}/{instance}/None"
                contexts_and_improvements.append((_context(task_id, inner_seed, digest), improvement))

    baseline_records = [
        _record(context, method=f"{STATIC_ACTION_PREFIX}0", normalized_regret=20.0)
        for context, _improvement in contexts_and_improvements
    ]
    learned_records = [
        _record(
            context,
            method=LEARNED_FINAL,
            checkpoint_type="final",
            outer_seed=4,
            normalized_regret=20.0 - improvement,
        )
        for context, improvement in contexts_and_improvements
    ]
    records = learned_records + baseline_records
    return records, learned_records[0].method_cell, baseline_records[0].method_cell


def test_hierarchical_bootstrap_covers_bbob_yahpo_and_fixed_mixed_weight() -> None:
    records, method_cell, baseline_cell = _bootstrap_records()

    mixed = hierarchical_paired_bootstrap(
        records,
        method_cell,
        baseline_cell,
        n_resamples=300,
        seed=17,
    )

    assert mixed.per_bbob_dimension[2].point_estimate == pytest.approx(2.0)
    assert mixed.per_bbob_dimension[8].point_estimate == pytest.approx(6.0)
    assert mixed.per_domain["bbob"].point_estimate == pytest.approx(4.0)
    assert mixed.per_domain["yahpo"].point_estimate == pytest.approx(8.0)
    assert mixed.overall.point_estimate == pytest.approx(6.0)
    assert mixed.overall.confidence_lower <= mixed.overall.point_estimate
    assert mixed.overall.confidence_upper >= mixed.overall.point_estimate

    repeated = hierarchical_paired_bootstrap(
        records,
        method_cell,
        baseline_cell,
        n_resamples=300,
        seed=17,
    )
    assert repeated == mixed


@pytest.mark.parametrize(("domain", "expected"), [("bbob", 4.0), ("yahpo", 8.0)])
def test_hierarchical_bootstrap_reports_each_domain_independently(domain: str, expected: float) -> None:
    records, method_cell, baseline_cell = _bootstrap_records()
    filtered = [record for record in records if record.domain == domain]

    result = hierarchical_paired_bootstrap(
        filtered,
        method_cell,
        baseline_cell,
        n_resamples=100,
        seed=3,
    )

    assert result.overall.point_estimate == pytest.approx(expected)
    assert set(result.per_domain) == {domain}


def test_validation_aggregation_uses_hierarchies_and_exact_half_domain_weight() -> None:
    records, method_cell, _baseline_cell = _bootstrap_records()
    learned_records = [record for record in records if record.method_cell == method_cell]

    score = aggregate_validation_score(learned_records)

    # Lower normalized regret is oriented as a higher-is-better negative score.
    assert score.per_bbob_dimension == pytest.approx({2: -18.0, 8: -14.0})
    assert score.bbob_score == pytest.approx(-16.0)
    assert score.per_yahpo_scenario == pytest.approx({"lcbench": -14.0, "rbv2_super": -10.0})
    assert score.yahpo_score == pytest.approx(-12.0)
    assert score.balanced_score == pytest.approx(0.5 * -16.0 + 0.5 * -12.0)
    assert score.worst_domain_score == pytest.approx(-16.0)


def test_checkpoint_selection_excludes_step_zero_and_frequent_panel_scores() -> None:
    records, method_cell, _baseline_cell = _bootstrap_records()
    base_score = aggregate_validation_score([record for record in records if record.method_cell == method_cell])
    step_zero = ValidationCheckpointEvaluation(
        checkpoint_id="step-zero",
        training_step=0,
        panel_tier="full",
        is_step_zero=True,
        trained=False,
        score=replace(base_score, balanced_score=100.0),
    )
    frequent = ValidationCheckpointEvaluation(
        checkpoint_id="frequent-winner",
        training_step=10,
        panel_tier="frequent",
        is_step_zero=False,
        trained=True,
        score=replace(base_score, balanced_score=50.0),
    )
    full_early = ValidationCheckpointEvaluation(
        checkpoint_id="full-early",
        training_step=10,
        panel_tier="full",
        is_step_zero=False,
        trained=True,
        score=replace(base_score, balanced_score=1.0, bbob_score=3.0, yahpo_score=0.0),
    )
    full_late = ValidationCheckpointEvaluation(
        checkpoint_id="full-late",
        training_step=20,
        panel_tier="full",
        is_step_zero=False,
        trained=True,
        score=replace(base_score, balanced_score=2.0, bbob_score=2.0, yahpo_score=4.0),
    )

    candidates = [step_zero, frequent, full_early, full_late]
    assert select_full_panel_checkpoint(candidates).checkpoint_id == "full-late"
    assert select_full_panel_checkpoint(candidates, target="bbob").checkpoint_id == "full-early"
    assert select_full_panel_checkpoint(candidates, target="yahpo").checkpoint_id == "full-late"


def test_checkpoint_selection_fails_without_trained_full_panel_candidate() -> None:
    records, method_cell, _baseline_cell = _bootstrap_records()
    score = aggregate_validation_score([record for record in records if record.method_cell == method_cell])
    step_zero = ValidationCheckpointEvaluation(
        checkpoint_id="step-zero",
        training_step=0,
        panel_tier="full",
        is_step_zero=True,
        trained=False,
        score=score,
    )
    frequent = ValidationCheckpointEvaluation(
        checkpoint_id="frequent",
        training_step=10,
        panel_tier="frequent",
        is_step_zero=False,
        trained=True,
        score=score,
    )

    with pytest.raises(CheckpointSelectionError, match="full-panel"):
        select_full_panel_checkpoint([step_zero, frequent])
    with pytest.raises(ValueError, match="is_step_zero"):
        replace(step_zero, is_step_zero=False)


def _policy_observation(action_scores: tuple[float, float, float]) -> dict[str, np.ndarray]:
    return {
        "global_state": np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        "action_features": np.asarray(
            [[action_scores[0], 0.0], [action_scores[1], 0.0], [action_scores[2], 0.0]],
            dtype=np.float32,
        ),
    }


def _dummy_actor_probability(observation: dict[str, np.ndarray]) -> np.ndarray:
    logits = np.asarray(observation["action_features"][:, 0], dtype=float)
    exponential = np.exp(logits - np.max(logits))
    return exponential / np.sum(exponential)


def test_state_substitution_sources_are_provably_distinct() -> None:
    samples = [
        PolicyStateSample("bbob/4/2/1", 0.10, _policy_observation((3.0, 1.0, 0.0))),
        PolicyStateSample("bbob/4/7/1", 0.12, _policy_observation((0.0, 3.0, 1.0))),
        PolicyStateSample("bbob/4/2/1", 0.80, _policy_observation((1.0, 0.0, 3.0))),
    ]

    sensitivity = policy_state_substitution_sensitivity(samples, _dummy_actor_probability)

    another_task = sensitivity["state_from_another_task"]
    assert another_task.source_task_id == "bbob/4/7/1"
    assert another_task.task_changed is True
    assert another_task.budget_phase_changed is False
    assert another_task.kl_divergence > 0.0
    assert another_task.total_variation_distance > 0.0
    assert another_task.top_action_changed is True

    another_budget = sensitivity["state_from_another_budget_phase"]
    assert another_budget.source_task_id == "bbob/4/2/1"
    assert another_budget.task_changed is False
    assert another_budget.source_budget_phase != validation_budget_phase(samples[0].budget_fraction)
    assert another_budget.budget_phase_changed is True
    assert another_budget.top_action_changed is True


def test_state_substitution_fails_instead_of_reusing_same_task_or_budget_phase() -> None:
    same_task_and_phase = [
        PolicyStateSample("bbob/4/2/1", 0.10, _policy_observation((3.0, 1.0, 0.0))),
        PolicyStateSample("bbob/4/2/1", 0.20, _policy_observation((0.0, 3.0, 1.0))),
    ]
    with pytest.raises(DistinctStateSubstitutionError, match="task ID distinct"):
        policy_state_substitution_sensitivity(same_task_and_phase, _dummy_actor_probability)

    different_tasks_same_phase = [
        PolicyStateSample("bbob/4/2/1", 0.10, _policy_observation((3.0, 1.0, 0.0))),
        PolicyStateSample("bbob/4/7/1", 0.20, _policy_observation((0.0, 3.0, 1.0))),
    ]
    with pytest.raises(DistinctStateSubstitutionError, match="budget phase distinct"):
        policy_state_substitution_sensitivity(different_tasks_same_phase, _dummy_actor_probability)
