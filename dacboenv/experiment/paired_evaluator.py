"""Manifest-guarded paired-evaluation records and statistics.

This module deliberately contains no benchmark, policy, or optimizer runner.
Callers must register explicit runner callbacks before any method can execute.
That separation makes the scientific protocol reusable without suggesting that
declaring a method in the registry has evaluated it.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
)

Domain = Literal["bbob", "yahpo"]
ReferenceKind = Literal["exact", "best_known"]

LEARNED_VALIDATION_SELECTED = "learned_validation_selected"
LEARNED_FINAL = "learned_final"
MODAL_STATIC_CLONE = "modal_static_clone"
MARGINAL_RANDOM_CONTROL = "marginal_frequency_matched_random"
BEST_VALIDATION_STATIC = "best_validation_static"
UNIFORM_RANDOM = "uniform_random"
DEFAULT_SMAC = "default_smac"
SAWEI = "sawei"
DYNAMIC_ORACLE = "dynamic_oracle"
STATIC_ACTION_PREFIX = "static_action_"

_MIN_CATEGORICAL_ACTIONS = 2
_BBOB_TASK_ID_PARTS = 4
_BBOB_FUNCTION_GROUP_UPPER_BOUNDS = (5, 9, 14, 19, 24)

TIDY_RECORD_FIELDS = (
    "domain",
    "scenario_or_function",
    "dimension",
    "task_id",
    "native_instance",
    "inner_seed",
    "outer_ppo_seed",
    "method",
    "action_family",
    "checkpoint_type",
    "evaluation_budget",
    "interaction_frequency",
    "reference_kind",
    "reference_value",
    "objective_transform",
    "final_incumbent",
    "final_reference_regret",
    "normalized_final_regret",
    "anytime_auc",
    "episode_return",
    "action_histogram",
    "deterministic_switch_rate",
    "constant_policy",
    "runtime_seconds",
    "manifest_hash",
    "code_commit",
)

_LOWER_IS_BETTER_METRICS = frozenset(
    {
        "final_incumbent",
        "final_reference_regret",
        "normalized_final_regret",
        "anytime_auc",
    }
)
_HIGHER_IS_BETTER_METRICS = frozenset({"episode_return"})


class PairingError(ValueError):
    """Raised when methods do not cover identical scientific contexts."""


class SealedManifestError(PermissionError):
    """Raised when a final-test manifest was not explicitly authorized."""


class MethodRunnerUnavailableError(RuntimeError):
    """Raised before execution when a declared method has no runner callback."""


class AnalysisOnlyMethodError(PermissionError):
    """Raised when an analysis-only method was not explicitly requested."""


class ControlDerivationError(ValueError):
    """Raised when modal or marginal controls would use non-validation data."""


class CheckpointSelectionError(ValueError):
    """Raised when validation checkpoint selection would violate the protocol."""


class DistinctStateSubstitutionError(ValueError):
    """Raised when a requested policy-state substitution is not truly distinct."""


@dataclass(frozen=True)
class ContextKey:
    """All fields that must be identical for a paired method comparison."""

    domain: Domain
    scenario_or_function: str
    dimension: int | None
    task_id: str
    native_instance: str
    inner_seed: int
    evaluation_budget: int
    reference_kind: ReferenceKind
    reference_value: float
    objective_transform: str
    manifest_hash: str
    interaction_frequency: int = 1

    def sort_key(self) -> tuple[Any, ...]:
        """Return a total ordering that also handles YAHPO's null dimension."""
        return (
            self.domain,
            -1 if self.dimension is None else self.dimension,
            self.scenario_or_function,
            self.task_id,
            self.native_instance,
            self.inner_seed,
            self.evaluation_budget,
            self.reference_kind,
            self.reference_value,
            self.objective_transform,
            self.manifest_hash,
            self.interaction_frequency,
        )


@dataclass(frozen=True)
class MethodCell:
    """One independently reportable method/checkpoint/outer-seed cell."""

    method: str
    action_family: str
    checkpoint_type: str
    outer_ppo_seed: int | None

    @classmethod
    def from_record(cls, record: EvaluationRecord) -> MethodCell:
        """Build a cell identifier from one tidy record."""
        return cls(
            method=record.method,
            action_family=record.action_family,
            checkpoint_type=record.checkpoint_type,
            outer_ppo_seed=record.outer_ppo_seed,
        )

    def sort_key(self) -> tuple[Any, ...]:
        """Return a stable ordering for reports and tests."""
        return (
            self.method,
            self.action_family,
            self.checkpoint_type,
            -1 if self.outer_ppo_seed is None else self.outer_ppo_seed,
        )


@dataclass(frozen=True)
class EvaluationContext:
    """A manifest context plus budget and objective/reference conventions."""

    domain: Domain
    scenario_or_function: str
    dimension: int | None
    task_id: str
    native_instance: str
    inner_seed: int
    evaluation_budget: int
    reference_kind: ReferenceKind
    reference_value: float
    objective_transform: str
    manifest_hash: str
    interaction_frequency: int = 1

    @property
    def key(self) -> ContextKey:
        """Return the exact pairing key."""
        return ContextKey(**asdict(self))

    @classmethod
    def from_record(cls, record: EvaluationRecord) -> EvaluationContext:
        """Discard method outcomes while retaining pairing invariants."""
        return cls(
            domain=record.domain,
            scenario_or_function=record.scenario_or_function,
            dimension=record.dimension,
            task_id=record.task_id,
            native_instance=record.native_instance,
            inner_seed=record.inner_seed,
            evaluation_budget=record.evaluation_budget,
            reference_kind=record.reference_kind,
            reference_value=record.reference_value,
            objective_transform=record.objective_transform,
            manifest_hash=record.manifest_hash,
            interaction_frequency=record.interaction_frequency,
        )


@dataclass(frozen=True)
class EvaluationRecord:
    """One tidy final/anytime result for a single method and context."""

    domain: Domain
    scenario_or_function: str
    dimension: int | None
    task_id: str
    native_instance: str
    inner_seed: int
    outer_ppo_seed: int | None
    method: str
    action_family: str
    checkpoint_type: str
    evaluation_budget: int
    reference_kind: ReferenceKind
    reference_value: float
    objective_transform: str
    final_incumbent: float
    final_reference_regret: float
    normalized_final_regret: float
    anytime_auc: float
    episode_return: float
    action_histogram: tuple[int, ...]
    deterministic_switch_rate: float
    constant_policy: bool
    runtime_seconds: float
    manifest_hash: str
    code_commit: str
    interaction_frequency: int = 1

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Reject malformed rows before they can enter paired statistics."""
        if self.domain not in {"bbob", "yahpo"}:
            raise ValueError(f"Unsupported record domain {self.domain!r}.")
        for field_name in (
            "scenario_or_function",
            "task_id",
            "native_instance",
            "method",
            "action_family",
            "checkpoint_type",
            "objective_transform",
            "manifest_hash",
            "code_commit",
        ):
            if not str(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.domain == "bbob" and (self.dimension is None or self.dimension <= 0):
            raise ValueError("BBOB records require a positive dimension.")
        if self.domain == "yahpo" and self.dimension is not None:
            raise ValueError("YAHPO records must use dimension=None.")
        if self.inner_seed < 0 or (self.outer_ppo_seed is not None and self.outer_ppo_seed < 0):
            raise ValueError("Seeds must be non-negative integers.")
        if self.evaluation_budget <= 0:
            raise ValueError("evaluation_budget must be positive.")
        if self.reference_kind not in {"exact", "best_known"}:
            raise ValueError(f"Unsupported reference kind {self.reference_kind!r}.")
        numeric_fields = (
            "reference_value",
            "final_incumbent",
            "final_reference_regret",
            "normalized_final_regret",
            "anytime_auc",
            "episode_return",
            "deterministic_switch_rate",
            "runtime_seconds",
        )
        for field_name in numeric_fields:
            if not np.isfinite(float(getattr(self, field_name))):
                raise ValueError(f"{field_name} must be finite.")
        if self.final_reference_regret < 0.0 or self.normalized_final_regret < 0.0:
            raise ValueError("Reference regrets must be non-negative.")
        if not 0.0 <= self.deterministic_switch_rate <= 1.0:
            raise ValueError("deterministic_switch_rate must lie in [0, 1].")
        if self.runtime_seconds < 0.0:
            raise ValueError("runtime_seconds must be non-negative.")
        histogram = tuple(self.action_histogram)
        if any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in histogram):
            raise ValueError("action_histogram must contain non-negative integer counts.")
        object.__setattr__(self, "action_histogram", histogram)

    @property
    def context_key(self) -> ContextKey:
        """Return the exact scientific context represented by this row."""
        return EvaluationContext.from_record(self).key

    @property
    def method_cell(self) -> MethodCell:
        """Return the independently reportable method cell."""
        return MethodCell.from_record(self)

    def to_tidy_row(self) -> dict[str, Any]:
        """Serialize the row with a stable CSV-friendly histogram field."""
        row = asdict(self)
        row["action_histogram"] = json.dumps(self.action_histogram, separators=(",", ":"))
        return {field: row[field] for field in TIDY_RECORD_FIELDS}


@dataclass(frozen=True)
class EvaluationMethod:
    """A declared evaluator method; declaration does not imply execution."""

    name: str
    requires_trained_model: bool = False
    derived_from_validation: bool = False
    analysis_only: bool = False


MethodRunner = Callable[[EvaluationContext, EvaluationMethod], EvaluationRecord]


class MethodRegistry:
    """Registry of method names and explicitly supplied runner callbacks."""

    def __init__(self, *, n_static_actions: int = 5) -> None:
        if n_static_actions <= 0:
            raise ValueError("n_static_actions must be positive.")
        self._methods: dict[str, EvaluationMethod] = {}
        self._runners: dict[str, MethodRunner] = {}
        for method in (
            EvaluationMethod(LEARNED_VALIDATION_SELECTED, requires_trained_model=True),
            EvaluationMethod(LEARNED_FINAL, requires_trained_model=True),
            EvaluationMethod(MODAL_STATIC_CLONE, derived_from_validation=True),
            EvaluationMethod(MARGINAL_RANDOM_CONTROL, derived_from_validation=True),
            EvaluationMethod(BEST_VALIDATION_STATIC, derived_from_validation=True),
            EvaluationMethod(UNIFORM_RANDOM),
            EvaluationMethod(DEFAULT_SMAC),
            EvaluationMethod(SAWEI),
            EvaluationMethod(DYNAMIC_ORACLE, analysis_only=True),
        ):
            self.register_method(method)
        for action in range(n_static_actions):
            self.register_method(EvaluationMethod(f"{STATIC_ACTION_PREFIX}{action}"))

    @property
    def method_names(self) -> tuple[str, ...]:
        """Return all declared methods in stable lexical order."""
        return tuple(sorted(self._methods))

    def register_method(self, method: EvaluationMethod) -> None:
        """Declare a method without claiming that it has an implementation."""
        if not method.name:
            raise ValueError("A method name must be non-empty.")
        if method.name in self._methods:
            raise ValueError(f"Method {method.name!r} is already registered.")
        self._methods[method.name] = method

    def register_runner(self, method_name: str, runner: MethodRunner) -> None:
        """Attach the concrete callback required to execute one method."""
        if method_name not in self._methods:
            raise KeyError(f"Unknown evaluation method {method_name!r}.")
        if method_name in self._runners:
            raise ValueError(f"A runner is already registered for {method_name!r}.")
        self._runners[method_name] = runner

    def method(self, method_name: str) -> EvaluationMethod:
        """Return a declared method or fail clearly."""
        try:
            return self._methods[method_name]
        except KeyError as error:
            raise KeyError(f"Unknown evaluation method {method_name!r}.") from error

    def runner(self, method_name: str) -> MethodRunner:
        """Return a concrete runner without inventing a fallback."""
        self.method(method_name)
        try:
            return self._runners[method_name]
        except KeyError as error:
            raise MethodRunnerUnavailableError(
                f"Evaluation method {method_name!r} is declared but has no runner callback."
            ) from error


def authorize_manifest_execution(
    manifest: Mapping[str, Any],
    *,
    allow_sealed_test: bool = False,
) -> None:
    """Validate a runnable manifest and require an explicit test-set flag."""
    validate_manifest_structure(manifest)
    require_runnable_manifest(manifest)
    if manifest["split"] == "test" and not allow_sealed_test:
        raise SealedManifestError(
            f"Manifest {manifest['id']!r} is sealed final-test data; "
            "pass allow_sealed_test=True only for an authorized final report."
        )


def validate_contexts_against_manifest(
    contexts: Sequence[EvaluationContext],
    manifest: Mapping[str, Any],
) -> None:
    """Require exactly one context for every frozen task/inner-seed pair."""
    if any(seed is None for seed in manifest["inner_seeds"]):
        raise PairingError("Evaluation manifests must contain only frozen integer inner seeds.")
    expected_pairs = {(str(task_id), int(seed)) for seed in manifest["inner_seeds"] for task_id in manifest["task_ids"]}
    actual_pairs = [(context.task_id, context.inner_seed) for context in contexts]
    if len(set(actual_pairs)) != len(actual_pairs):
        raise PairingError("Evaluation contexts contain duplicate task/inner-seed pairs.")
    if set(actual_pairs) != expected_pairs:
        missing = sorted(expected_pairs.difference(actual_pairs))
        extra = sorted(set(actual_pairs).difference(expected_pairs))
        raise PairingError(f"Contexts do not match manifest: missing={missing}, extra={extra}.")
    stale_hashes = sorted(
        {context.manifest_hash for context in contexts if context.manifest_hash != manifest["manifest_hash"]}
    )
    if stale_hashes:
        raise PairingError(f"Contexts carry manifest hashes {stale_hashes}, expected {manifest['manifest_hash']}.")


def evaluate_registered_methods(
    manifest: Mapping[str, Any],
    contexts: Sequence[EvaluationContext],
    method_names: Sequence[str],
    registry: MethodRegistry,
    *,
    allow_sealed_test: bool = False,
    allow_analysis_only: bool = False,
) -> list[EvaluationRecord]:
    """Execute only explicitly registered callbacks on one paired manifest.

    Every requested runner is resolved before the first callback is invoked, so
    a missing implementation cannot leave a misleading partial result set.
    """
    authorize_manifest_execution(manifest, allow_sealed_test=allow_sealed_test)
    validate_contexts_against_manifest(contexts, manifest)
    if not method_names:
        raise ValueError("At least one evaluation method is required.")
    if len(set(method_names)) != len(method_names):
        raise ValueError("Evaluation method names must be unique.")

    resolved: list[tuple[EvaluationMethod, MethodRunner]] = []
    for method_name in method_names:
        method = registry.method(method_name)
        if method.analysis_only and not allow_analysis_only:
            raise AnalysisOnlyMethodError(
                f"Method {method_name!r} is analysis-only; pass allow_analysis_only=True explicitly."
            )
        resolved.append((method, registry.runner(method_name)))

    records: list[EvaluationRecord] = []
    for method, runner in resolved:
        for context in contexts:
            record = runner(context, method)
            if record.method != method.name:
                raise PairingError(f"Runner {method.name!r} returned a row labeled {record.method!r}.")
            if record.context_key != context.key:
                raise PairingError(f"Runner {method.name!r} changed its assigned context.")
            records.append(record)
    validate_paired_contexts(records)
    return records


def available_method_cells(records: Iterable[EvaluationRecord]) -> tuple[MethodCell, ...]:
    """Return stable unique method cells represented in a record collection."""
    return tuple(sorted({record.method_cell for record in records}, key=MethodCell.sort_key))


def _records_by_cell(
    records: Iterable[EvaluationRecord],
) -> dict[MethodCell, dict[ContextKey, EvaluationRecord]]:
    grouped: dict[MethodCell, dict[ContextKey, EvaluationRecord]] = defaultdict(dict)
    for record in records:
        cell_records = grouped[record.method_cell]
        if record.context_key in cell_records:
            raise PairingError(
                f"Duplicate result for method cell {record.method_cell!r} and context {record.context_key!r}."
            )
        cell_records[record.context_key] = record
    return dict(grouped)


def validate_paired_contexts(records: Sequence[EvaluationRecord]) -> tuple[ContextKey, ...]:
    """Prove that every represented method cell covers exactly the same contexts."""
    if not records:
        raise PairingError("At least one evaluation record is required.")
    grouped = _records_by_cell(records)
    reference_cell = min(grouped, key=MethodCell.sort_key)
    reference_contexts = set(grouped[reference_cell])
    for cell, cell_records in grouped.items():
        contexts = set(cell_records)
        if contexts != reference_contexts:
            missing = sorted(reference_contexts.difference(contexts), key=ContextKey.sort_key)
            extra = sorted(contexts.difference(reference_contexts), key=ContextKey.sort_key)
            raise PairingError(
                f"Method cell {cell!r} is not exactly paired with {reference_cell!r}: missing={missing}, extra={extra}."
            )
    return tuple(sorted(reference_contexts, key=ContextKey.sort_key))


def validate_records_against_manifest(
    records: Sequence[EvaluationRecord],
    manifest: Mapping[str, Any],
) -> None:
    """Require one method cell and exact coverage of a frozen manifest."""
    cells = available_method_cells(records)
    if len(cells) != 1:
        raise PairingError(f"Expected exactly one source method cell, found {cells}.")
    contexts = [EvaluationContext.from_record(record) for record in records]
    validate_contexts_against_manifest(contexts, manifest)


@dataclass(frozen=True)
class DerivedControlProvenance:
    """Validation-only source metadata for modal and marginal controls."""

    schema_version: int
    source_split: str
    source_method: str
    source_action_family: str
    source_checkpoint: str
    source_outer_ppo_seed: int
    source_validation_manifest_hash: str
    source_code_commit: str
    source_record_count: int
    source_action_counts: tuple[int, ...]
    source_action_frequencies: tuple[float, ...]
    modal_action: int
    modal_control_method: str = MODAL_STATIC_CLONE
    marginal_control_method: str = MARGINAL_RANDOM_CONTROL

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible provenance without dropping exact counts."""
        payload = asdict(self)
        payload["source_action_counts"] = list(self.source_action_counts)
        payload["source_action_frequencies"] = list(self.source_action_frequencies)
        return payload

    def save(self, path: Path) -> None:
        """Persist the derivation inputs for later test-set evaluation."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derive_validation_controls(
    records: Sequence[EvaluationRecord],
    manifest: Mapping[str, Any],
    *,
    output_path: Path | None = None,
) -> DerivedControlProvenance:
    """Derive modal and marginal controls from one complete validation cell."""
    validate_manifest_structure(manifest)
    if manifest["split"] != "validation":
        raise ControlDerivationError(
            "Modal and marginal controls may be derived only from a validation manifest, "
            f"not split={manifest['split']!r}."
        )
    require_runnable_manifest(manifest)
    if not records:
        raise ControlDerivationError("Cannot derive controls from an empty record collection.")
    validate_records_against_manifest(records, manifest)

    cell = records[0].method_cell
    if cell.method not in {LEARNED_VALIDATION_SELECTED, LEARNED_FINAL}:
        raise ControlDerivationError(f"Control provenance requires a learned source method, got {cell.method!r}.")
    if cell.outer_ppo_seed is None:
        raise ControlDerivationError("Learned-policy control derivation requires an outer PPO seed.")
    commits = {record.code_commit for record in records}
    if len(commits) != 1:
        raise ControlDerivationError(f"Source records span multiple code commits: {sorted(commits)}.")
    histogram_widths = {len(record.action_histogram) for record in records}
    if len(histogram_widths) != 1 or next(iter(histogram_widths)) < _MIN_CATEGORICAL_ACTIONS:
        raise ControlDerivationError("Source action histograms must share a width of at least two.")

    action_counts = np.sum(np.asarray([record.action_histogram for record in records], dtype=np.int64), axis=0)
    total_actions = int(np.sum(action_counts))
    if total_actions <= 0:
        raise ControlDerivationError("Source action histograms contain no policy decisions.")
    frequencies = action_counts.astype(float) / total_actions
    provenance = DerivedControlProvenance(
        schema_version=1,
        source_split="validation",
        source_method=cell.method,
        source_action_family=cell.action_family,
        source_checkpoint=cell.checkpoint_type,
        source_outer_ppo_seed=cell.outer_ppo_seed,
        source_validation_manifest_hash=str(manifest["manifest_hash"]),
        source_code_commit=next(iter(commits)),
        source_record_count=len(records),
        source_action_counts=tuple(int(value) for value in action_counts),
        source_action_frequencies=tuple(float(value) for value in frequencies),
        modal_action=int(np.argmax(action_counts)),
    )
    if output_path is not None:
        provenance.save(output_path)
    return provenance


def _metric_direction(metric: str, *, higher_is_better: bool | None) -> bool:
    if higher_is_better is not None:
        return higher_is_better
    if metric in _HIGHER_IS_BETTER_METRICS:
        return True
    if metric in _LOWER_IS_BETTER_METRICS:
        return False
    raise ValueError(f"Unknown metric direction for {metric!r}; pass higher_is_better explicitly.")


def _cell_records(
    records: Sequence[EvaluationRecord],
    cell: MethodCell,
) -> dict[ContextKey, EvaluationRecord]:
    grouped = _records_by_cell(records)
    try:
        return grouped[cell]
    except KeyError as error:
        raise PairingError(f"Method cell {cell!r} is absent from the supplied records.") from error


@dataclass(frozen=True)
class PairedComparison:
    """Finite paired effects where positive differences favor the method."""

    method_cell: MethodCell
    baseline_cell: MethodCell
    metric: str
    higher_is_better: bool
    n_contexts: int
    mean_difference: float
    median_difference: float
    wins: int
    ties: int
    losses: int
    win_loss_effect: float
    standardized_mean_difference: float | None
    contexts: tuple[ContextKey, ...]
    differences: tuple[float, ...]


def paired_method_comparison(
    records: Sequence[EvaluationRecord],
    method_cell: MethodCell,
    baseline_cell: MethodCell,
    *,
    metric: str = "normalized_final_regret",
    higher_is_better: bool | None = None,
    tie_tolerance: float = 1e-12,
) -> PairedComparison:
    """Compute paired differences, win/tie/loss counts, and effect sizes."""
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative.")
    direction = _metric_direction(metric, higher_is_better=higher_is_better)
    method_records = _cell_records(records, method_cell)
    baseline_records = _cell_records(records, baseline_cell)
    if set(method_records) != set(baseline_records):
        raise PairingError("The selected method and baseline cells do not share exact contexts.")

    ordered_contexts = sorted(method_records, key=ContextKey.sort_key)
    differences: list[float] = []
    for context in ordered_contexts:
        try:
            method_value = float(getattr(method_records[context], metric))
            baseline_value = float(getattr(baseline_records[context], metric))
        except AttributeError as error:
            raise ValueError(f"EvaluationRecord has no metric {metric!r}.") from error
        raw_difference = method_value - baseline_value
        differences.append(raw_difference if direction else -raw_difference)

    values = np.asarray(differences, dtype=float)
    wins = int(np.sum(values > tie_tolerance))
    ties = int(np.sum(np.abs(values) <= tie_tolerance))
    losses = int(np.sum(values < -tie_tolerance))
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    standardized_effect = None if standard_deviation == 0.0 else float(np.mean(values) / standard_deviation)
    return PairedComparison(
        method_cell=method_cell,
        baseline_cell=baseline_cell,
        metric=metric,
        higher_is_better=direction,
        n_contexts=len(values),
        mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)),
        wins=wins,
        ties=ties,
        losses=losses,
        win_loss_effect=float((wins - losses) / len(values)),
        standardized_mean_difference=standardized_effect,
        contexts=tuple(ordered_contexts),
        differences=tuple(float(value) for value in values),
    )


@dataclass(frozen=True)
class OuterSeedBeatProbability:
    """Cross-outer-seed frequency of a learned cell beating one baseline."""

    method: str
    action_family: str
    checkpoint_type: str
    baseline_cell: MethodCell
    metric: str
    n_outer_seeds: int
    beating_seeds: int
    tying_seeds: int
    losing_seeds: int
    probability_beating: float
    mean_difference_by_seed: Mapping[int, float]


def outer_seed_beat_probability(
    records: Sequence[EvaluationRecord],
    *,
    method: str,
    action_family: str,
    checkpoint_type: str,
    baseline_cell: MethodCell,
    metric: str = "normalized_final_regret",
    higher_is_better: bool | None = None,
    tie_tolerance: float = 1e-12,
) -> OuterSeedBeatProbability:
    """Aggregate paired context effects once per outer PPO seed."""
    learned_cells = [
        cell
        for cell in available_method_cells(records)
        if cell.method == method
        and cell.action_family == action_family
        and cell.checkpoint_type == checkpoint_type
        and cell.outer_ppo_seed is not None
    ]
    if not learned_cells:
        raise PairingError("No learned outer-seed cells match the requested selector.")

    mean_by_seed: dict[int, float] = {}
    for cell in learned_cells:
        assert cell.outer_ppo_seed is not None
        comparison = paired_method_comparison(
            records,
            cell,
            baseline_cell,
            metric=metric,
            higher_is_better=higher_is_better,
            tie_tolerance=tie_tolerance,
        )
        if cell.outer_ppo_seed in mean_by_seed:
            raise PairingError(f"Duplicate learned method cell for outer seed {cell.outer_ppo_seed}.")
        mean_by_seed[cell.outer_ppo_seed] = comparison.mean_difference

    values = np.asarray(list(mean_by_seed.values()), dtype=float)
    beating = int(np.sum(values > tie_tolerance))
    tying = int(np.sum(np.abs(values) <= tie_tolerance))
    losing = int(np.sum(values < -tie_tolerance))
    return OuterSeedBeatProbability(
        method=method,
        action_family=action_family,
        checkpoint_type=checkpoint_type,
        baseline_cell=baseline_cell,
        metric=metric,
        n_outer_seeds=len(values),
        beating_seeds=beating,
        tying_seeds=tying,
        losing_seeds=losing,
        probability_beating=float(beating / len(values)),
        mean_difference_by_seed=dict(sorted(mean_by_seed.items())),
    )


@dataclass(frozen=True)
class _PairedUnit:
    record: EvaluationRecord
    difference: float


def _bbob_function(unit: _PairedUnit) -> int:
    parts = unit.record.task_id.split("/")
    if len(parts) != _BBOB_TASK_ID_PARTS or parts[0].lower() != "bbob":
        raise PairingError(f"Invalid BBOB task ID in evaluation record: {unit.record.task_id!r}.")
    return int(parts[2])


def _bbob_function_group(unit: _PairedUnit) -> int:
    function_id = _bbob_function(unit)
    for group, upper_bound in enumerate(_BBOB_FUNCTION_GROUP_UPPER_BOUNDS):
        if 1 <= function_id <= upper_bound:
            return group
    maximum = _BBOB_FUNCTION_GROUP_UPPER_BOUNDS[-1]
    raise PairingError(f"BBOB function ID must lie in [1, {maximum}], got {function_id}.")


GroupingKey = Callable[[_PairedUnit], Any]


def _nested_mean(units: Sequence[_PairedUnit], grouping_keys: Sequence[GroupingKey]) -> float:
    if not grouping_keys:
        return float(np.mean([unit.difference for unit in units]))
    groups: dict[Any, list[_PairedUnit]] = defaultdict(list)
    for unit in units:
        groups[grouping_keys[0](unit)].append(unit)
    return float(np.mean([_nested_mean(group, grouping_keys[1:]) for group in groups.values()]))


def _resampled_nested_mean(
    units: Sequence[_PairedUnit],
    grouping_keys: Sequence[GroupingKey],
    rng: np.random.Generator,
) -> float:
    if not grouping_keys:
        sampled = rng.integers(0, len(units), size=len(units))
        return float(np.mean([units[int(index)].difference for index in sampled]))
    groups: dict[Any, list[_PairedUnit]] = defaultdict(list)
    for unit in units:
        groups[grouping_keys[0](unit)].append(unit)
    group_values = list(groups.values())
    sampled = rng.integers(0, len(group_values), size=len(group_values))
    return float(
        np.mean([_resampled_nested_mean(group_values[int(index)], grouping_keys[1:], rng) for index in sampled])
    )


_BBOB_GROUPING_KEYS: tuple[GroupingKey, ...] = (
    _bbob_function_group,
    _bbob_function,
    lambda unit: (unit.record.task_id, unit.record.native_instance),
    lambda unit: unit.record.inner_seed,
)
_YAHPO_GROUPING_KEYS: tuple[GroupingKey, ...] = (
    lambda unit: unit.record.scenario_or_function,
    lambda unit: (unit.record.task_id, unit.record.native_instance),
    lambda unit: unit.record.inner_seed,
)


@dataclass(frozen=True)
class HierarchicalValidationScore:
    """One validation cell aggregated with benchmark-aware equal weighting.

    All fields are oriented so that larger is better. Lower-is-better record
    metrics (for example normalized reference regret) are negated before the
    hierarchy is aggregated. A mixed score is always exactly 0.5 BBOB plus
    0.5 YAHPO; a single-domain score equals its available domain score.
    """

    method_cell: MethodCell
    metric: str
    higher_is_better: bool
    balanced_score: float
    bbob_score: float | None
    yahpo_score: float | None
    worst_domain_score: float
    per_bbob_dimension: Mapping[int, float]
    per_yahpo_scenario: Mapping[str, float]


def aggregate_validation_score(
    records: Sequence[EvaluationRecord],
    *,
    metric: str = "normalized_final_regret",
    higher_is_better: bool | None = None,
) -> HierarchicalValidationScore:
    """Aggregate one method cell without pooling heterogeneous episodes.

    BBOB follows function group -> function -> native task -> inner seed
    independently within every dimension, followed by equal dimension
    weighting. YAHPO follows scenario -> dataset instance -> inner seed,
    followed by equal scenario weighting.
    """
    cells = available_method_cells(records)
    if len(cells) != 1:
        raise PairingError(f"Validation aggregation requires exactly one method cell, found {cells}.")
    direction = _metric_direction(metric, higher_is_better=higher_is_better)
    units: list[_PairedUnit] = []
    for record in records:
        try:
            value = float(getattr(record, metric))
        except AttributeError as error:
            raise ValueError(f"EvaluationRecord has no metric {metric!r}.") from error
        units.append(_PairedUnit(record=record, difference=value if direction else -value))

    bbob_by_dimension: dict[int, list[_PairedUnit]] = defaultdict(list)
    yahpo_by_scenario: dict[str, list[_PairedUnit]] = defaultdict(list)
    for unit in units:
        if unit.record.domain == "bbob":
            assert unit.record.dimension is not None
            bbob_by_dimension[unit.record.dimension].append(unit)
        else:
            yahpo_by_scenario[unit.record.scenario_or_function].append(unit)

    per_dimension = {
        dimension: _nested_mean(dimension_units, _BBOB_GROUPING_KEYS)
        for dimension, dimension_units in sorted(bbob_by_dimension.items())
    }
    yahpo_instance_seed_keys: tuple[GroupingKey, ...] = (
        lambda unit: (unit.record.task_id, unit.record.native_instance),
        lambda unit: unit.record.inner_seed,
    )
    per_scenario = {
        scenario: _nested_mean(scenario_units, yahpo_instance_seed_keys)
        for scenario, scenario_units in sorted(yahpo_by_scenario.items())
    }
    bbob_score = float(np.mean(list(per_dimension.values()))) if per_dimension else None
    yahpo_score = float(np.mean(list(per_scenario.values()))) if per_scenario else None
    domain_scores = [score for score in (bbob_score, yahpo_score) if score is not None]
    if not domain_scores:
        raise PairingError("Validation aggregation requires at least one BBOB or YAHPO record.")
    balanced_score = (
        0.5 * bbob_score + 0.5 * yahpo_score if bbob_score is not None and yahpo_score is not None else domain_scores[0]
    )
    return HierarchicalValidationScore(
        method_cell=cells[0],
        metric=metric,
        higher_is_better=direction,
        balanced_score=float(balanced_score),
        bbob_score=bbob_score,
        yahpo_score=yahpo_score,
        worst_domain_score=float(min(domain_scores)),
        per_bbob_dimension=per_dimension,
        per_yahpo_scenario=per_scenario,
    )


@dataclass(frozen=True)
class BestStaticSelectionProvenance:
    """Validation-only selection of one deployable static action."""

    schema_version: int
    source_split: str
    source_action_family: str
    source_validation_manifest_hash: str
    source_metric: str
    source_code_commit: str
    source_static_scores: Mapping[int, float]
    selected_action: int
    method: str = BEST_VALIDATION_STATIC

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, auditable selection record."""
        payload = asdict(self)
        payload["source_static_scores"] = {
            str(action): float(score) for action, score in sorted(self.source_static_scores.items())
        }
        return payload

    def save(self, path: Path) -> None:
        """Persist the frozen validation selection for later paired use."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def select_validation_static_action(
    records: Sequence[EvaluationRecord],
    manifest: Mapping[str, Any],
    *,
    n_actions: int,
    metric: str = "normalized_final_regret",
    output_path: Path | None = None,
) -> BestStaticSelectionProvenance:
    """Select one static action on a complete validation panel only.

    This selects one action globally under the benchmark-aware validation
    hierarchy. It is not a per-task virtual best solver.
    """
    validate_manifest_structure(manifest)
    if manifest["split"] != "validation":
        raise ControlDerivationError("Best-static selection is allowed only on a validation manifest.")
    require_runnable_manifest(manifest)
    if isinstance(n_actions, bool) or not isinstance(n_actions, int) or n_actions <= 0:
        raise ValueError("n_actions must be a positive integer.")

    cells = available_method_cells(records)
    expected_methods = {f"{STATIC_ACTION_PREFIX}{action}" for action in range(n_actions)}
    actual_methods = {cell.method for cell in cells}
    if actual_methods != expected_methods or len(cells) != n_actions:
        raise ControlDerivationError(
            "Best-static selection requires exactly one complete cell for every static action; "
            f"expected={sorted(expected_methods)}, got={sorted(actual_methods)}."
        )
    action_families = {cell.action_family for cell in cells}
    if len(action_families) != 1:
        raise ControlDerivationError(f"Static source cells span action families: {sorted(action_families)}.")
    if any(cell.outer_ppo_seed is not None for cell in cells):
        raise ControlDerivationError("Static validation source cells must not carry outer PPO seeds.")

    scores: dict[int, float] = {}
    commits: set[str] = set()
    for cell in cells:
        cell_records = [record for record in records if record.method_cell == cell]
        validate_records_against_manifest(cell_records, manifest)
        action = int(cell.method.removeprefix(STATIC_ACTION_PREFIX))
        scores[action] = aggregate_validation_score(cell_records, metric=metric).balanced_score
        commits.update(record.code_commit for record in cell_records)
    if len(commits) != 1:
        raise ControlDerivationError(f"Static validation records span code commits: {sorted(commits)}.")
    selected_action = min(scores, key=lambda action: (-scores[action], action))
    provenance = BestStaticSelectionProvenance(
        schema_version=1,
        source_split="validation",
        source_action_family=next(iter(action_families)),
        source_validation_manifest_hash=str(manifest["manifest_hash"]),
        source_metric=metric,
        source_code_commit=next(iter(commits)),
        source_static_scores=dict(sorted(scores.items())),
        selected_action=selected_action,
    )
    if output_path is not None:
        provenance.save(output_path)
    return provenance


@dataclass(frozen=True)
class ValidationCheckpointEvaluation:
    """Full/frequent validation metadata kept separate from model artifacts."""

    checkpoint_id: str
    training_step: int
    panel_tier: Literal["frequent", "full"]
    is_step_zero: bool
    trained: bool
    score: HierarchicalValidationScore

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must be non-empty.")
        if isinstance(self.training_step, bool) or not isinstance(self.training_step, int):
            raise ValueError("training_step must be an integer.")
        if self.training_step < 0:
            raise ValueError("training_step must be non-negative.")
        if self.is_step_zero != (self.training_step == 0):
            raise ValueError("is_step_zero must be true exactly when training_step is zero.")
        if self.is_step_zero and self.trained:
            raise ValueError("The step-zero policy is untrained by definition.")
        if not self.is_step_zero and not self.trained:
            raise ValueError("Every positive-step checkpoint must be marked trained.")


CheckpointTarget = Literal["balanced", "bbob", "yahpo"]


def _checkpoint_target_score(
    candidate: ValidationCheckpointEvaluation,
    target: CheckpointTarget,
) -> float | None:
    if target == "balanced":
        return candidate.score.balanced_score
    if target == "bbob":
        return candidate.score.bbob_score
    if target == "yahpo":
        return candidate.score.yahpo_score
    raise ValueError(f"Unsupported checkpoint target {target!r}.")


def select_full_panel_checkpoint(
    candidates: Sequence[ValidationCheckpointEvaluation],
    *,
    target: CheckpointTarget = "balanced",
) -> ValidationCheckpointEvaluation:
    """Select only a trained positive-step checkpoint evaluated on the full panel.

    Frequent-panel scores can nominate checkpoints for full evaluation but can
    never select the reported model. Step-zero diagnostics are ineligible even
    if their score exceeds every trained checkpoint.
    """
    eligible = [
        candidate
        for candidate in candidates
        if candidate.panel_tier == "full"
        and candidate.trained
        and not candidate.is_step_zero
        and candidate.training_step > 0
        and _checkpoint_target_score(candidate, target) is not None
    ]
    if not eligible:
        raise CheckpointSelectionError(f"No trained positive-step full-panel checkpoint has a {target!r} score.")
    metric_protocols = {(candidate.score.metric, candidate.score.higher_is_better) for candidate in eligible}
    if len(metric_protocols) != 1:
        raise CheckpointSelectionError(
            f"Eligible checkpoints use incompatible metric protocols: {sorted(metric_protocols)}."
        )
    return sorted(
        eligible,
        key=lambda candidate: (
            -float(_checkpoint_target_score(candidate, target)),
            candidate.training_step,
            candidate.checkpoint_id,
        ),
    )[0]


@dataclass(frozen=True)
class IntervalEstimate:
    """Point estimate and percentile bootstrap confidence interval."""

    point_estimate: float
    confidence_lower: float
    confidence_upper: float


@dataclass(frozen=True)
class HierarchicalBootstrapResult:
    """Paired hierarchical bootstrap with fixed mixed-domain weighting."""

    method_cell: MethodCell
    baseline_cell: MethodCell
    metric: str
    higher_is_better: bool
    confidence_level: float
    n_resamples: int
    bootstrap_seed: int
    overall: IntervalEstimate
    per_domain: Mapping[str, IntervalEstimate]
    per_bbob_dimension: Mapping[int, IntervalEstimate]


def _interval(point: float, samples: Sequence[float], confidence_level: float) -> IntervalEstimate:
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(np.asarray(samples, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return IntervalEstimate(
        point_estimate=float(point),
        confidence_lower=float(lower),
        confidence_upper=float(upper),
    )


def _paired_units(
    records: Sequence[EvaluationRecord],
    method_cell: MethodCell,
    baseline_cell: MethodCell,
    metric: str,
    *,
    higher_is_better: bool,
) -> list[_PairedUnit]:
    method_records = _cell_records(records, method_cell)
    baseline_records = _cell_records(records, baseline_cell)
    if set(method_records) != set(baseline_records):
        raise PairingError("Hierarchical bootstrap requires exact paired contexts.")
    units: list[_PairedUnit] = []
    for context in sorted(method_records, key=ContextKey.sort_key):
        try:
            method_value = float(getattr(method_records[context], metric))
            baseline_value = float(getattr(baseline_records[context], metric))
        except AttributeError as error:
            raise ValueError(f"EvaluationRecord has no metric {metric!r}.") from error
        difference = method_value - baseline_value
        if not higher_is_better:
            difference = -difference
        units.append(_PairedUnit(record=method_records[context], difference=difference))
    return units


def hierarchical_paired_bootstrap(  # noqa: C901, PLR0912
    records: Sequence[EvaluationRecord],
    method_cell: MethodCell,
    baseline_cell: MethodCell,
    *,
    metric: str = "normalized_final_regret",
    higher_is_better: bool | None = None,
    n_resamples: int = 2_000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> HierarchicalBootstrapResult:
    """Bootstrap paired contexts at the benchmark's scientific hierarchy.

    BBOB is resampled as function group -> function -> native task -> inner
    seed within each fixed dimension, after which dimensions receive equal
    weight. YAHPO is resampled as scenario -> dataset instance -> inner seed.
    When both domains are present, every replicate uses an exact 0.5/0.5 domain
    weight. BO iterations never enter the sample units.
    """
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    direction = _metric_direction(metric, higher_is_better=higher_is_better)
    units = _paired_units(
        records,
        method_cell,
        baseline_cell,
        metric,
        higher_is_better=direction,
    )
    domains = {unit.record.domain for unit in units}
    if not domains.issubset({"bbob", "yahpo"}):
        raise PairingError(f"Unsupported bootstrap domains: {sorted(domains)}.")

    bbob_by_dimension: dict[int, list[_PairedUnit]] = defaultdict(list)
    yahpo_units: list[_PairedUnit] = []
    for unit in units:
        if unit.record.domain == "bbob":
            assert unit.record.dimension is not None
            bbob_by_dimension[unit.record.dimension].append(unit)
        else:
            yahpo_units.append(unit)

    dimension_points = {
        dimension: _nested_mean(dimension_units, _BBOB_GROUPING_KEYS)
        for dimension, dimension_units in sorted(bbob_by_dimension.items())
    }
    domain_points: dict[str, float] = {}
    if dimension_points:
        domain_points["bbob"] = float(np.mean(list(dimension_points.values())))
    if yahpo_units:
        domain_points["yahpo"] = _nested_mean(yahpo_units, _YAHPO_GROUPING_KEYS)
    overall_point = (
        0.5 * domain_points["bbob"] + 0.5 * domain_points["yahpo"]
        if set(domain_points) == {"bbob", "yahpo"}
        else next(iter(domain_points.values()))
    )

    rng = np.random.default_rng(seed)
    overall_samples: list[float] = []
    domain_samples: dict[str, list[float]] = defaultdict(list)
    dimension_samples: dict[int, list[float]] = defaultdict(list)
    for _ in range(n_resamples):
        replicate_domains: dict[str, float] = {}
        if bbob_by_dimension:
            replicate_dimensions: list[float] = []
            for dimension, dimension_units in sorted(bbob_by_dimension.items()):
                value = _resampled_nested_mean(dimension_units, _BBOB_GROUPING_KEYS, rng)
                dimension_samples[dimension].append(value)
                replicate_dimensions.append(value)
            replicate_domains["bbob"] = float(np.mean(replicate_dimensions))
            domain_samples["bbob"].append(replicate_domains["bbob"])
        if yahpo_units:
            replicate_domains["yahpo"] = _resampled_nested_mean(yahpo_units, _YAHPO_GROUPING_KEYS, rng)
            domain_samples["yahpo"].append(replicate_domains["yahpo"])
        if set(replicate_domains) == {"bbob", "yahpo"}:
            overall_samples.append(0.5 * replicate_domains["bbob"] + 0.5 * replicate_domains["yahpo"])
        else:
            overall_samples.append(next(iter(replicate_domains.values())))

    return HierarchicalBootstrapResult(
        method_cell=method_cell,
        baseline_cell=baseline_cell,
        metric=metric,
        higher_is_better=direction,
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        bootstrap_seed=seed,
        overall=_interval(overall_point, overall_samples, confidence_level),
        per_domain={
            domain: _interval(point, domain_samples[domain], confidence_level)
            for domain, point in domain_points.items()
        },
        per_bbob_dimension={
            dimension: _interval(point, dimension_samples[dimension], confidence_level)
            for dimension, point in dimension_points.items()
        },
    )


_POLICY_OBSERVATION_KEYS = frozenset({"global_state", "action_features"})
_STATE_SENSITIVITY_EPSILON = 1e-12
_N_BUDGET_PHASES = 4


def validation_budget_phase(budget_fraction: float) -> int:
    """Map a finite fraction in [0, 1] to one of four frozen budget phases."""
    if not np.isfinite(budget_fraction) or not 0.0 <= budget_fraction <= 1.0:
        raise ValueError("budget_fraction must be finite and lie in [0, 1].")
    return min(int(np.floor(float(budget_fraction) * _N_BUDGET_PHASES)), _N_BUDGET_PHASES - 1)


@dataclass(frozen=True)
class PolicyStateSample:
    """Policy-visible state with non-visible provenance for sensitivity tests."""

    task_id: str
    budget_fraction: float
    observation: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError("task_id must be non-empty.")
        validation_budget_phase(self.budget_fraction)
        if set(self.observation) != _POLICY_OBSERVATION_KEYS:
            raise ValueError("Policy observations must contain exactly global_state and action_features.")
        for name, value in self.observation.items():
            array = np.asarray(value)
            if array.size == 0 or not np.all(np.isfinite(array)):
                raise ValueError(f"Observation component {name!r} must be non-empty and finite.")

    @property
    def budget_phase(self) -> int:
        """Return the frozen quarter-budget phase used for substitutions."""
        return validation_budget_phase(self.budget_fraction)


@dataclass(frozen=True)
class StateSubstitutionSensitivity:
    """Actor-distribution change plus provenance proving distinctness."""

    intervention: Literal["state_from_another_task", "state_from_another_budget_phase"]
    source_index: int
    source_task_id: str
    source_budget_fraction: float
    source_budget_phase: int
    task_changed: bool
    budget_phase_changed: bool
    kl_divergence: float
    total_variation_distance: float
    top_action_changed: bool


PolicyProbabilityFunction = Callable[[Mapping[str, np.ndarray]], np.ndarray]


def _state_probability(
    sample: PolicyStateSample,
    probability_function: PolicyProbabilityFunction,
) -> np.ndarray:
    observation = {name: np.asarray(value).copy() for name, value in sample.observation.items()}
    probability = np.asarray(probability_function(observation), dtype=float)
    if probability.ndim != 1 or len(probability) < _MIN_CATEGORICAL_ACTIONS:
        raise ValueError("probability_function must return one categorical vector with at least two actions.")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0.0):
        raise ValueError("Actor probabilities must be finite and non-negative.")
    total = float(np.sum(probability))
    if not np.isclose(total, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError(f"Actor probabilities must sum to one, got {total}.")
    return probability / total


def _require_compatible_observation_shapes(
    reference: PolicyStateSample,
    candidate: PolicyStateSample,
) -> None:
    for name in _POLICY_OBSERVATION_KEYS:
        reference_shape = np.asarray(reference.observation[name]).shape
        candidate_shape = np.asarray(candidate.observation[name]).shape
        if candidate_shape != reference_shape:
            raise DistinctStateSubstitutionError(
                f"Substitution component {name!r} has shape {candidate_shape}, expected {reference_shape}."
            )


def policy_state_substitution_sensitivity(
    samples: Sequence[PolicyStateSample],
    probability_function: PolicyProbabilityFunction,
    *,
    reference_index: int = 0,
) -> Mapping[str, StateSubstitutionSensitivity]:
    """Measure substitutions selected from provably distinct task/budget strata.

    The another-task source must carry a different task ID and preferentially
    uses the same budget phase. The another-budget source must carry a
    different quarter-budget phase and preferentially uses the same task. The
    function fails instead of silently using a circular worker shift that does
    not establish either distinction.
    """
    if not samples:
        raise DistinctStateSubstitutionError("At least one policy-state sample is required.")
    if isinstance(reference_index, bool) or not isinstance(reference_index, int):
        raise ValueError("reference_index must be an integer.")
    if not 0 <= reference_index < len(samples):
        raise IndexError(f"reference_index {reference_index} is outside {len(samples)} samples.")
    reference = samples[reference_index]
    indexed_candidates = [(index, sample) for index, sample in enumerate(samples) if index != reference_index]

    task_candidates = [item for item in indexed_candidates if item[1].task_id != reference.task_id]
    if not task_candidates:
        raise DistinctStateSubstitutionError(
            f"No sample has a task ID distinct from reference task {reference.task_id!r}."
        )
    task_index, task_sample = sorted(
        task_candidates,
        key=lambda item: (item[1].budget_phase != reference.budget_phase, item[0]),
    )[0]

    budget_candidates = [item for item in indexed_candidates if item[1].budget_phase != reference.budget_phase]
    if not budget_candidates:
        raise DistinctStateSubstitutionError(
            f"No sample has a budget phase distinct from reference phase {reference.budget_phase}."
        )
    budget_index, budget_sample = sorted(
        budget_candidates,
        key=lambda item: (item[1].task_id != reference.task_id, item[0]),
    )[0]
    _require_compatible_observation_shapes(reference, task_sample)
    _require_compatible_observation_shapes(reference, budget_sample)

    reference_probability = _state_probability(reference, probability_function)
    substitutions = (
        ("state_from_another_task", task_index, task_sample),
        ("state_from_another_budget_phase", budget_index, budget_sample),
    )
    results: dict[str, StateSubstitutionSensitivity] = {}
    safe_reference = np.clip(reference_probability, _STATE_SENSITIVITY_EPSILON, 1.0)
    for intervention, source_index, source in substitutions:
        source_probability = _state_probability(source, probability_function)
        if source_probability.shape != reference_probability.shape:
            raise ValueError("All substituted actor distributions must have the same action count.")
        safe_source = np.clip(source_probability, _STATE_SENSITIVITY_EPSILON, 1.0)
        kl_divergence = float(np.sum(safe_reference * np.log(safe_reference / safe_source)))
        total_variation = float(0.5 * np.sum(np.abs(reference_probability - source_probability)))
        results[intervention] = StateSubstitutionSensitivity(
            intervention=intervention,
            source_index=source_index,
            source_task_id=source.task_id,
            source_budget_fraction=float(source.budget_fraction),
            source_budget_phase=source.budget_phase,
            task_changed=source.task_id != reference.task_id,
            budget_phase_changed=source.budget_phase != reference.budget_phase,
            kl_divergence=kl_divergence,
            total_variation_distance=total_variation,
            top_action_changed=int(np.argmax(reference_probability)) != int(np.argmax(source_probability)),
        )
    return results


def write_evaluation_records_csv(records: Sequence[EvaluationRecord], path: Path) -> None:
    """Persist validated tidy rows with a stable, explicit schema."""
    validate_paired_contexts(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(TIDY_RECORD_FIELDS))
        writer.writeheader()
        writer.writerows(record.to_tidy_row() for record in records)
