"""Contracts for exact/best-known references and breach persistence."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dacboenv.reference import (
    BBOBExactReferenceProvider,
    CompositeReferenceProvider,
    ExactReferenceBreachError,
    JSONLReferenceBreachRecorder,
    ManifestReferenceProvider,
    ObjectiveReference,
    ObjectiveReferenceError,
    ReferenceBreachContext,
    ReferenceCompatibilityError,
    ReferenceLookupError,
    ReferenceManifestError,
    ReferenceProvenanceError,
    reference_regret,
)

SOURCE_HASH = "a" * 64
DATA_HASH = "b" * 64


def _yahpo_row(task_id: str = "yahpo/so/lcbench/3945/None", **updates: Any) -> dict[str, Any]:
    scenario, instance = task_id.split("/")[2:4]
    row: dict[str, Any] = {
        "task_id": task_id,
        "value": -91.25,
        "kind": "best_known",
        "runtime_objective_transform": "negative_accuracy",
        "reporting_objective_transform": "one_minus_accuracy",
        "fidelity": "fixed_maximum",
        "source": "reproducible-reference-campaign-v1",
        "source_hash": SOURCE_HASH,
        "benchmark_code_version": "yahpo_gym=1.0.2",
        "benchmark_data_version": "yahpo_data=1.0.2",
        "tolerance": 1e-8,
        "metadata": {
            "scenario": scenario,
            "instance": instance,
            "objective_target": "val_accuracy",
            "reporting_value": 0.0875,
            "source_method": "seeded Sobol plus SMAC portfolio",
            "source_seeds": [11, 22],
            "source_evaluation_budget": 1000,
            "generation_date": "2026-08-02",
            "source_code_commit": "c" * 40,
            "benchmark_data_hash": DATA_HASH,
            "runtime_units": "negative_accuracy",
            "reporting_units": "one_minus_accuracy",
            "maximum_fidelity": True,
            "provenance_status": "complete",
            "nested": {"portfolio": ["sobol", "smac"]},
        },
    }
    row.update(updates)
    return row


def _exact_reference(*, tolerance: float = 1e-8) -> ObjectiveReference:
    return ObjectiveReference(
        task_id="bbob/2/3/0",
        value=2.0,
        kind="exact",
        runtime_objective_transform="identity",
        reporting_objective_transform="identity",
        fidelity="not_applicable",
        source="live objective f_min",
        source_hash=SOURCE_HASH,
        benchmark_code_version="carps=test;ioh=test",
        benchmark_data_version="ioh=test;generated-live",
        tolerance=tolerance,
        metadata={"nested": {"values": [1, 2]}},
    )


def _context() -> ReferenceBreachContext:
    return ReferenceBreachContext(
        run_id="smoke-run",
        trial=7,
        outer_seed=3,
        inner_seed=19,
        scenario="bbob",
        instance="2/3/0",
    )


def test_objective_reference_is_recursively_immutable_and_detached_from_input() -> None:
    metadata = {"nested": {"values": [1, 2]}}
    reference = _exact_reference()
    detached = ObjectiveReference(
        task_id=reference.task_id,
        value=reference.value,
        kind=reference.kind,
        runtime_objective_transform=reference.runtime_objective_transform,
        reporting_objective_transform=reference.reporting_objective_transform,
        fidelity={"mode": "none", "levels": [1, 2]},
        source=reference.source,
        source_hash=reference.source_hash,
        benchmark_code_version=reference.benchmark_code_version,
        benchmark_data_version=reference.benchmark_data_version,
        tolerance=reference.tolerance,
        metadata=metadata,
    )
    metadata["nested"]["values"].append(3)

    assert detached.metadata["nested"]["values"] == (1, 2)
    assert detached.fidelity["levels"] == (1, 2)
    with pytest.raises(TypeError):
        detached.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        detached.metadata["nested"]["new"] = "value"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        detached.value = 0.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"value": math.nan}, "value must be finite"),
        ({"kind": "claimed_optimum"}, "kind must be"),
        ({"fidelity": None}, "fidelity must be explicit"),
        ({"source_hash": "not-a-hash"}, "source_hash"),
        ({"benchmark_data_version": ""}, "benchmark_data_version"),
        ({"tolerance": -1.0}, "non-negative"),
    ],
)
def test_objective_reference_strict_validation(updates: dict[str, Any], message: str) -> None:
    values = {
        "task_id": "bbob/2/3/0",
        "value": 1.0,
        "kind": "exact",
        "runtime_objective_transform": "identity",
        "reporting_objective_transform": "identity",
        "fidelity": "not_applicable",
        "source": "live",
        "source_hash": SOURCE_HASH,
        "benchmark_code_version": "test",
        "benchmark_data_version": "test",
        "tolerance": 0.0,
    }
    values.update(updates)
    with pytest.raises(ObjectiveReferenceError, match=message):
        ObjectiveReference(**values)


def test_bbob_provider_reads_live_f_min_and_checks_runtime_context() -> None:
    provider = BBOBExactReferenceProvider(
        source_hash=SOURCE_HASH,
        benchmark_code_version="carps=test;ioh=test",
        benchmark_data_version="ioh=test;generated-live",
    )
    objective = SimpleNamespace(f_min=-12.5)

    reference = provider.get_reference(
        "bbob/2/3/0",
        objective,
        {
            "task_id": "bbob/2/3/0",
            "runtime_objective_transform": "identity",
            "reporting_objective_transform": "identity",
            "fidelity": "not_applicable",
        },
    )

    assert reference.value == -12.5
    assert reference.kind == "exact"
    assert reference.metadata["objective_class"].endswith("SimpleNamespace")
    with pytest.raises(ReferenceLookupError, match="does not cover"):
        provider.get_reference("yahpo/so/lcbench/3945/None", objective, None)
    with pytest.raises(ReferenceLookupError, match="finite live f_min"):
        provider.get_reference("bbob/2/3/0", SimpleNamespace(f_min=math.inf), None)
    with pytest.raises(ReferenceCompatibilityError, match="fidelity mismatch"):
        provider.get_reference("bbob/2/3/0", objective, {"fidelity": "fixed_maximum"})


def test_manifest_provider_lookup_missing_duplicate_and_immutable_index() -> None:
    row = _yahpo_row()
    provider = ManifestReferenceProvider([row])
    reference = provider.get_reference(
        row["task_id"],
        object(),
        {
            "runtime_objective_transform": "negative_accuracy",
            "reporting_objective_transform": "one_minus_accuracy",
            "fidelity": "fixed_maximum",
        },
    )

    assert reference.value == -91.25
    assert reference.kind == "best_known"
    assert reference.metadata["source_seeds"] == (11, 22)
    with pytest.raises(TypeError):
        provider.references["new"] = reference  # type: ignore[index]
    with pytest.raises(ReferenceLookupError, match="No objective reference"):
        provider.get_reference("yahpo/so/lcbench/7593/None", object(), None)
    with pytest.raises(ReferenceCompatibilityError, match="objective_target mismatch"):
        provider.get_reference(row["task_id"], object(), {"objective_target": "acc"})
    with pytest.raises(ReferenceManifestError, match="Duplicate"):
        ManifestReferenceProvider([row, row])


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("runtime_objective_transform", "one_minus_accuracy", "runtime_objective_transform mismatch"),
        ("reporting_objective_transform", "identity", "reporting_objective_transform mismatch"),
        ("fidelity", "minimum", "fidelity mismatch"),
    ],
)
def test_manifest_provider_rejects_runtime_compatibility_mismatch(
    field: str,
    replacement: Any,
    message: str,
) -> None:
    provider = ManifestReferenceProvider([_yahpo_row()])
    metadata = {
        "runtime_objective_transform": "negative_accuracy",
        "reporting_objective_transform": "one_minus_accuracy",
        "fidelity": "fixed_maximum",
    }
    metadata[field] = replacement

    with pytest.raises(ReferenceCompatibilityError, match=message):
        provider.get_reference("yahpo/so/lcbench/3945/None", object(), metadata)


def test_manifest_provider_rejects_incomplete_and_invalid_best_known_provenance() -> None:
    missing = _yahpo_row()
    del missing["metadata"]["source_evaluation_budget"]
    with pytest.raises(ReferenceProvenanceError, match="source_evaluation_budget"):
        ManifestReferenceProvider([missing])

    invalid = _yahpo_row()
    invalid["metadata"]["scenario"] = "rbv2_super"
    with pytest.raises(ReferenceProvenanceError, match="does not match task ID"):
        ManifestReferenceProvider([invalid])

    invalid_date = _yahpo_row()
    invalid_date["metadata"]["generation_date"] = "sometime"
    with pytest.raises(ReferenceProvenanceError, match="ISO-8601"):
        ManifestReferenceProvider([invalid_date])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row["metadata"].pop("maximum_fidelity"), "maximum_fidelity"),
        (lambda row: row["metadata"].pop("runtime_units"), "runtime_units"),
        (lambda row: row["metadata"].update(maximum_fidelity=False), "maximum_fidelity=true"),
        (lambda row: row["metadata"].__setitem__("benchmark_data_hash", "not-a-hash"), "SHA-256"),
        (lambda row: row["metadata"].__setitem__("reporting_value", 0.9), "reporting_value"),
        (lambda row: row.__setitem__("benchmark_code_version", "unavailable"), "incomplete"),
        (lambda row: row["metadata"].__setitem__("source_code_commit", "unavailable"), "Git object ID"),
    ],
)
def test_yahpo_provider_rejects_scientifically_incomplete_rows(mutation: Any, message: str) -> None:
    row = _yahpo_row()
    mutation(row)

    with pytest.raises(ReferenceProvenanceError, match=message):
        ManifestReferenceProvider([row])


def test_incomplete_smoke_reference_requires_explicit_nontraining_override() -> None:
    row = _yahpo_row()
    row["metadata"]["provenance_status"] = "smoke_only_incomplete"

    with pytest.raises(ReferenceProvenanceError, match="cannot enable training"):
        ManifestReferenceProvider([row])

    provider = ManifestReferenceProvider([row], allow_incomplete_best_known=True)
    reference = provider.get_reference(row["task_id"], object(), None)
    assert reference.metadata["provenance_status"] == "smoke_only_incomplete"


def test_manifest_provider_accepts_json_csv_yaml_and_task_keyed_rows(tmp_path: Path) -> None:
    row = _yahpo_row()
    json_path = tmp_path / "references.json"
    json_path.write_text(json.dumps({"references": [row]}), encoding="utf-8")

    yaml_path = tmp_path / "references.yaml"
    yaml_path.write_text(json.dumps({"references": [row]}), encoding="utf-8")

    csv_path = tmp_path / "references.csv"
    scalar_row = {key: value for key, value in row.items() if key != "metadata"}
    scalar_row["metadata"] = json.dumps(row["metadata"])
    header = list(scalar_row)

    task_keyed = {row["task_id"]: {key: value for key, value in row.items() if key != "task_id"}}
    for source in (json_path, yaml_path, task_keyed):
        provider = ManifestReferenceProvider(source)
        assert provider.get_reference(row["task_id"], object(), None).value == -91.25

    # The stdlib CSV writer is exercised here to ensure embedded JSON commas
    # round-trip correctly through the provider's DictReader path.
    with csv_path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=header)
        writer.writeheader()
        writer.writerow(scalar_row)
    assert ManifestReferenceProvider(csv_path).get_reference(row["task_id"], object(), None).value == -91.25


def test_composite_provider_dispatches_by_namespace() -> None:
    bbob = BBOBExactReferenceProvider(
        source_hash=SOURCE_HASH,
        benchmark_code_version="test",
        benchmark_data_version="test",
    )
    yahpo = ManifestReferenceProvider([_yahpo_row()])
    composite = CompositeReferenceProvider({"BBOB": bbob, "yahpo": yahpo})

    assert composite.get_reference("bbob/2/3/0", SimpleNamespace(f_min=1.5), None).value == 1.5
    assert composite.get_reference("yahpo/so/lcbench/3945/None", object(), None).value == -91.25
    with pytest.raises(ReferenceLookupError, match="namespace"):
        composite.get_reference("unknown/task", object(), None)


def test_best_known_breach_is_appended_and_regret_is_clipped(tmp_path: Path) -> None:
    reference = ManifestReferenceProvider([_yahpo_row()]).get_reference(
        "yahpo/so/lcbench/3945/None",
        object(),
        None,
    )
    path = tmp_path / "breaches.jsonl"
    path.write_text('{"preexisting":true}\n', encoding="utf-8")
    recorder = JSONLReferenceBreachRecorder(path)
    context = ReferenceBreachContext(
        run_id="yahpo-smoke",
        trial=12,
        outer_seed=5,
        inner_seed=7,
        scenario="lcbench",
        instance="3945",
    )

    assert reference_regret(reference, -92.0, recorder=recorder, context=context) == 0.0
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert records[0] == {"preexisting": True}
    breach = records[1]
    assert breach["timestamp"].endswith("Z")
    assert breach["run_id"] == "yahpo-smoke"
    assert breach["task_id"] == reference.task_id
    assert breach["reference_kind"] == "best_known"
    assert breach["reference_value"] == -91.25
    assert breach["observed_value"] == -92.0
    assert breach["breach_magnitude"] == 0.75
    assert breach["trial"] == 12
    assert breach["outer_seed"] == 5
    assert breach["inner_seed"] == 7
    assert breach["scenario"] == "lcbench"
    assert breach["instance"] == "3945"
    assert breach["source"] == reference.source
    assert breach["source_hash"] == SOURCE_HASH
    assert breach["benchmark_code_version"] == "yahpo_gym=1.0.2"
    assert breach["benchmark_data_version"] == "yahpo_data=1.0.2"
    assert breach["hard_error"] is False
    assert breach["review_required"] is True


def test_exact_breach_is_persisted_before_hard_error(tmp_path: Path) -> None:
    path = tmp_path / "exact-breaches.jsonl"
    recorder = JSONLReferenceBreachRecorder(path)
    reference = _exact_reference()

    with pytest.raises(ExactReferenceBreachError, match="Exact reference breached") as error_info:
        reference_regret(reference, 1.0, recorder=recorder, context=_context())

    assert error_info.value.record.hard_error is True
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["reference_kind"] == "exact"
    assert record["hard_error"] is True


def test_values_within_tolerance_and_failed_trials_do_not_create_breaches(tmp_path: Path) -> None:
    path = tmp_path / "no-breach.jsonl"
    recorder = JSONLReferenceBreachRecorder(path)
    reference = _exact_reference(tolerance=1e-3)

    assert reference_regret(reference, 2.0 - 5e-4, recorder=recorder, context=_context()) == 0.0
    assert math.isinf(reference_regret(reference, math.nan, recorder=recorder, context=_context()))
    assert math.isinf(reference_regret(reference, math.inf, recorder=recorder, context=_context()))
    assert not path.exists()
