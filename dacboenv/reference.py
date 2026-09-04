"""Objective-reference providers and persistent breach handling.

References are privileged scientific metadata.  This module intentionally has
no dependency on observation construction, and none of its objects should be
placed in policy-visible reset information.
"""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from omegaconf import DictConfig, ListConfig, OmegaConf

ReferenceKind = Literal["exact", "best_known"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_BBOB_TASK_ID = re.compile(r"^bbob/\d+/\d+/\d+$", flags=re.IGNORECASE)
_OPTBENCH_TASK_ID = re.compile(r"^optbench/[^/]+$", flags=re.IGNORECASE)
_INCOMPLETE_PROVENANCE_VALUES = frozenset({"n/a", "none", "unknown", "unavailable"})
_REFERENCE_FIELDS = frozenset(
    {
        "task_id",
        "value",
        "kind",
        "runtime_objective_transform",
        "reporting_objective_transform",
        "fidelity",
        "source",
        "source_hash",
        "benchmark_code_version",
        "benchmark_data_version",
        "tolerance",
        "metadata",
    }
)

DEFAULT_BEST_KNOWN_PROVENANCE_KEYS = frozenset(
    {
        "source_method",
        "source_seeds",
        "source_evaluation_budget",
        "generation_date",
        "source_code_commit",
    }
)
"""Required provenance for every empirical or explicitly assumed best-known value."""

ASSUMED_BOUND_SOURCE_METHOD = "assumed_accuracy_upper_bound_v1"
"""Explicit non-empirical YAHPO reference convention requested by the protocol owner."""

YAHPO_BEST_KNOWN_PROVENANCE_KEYS = DEFAULT_BEST_KNOWN_PROVENANCE_KEYS | frozenset(
    {
        "scenario",
        "instance",
        "objective_target",
        "reporting_value",
        "benchmark_data_hash",
        "runtime_units",
        "reporting_units",
        "maximum_fidelity",
        "provenance_status",
    }
)
"""Additional fields required for a provenance-complete YAHPO reference."""


class ObjectiveReferenceError(ValueError):
    """Base class for invalid objective-reference data."""


class ReferenceManifestError(ObjectiveReferenceError):
    """Raised when a reference manifest is malformed or contains duplicates."""


class ReferenceProvenanceError(ReferenceManifestError):
    """Raised when required reference provenance is absent or invalid."""


class ReferenceCompatibilityError(ObjectiveReferenceError):
    """Raised when a reference and runtime objective context are incompatible."""


class ReferenceLookupError(KeyError):
    """Raised when no reference provider or manifest row covers a task."""


class ExactReferenceBreachError(RuntimeError):
    """A successful observation contradicted a reference declared exact."""

    def __init__(self, record: ReferenceBreachRecord) -> None:
        self.record = record
        super().__init__(
            f"Exact reference breached for {record.task_id!r}: observed "
            f"{record.observed_value:.17g} < {record.reference_value:.17g} "
            f"by {record.breach_magnitude:.17g}."
        )


def _require_nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ObjectiveReferenceError(f"{field_name} must be a non-empty string, got {value!r}.")
    return value.strip()


def _finite_float(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ObjectiveReferenceError(f"{field_name} must be a finite real number, got {value!r}.")
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise ObjectiveReferenceError(f"{field_name} must be a finite real number, got {value!r}.") from error
    if not math.isfinite(converted):
        raise ObjectiveReferenceError(f"{field_name} must be finite, got {value!r}.")
    return converted


def _deep_freeze(value: Any, *, field_name: str) -> Any:
    """Copy JSON-like data into recursively immutable containers."""
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObjectiveReferenceError(f"{field_name} contains a non-finite value: {value!r}.")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ObjectiveReferenceError(f"{field_name} mapping keys must be strings, got {key!r}.")
            frozen[key] = _deep_freeze(item, field_name=f"{field_name}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, tuple | list):
        return tuple(_deep_freeze(item, field_name=f"{field_name}[]") for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_deep_freeze(item, field_name=f"{field_name}[]") for item in value)
    raise ObjectiveReferenceError(f"{field_name} must contain only JSON-like values, got {type(value).__name__}.")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple | list | set | frozenset):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ObjectiveReference:
    """Immutable exact or empirical best-known objective reference.

    ``value`` is always expressed in the units returned by the runtime
    objective.  Reporting-unit values belong in ``metadata`` (normally under
    ``reporting_value``); the two explicit transform fields prevent silently
    comparing values on different scales.
    """

    task_id: str
    value: float
    kind: ReferenceKind
    runtime_objective_transform: str
    reporting_objective_transform: str
    fidelity: Any
    source: str
    source_hash: str
    benchmark_code_version: str
    benchmark_data_version: str
    tolerance: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_id", _require_nonempty_string(self.task_id, field_name="task_id"))
        object.__setattr__(self, "value", _finite_float(self.value, field_name="value"))
        if self.kind not in {"exact", "best_known"}:
            raise ObjectiveReferenceError(f"kind must be 'exact' or 'best_known', got {self.kind!r}.")
        object.__setattr__(
            self,
            "runtime_objective_transform",
            _require_nonempty_string(
                self.runtime_objective_transform,
                field_name="runtime_objective_transform",
            ),
        )
        object.__setattr__(
            self,
            "reporting_objective_transform",
            _require_nonempty_string(
                self.reporting_objective_transform,
                field_name="reporting_objective_transform",
            ),
        )
        if self.fidelity is None or (isinstance(self.fidelity, str) and not self.fidelity.strip()):
            raise ObjectiveReferenceError("fidelity must be explicit; use 'not_applicable' when appropriate.")
        object.__setattr__(self, "fidelity", _deep_freeze(self.fidelity, field_name="fidelity"))
        object.__setattr__(self, "source", _require_nonempty_string(self.source, field_name="source"))
        source_hash = _require_nonempty_string(self.source_hash, field_name="source_hash")
        if _SHA256.fullmatch(source_hash) is None:
            raise ObjectiveReferenceError("source_hash must be a lowercase SHA-256 digest.")
        object.__setattr__(self, "source_hash", source_hash)
        object.__setattr__(
            self,
            "benchmark_code_version",
            _require_nonempty_string(self.benchmark_code_version, field_name="benchmark_code_version"),
        )
        object.__setattr__(
            self,
            "benchmark_data_version",
            _require_nonempty_string(self.benchmark_data_version, field_name="benchmark_data_version"),
        )
        tolerance = _finite_float(self.tolerance, field_name="tolerance")
        if tolerance < 0.0:
            raise ObjectiveReferenceError(f"tolerance must be non-negative, got {tolerance!r}.")
        object.__setattr__(self, "tolerance", tolerance)
        if not isinstance(self.metadata, Mapping):
            raise ObjectiveReferenceError(f"metadata must be a mapping, got {type(self.metadata).__name__}.")
        object.__setattr__(self, "metadata", _deep_freeze(dict(self.metadata), field_name="metadata"))


@runtime_checkable
class ReferenceProvider(Protocol):
    """Protocol implemented by task-aware objective-reference providers."""

    def get_reference(
        self,
        task_id: str,
        objective_function: Any,
        task_metadata: Mapping[str, Any] | None,
    ) -> ObjectiveReference:
        """Return a validated reference for the active runtime objective."""


ObjectiveReferenceProvider = ReferenceProvider
"""Descriptive alias retained for callers that prefer the longer name."""


def _task_expectation(task_metadata: Mapping[str, Any] | None, key: str) -> Any:
    if task_metadata is None:
        return None
    return task_metadata.get(key)


def _validate_compatibility(
    reference: ObjectiveReference,
    task_metadata: Mapping[str, Any] | None,
) -> None:
    if task_metadata is None:
        return
    expected_task_id = task_metadata.get("task_id")
    if expected_task_id is not None and str(expected_task_id) != reference.task_id:
        raise ReferenceCompatibilityError(
            f"Reference task {reference.task_id!r} does not match runtime task {expected_task_id!r}."
        )
    checks = (
        ("runtime_objective_transform", reference.runtime_objective_transform),
        ("reporting_objective_transform", reference.reporting_objective_transform),
        ("fidelity", reference.fidelity),
    )
    for key, actual in checks:
        expected = _task_expectation(task_metadata, key)
        if expected is None:
            continue
        frozen_expected = _deep_freeze(expected, field_name=f"task_metadata.{key}")
        if frozen_expected != actual:
            raise ReferenceCompatibilityError(
                f"Reference {key} mismatch for {reference.task_id!r}: "
                f"manifest={_jsonable(actual)!r}, runtime={_jsonable(frozen_expected)!r}."
            )
    for key in ("scenario", "instance", "objective_target"):
        expected = task_metadata.get(key)
        if expected is None:
            continue
        actual = reference.metadata.get(key)
        if actual is None or str(actual) != str(expected):
            raise ReferenceCompatibilityError(
                f"Reference {key} mismatch for {reference.task_id!r}: manifest={actual!r}, runtime={expected!r}."
            )


def _installed_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "unavailable"


def _objective_source_hash(objective_function: Any) -> str:
    objective_type = type(objective_function)
    try:
        source = inspect.getsource(objective_type)
    except (OSError, TypeError):
        source = f"{objective_type.__module__}.{objective_type.__qualname__}"
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class BBOBExactReferenceProvider:
    """Read the exact optimum from the active CARP-S/IOH BBOB objective."""

    def __init__(
        self,
        *,
        runtime_objective_transform: str = "identity",
        reporting_objective_transform: str = "identity",
        fidelity: Any = "not_applicable",
        source: str = "live CARP-S BBOBObjectiveFunction.f_min backed by IOH",
        source_hash: str | None = None,
        benchmark_code_version: str | None = None,
        benchmark_data_version: str | None = None,
        tolerance: float = 1e-8,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._runtime_objective_transform = runtime_objective_transform
        self._reporting_objective_transform = reporting_objective_transform
        self._fidelity = _deep_freeze(fidelity, field_name="fidelity")
        self._source = source
        self._source_hash = source_hash
        self._benchmark_code_version = benchmark_code_version
        self._benchmark_data_version = benchmark_data_version
        self._tolerance = tolerance
        self._metadata = _deep_freeze(dict(metadata or {}), field_name="metadata")

    def get_reference(
        self,
        task_id: str,
        objective_function: Any,
        task_metadata: Mapping[str, Any] | None,
    ) -> ObjectiveReference:
        """Read and validate ``f_min`` from the live objective instance."""
        if _BBOB_TASK_ID.fullmatch(task_id) is None:
            raise ReferenceLookupError(f"BBOB exact-reference provider does not cover task {task_id!r}.")
        try:
            minimum = objective_function.f_min
        except AttributeError as error:
            raise ReferenceLookupError(
                f"Active objective {type(objective_function).__name__} exposes no live BBOB f_min."
            ) from error
        try:
            minimum = _finite_float(minimum, field_name=f"{task_id}.f_min")
        except ObjectiveReferenceError as error:
            raise ReferenceLookupError(f"BBOB task {task_id!r} exposes no finite live f_min: {minimum!r}.") from error

        code_version = self._benchmark_code_version or (
            f"carps={_installed_version('carps')};ioh={_installed_version('ioh')}"
        )
        data_version = self._benchmark_data_version or f"ioh={_installed_version('ioh')};generated-live"
        objective_type = type(objective_function)
        metadata = {
            **dict(self._metadata),
            "objective_class": f"{objective_type.__module__}.{objective_type.__qualname__}",
        }
        reference = ObjectiveReference(
            task_id=task_id,
            value=minimum,
            kind="exact",
            runtime_objective_transform=self._runtime_objective_transform,
            reporting_objective_transform=self._reporting_objective_transform,
            fidelity=self._fidelity,
            source=self._source,
            source_hash=self._source_hash or _objective_source_hash(objective_function),
            benchmark_code_version=code_version,
            benchmark_data_version=data_version,
            tolerance=self._tolerance,
            metadata=metadata,
        )
        _validate_compatibility(reference, task_metadata)
        return reference


class OptBenchExactReferenceProvider:
    """Read an exact global minimum from the active OptBench objective.

    OptBench exposes ``f_min`` on the instantiated objective, currently as a
    property. Callable forms are accepted as well because its wrapper class
    exposes the same value as a method. Tasks without a finite minimum fail
    closed and must not enter a reference-regret training manifest.
    """

    def __init__(
        self,
        *,
        tolerance: float = 1e-8,
        source_hash: str | None = None,
    ) -> None:
        self._tolerance = tolerance
        self._source_hash = source_hash

    def get_reference(
        self,
        task_id: str,
        objective_function: Any,
        task_metadata: Mapping[str, Any] | None,
    ) -> ObjectiveReference:
        """Return the finite live OptBench global minimum as reward-only data."""
        if _OPTBENCH_TASK_ID.fullmatch(task_id) is None:
            raise ReferenceLookupError(f"OptBench exact-reference provider does not cover task {task_id!r}.")
        try:
            minimum = objective_function.f_min
        except AttributeError as error:
            raise ReferenceLookupError(
                f"Active objective {type(objective_function).__name__} exposes no OptBench f_min."
            ) from error
        if callable(minimum):
            minimum = minimum()
        try:
            value = _finite_float(minimum, field_name=f"{task_id}.f_min")
        except ObjectiveReferenceError as error:
            raise ReferenceLookupError(
                f"OptBench task {task_id!r} exposes no finite global minimum: {minimum!r}."
            ) from error
        objective_type = type(objective_function)
        reference = ObjectiveReference(
            task_id=task_id,
            value=value,
            kind="exact",
            runtime_objective_transform="identity",
            reporting_objective_transform="identity",
            fidelity="not_applicable",
            source="live installed OptBench objective f_min",
            source_hash=self._source_hash or _objective_source_hash(objective_function),
            benchmark_code_version=f"OptBench={_installed_version('OptBench')};carps={_installed_version('carps')}",
            benchmark_data_version="analytic-function-no-external-data",
            tolerance=self._tolerance,
            metadata={
                "objective_class": f"{objective_type.__module__}.{objective_type.__qualname__}",
            },
        )
        _validate_compatibility(reference, task_metadata)
        return reference


def _parse_serialized_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return ""
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _plain_container(value: Any) -> Any:
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _rows_from_container(container: Any) -> list[Mapping[str, Any]]:
    container = _plain_container(container)
    if isinstance(container, Mapping):
        if "references" in container:
            return _rows_from_container(container["references"])
        if "task_id" in container:
            return [container]
        rows: list[Mapping[str, Any]] = []
        for task_id, value in container.items():
            if not isinstance(value, Mapping):
                raise ReferenceManifestError("A task-keyed reference manifest must map every task ID to a row mapping.")
            row = dict(value)
            if "task_id" in row and row["task_id"] != task_id:
                raise ReferenceManifestError(
                    f"Task-keyed row {task_id!r} contains conflicting task_id {row['task_id']!r}."
                )
            row["task_id"] = task_id
            rows.append(row)
        return rows
    if isinstance(container, Sequence) and not isinstance(container, str | bytes):
        rows = []
        for index, item in enumerate(container):
            plain_item = _plain_container(item)
            if not isinstance(plain_item, Mapping):
                raise ReferenceManifestError(
                    f"Reference row {index} must be a mapping, got {type(plain_item).__name__}."
                )
            rows.append(plain_item)
        return rows
    raise ReferenceManifestError(
        "Reference data must be a row mapping, a sequence of rows, a task-keyed mapping, or a references container."
    )


def _load_reference_rows(
    source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    if not isinstance(source, str | Path):
        return _rows_from_container(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Objective-reference manifest does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as file_handle:
            return [dict(row) for row in csv.DictReader(file_handle)]
    if suffix == ".json":
        return _rows_from_container(json.loads(path.read_text(encoding="utf-8")))
    if suffix in {".yaml", ".yml"}:
        return _rows_from_container(OmegaConf.load(path))
    raise ReferenceManifestError(f"Unsupported reference-manifest suffix {suffix!r} for {path}.")


def _normalize_reference_row(row: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {str(key): _parse_serialized_value(value) for key, value in row.items()}
    aliases = {
        "reference_runtime_value": "value",
        "reference_in_runtime_units": "value",
        "reference_reporting_value": "reporting_value",
        "reference_in_reporting_units": "reporting_value",
        "objective": "objective_target",
        "target": "objective_target",
    }
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized.pop(alias)

    metadata = normalized.pop("metadata", {})
    metadata = _parse_serialized_value(metadata)
    if metadata is None or metadata == "":
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise ReferenceManifestError("Reference-row metadata must be a mapping or a JSON-encoded mapping.")
    merged_metadata = dict(metadata)
    for key in list(normalized):
        if key not in _REFERENCE_FIELDS:
            merged_metadata[key] = normalized.pop(key)
    normalized["metadata"] = merged_metadata
    return normalized


def _validate_generation_date(value: Any, *, task_id: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ReferenceProvenanceError(f"Reference {task_id!r} has no valid generation_date.")
    candidate = value.strip().replace("Z", "+00:00")
    try:
        if "T" in candidate or " " in candidate:
            datetime.fromisoformat(candidate)
        else:
            date.fromisoformat(candidate)
    except ValueError as error:
        raise ReferenceProvenanceError(
            f"Reference {task_id!r} generation_date must be ISO-8601, got {value!r}."
        ) from error


def _validate_provenance_strings(reference: ObjectiveReference, required_keys: frozenset[str]) -> None:
    for key in ("source_method", "source_code_commit"):
        if key in required_keys:
            try:
                _require_nonempty_string(reference.metadata[key], field_name=f"metadata.{key}")
            except ObjectiveReferenceError as error:
                raise ReferenceProvenanceError(str(error)) from error
    source_commit = reference.metadata.get("source_code_commit")
    if "source_code_commit" in required_keys and _GIT_COMMIT.fullmatch(str(source_commit)) is None:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} source_code_commit must be a full Git object ID."
        )
    for field_name, value in (
        ("benchmark_code_version", reference.benchmark_code_version),
        ("benchmark_data_version", reference.benchmark_data_version),
    ):
        if value.strip().lower() in _INCOMPLETE_PROVENANCE_VALUES:
            raise ReferenceProvenanceError(
                f"Best-known reference {reference.task_id!r} has incomplete {field_name}: {value!r}."
            )


def _validate_provenance_seeds_and_budget(reference: ObjectiveReference, required_keys: frozenset[str]) -> None:
    assumption_based = reference.metadata.get("source_method") == ASSUMED_BOUND_SOURCE_METHOD
    if "source_seeds" in required_keys:
        seeds = reference.metadata["source_seeds"]
        if not isinstance(seeds, tuple) or (not seeds and not assumption_based):
            raise ReferenceProvenanceError(
                f"Best-known reference {reference.task_id!r} source_seeds must be a sequence and may be empty "
                f"only for {ASSUMED_BOUND_SOURCE_METHOD!r}."
            )
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ReferenceProvenanceError(
                f"Best-known reference {reference.task_id!r} source_seeds must contain only integers."
            )
    if "source_evaluation_budget" in required_keys:
        budget = reference.metadata["source_evaluation_budget"]
        valid_budget = (
            isinstance(budget, int)
            and not isinstance(budget, bool)
            and (budget == 0 if assumption_based else budget > 0)
        )
        if not valid_budget:
            raise ReferenceProvenanceError(
                f"Best-known reference {reference.task_id!r} source_evaluation_budget must be a positive integer, "
                f"or zero only for {ASSUMED_BOUND_SOURCE_METHOD!r}."
            )


def _validate_assumed_bound_provenance(reference: ObjectiveReference) -> None:
    """Require an assumed bound to remain explicit and never masquerade as an empirical optimum."""
    if reference.metadata.get("source_method") != ASSUMED_BOUND_SOURCE_METHOD:
        return
    required = {
        "reference_basis": "assumed_metric_upper_bound",
        "empirical": False,
        "exactness_proved": False,
        "assumption_authority": "user_specified_protocol",
    }
    mismatches = {
        key: {"expected": expected, "actual": reference.metadata.get(key)}
        for key, expected in required.items()
        if reference.metadata.get(key) != expected
    }
    if mismatches:
        raise ReferenceProvenanceError(
            f"Assumed-bound reference {reference.task_id!r} has invalid provenance markers: {mismatches}."
        )


def _validate_yahpo_identity(reference: ObjectiveReference) -> str:
    parts = reference.task_id.split("/")
    if len(parts) < 4:  # noqa: PLR2004
        raise ReferenceProvenanceError(f"Malformed YAHPO task ID {reference.task_id!r}.")
    scenario, instance = parts[2], parts[3]
    if str(reference.metadata["scenario"]) != scenario or str(reference.metadata["instance"]) != instance:
        raise ReferenceProvenanceError(
            f"YAHPO provenance scenario/instance does not match task ID {reference.task_id!r}."
        )
    for key in ("objective_target", "benchmark_data_hash", "runtime_units", "reporting_units"):
        try:
            _require_nonempty_string(reference.metadata[key], field_name=f"metadata.{key}")
        except ObjectiveReferenceError as error:
            raise ReferenceProvenanceError(str(error)) from error
    if _SHA256.fullmatch(str(reference.metadata["benchmark_data_hash"])) is None:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} benchmark_data_hash must be a SHA-256 digest."
        )
    return scenario


def _validate_yahpo_units_and_fidelity(reference: ObjectiveReference) -> None:
    if reference.metadata["runtime_units"] != reference.runtime_objective_transform:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} runtime units do not match its runtime transform."
        )
    if reference.metadata["reporting_units"] != reference.reporting_objective_transform:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} reporting units do not match its reporting transform."
        )
    if reference.metadata["maximum_fidelity"] is not True:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} must explicitly record maximum_fidelity=true."
        )


def _validate_yahpo_provenance_status(
    reference: ObjectiveReference,
    *,
    allow_incomplete_best_known: bool,
) -> None:
    provenance_status = reference.metadata["provenance_status"]
    if provenance_status not in {"complete", "smoke_only_incomplete"}:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} has invalid provenance_status {provenance_status!r}."
        )
    if provenance_status != "complete" and not allow_incomplete_best_known:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} is {provenance_status!r}, not provenance-complete. "
            "Incomplete rows require an explicit engineering-smoke override and cannot enable training."
        )


def _validate_yahpo_reporting_value(reference: ObjectiveReference, scenario: str) -> None:
    try:
        reporting_value = _finite_float(
            reference.metadata["reporting_value"],
            field_name="metadata.reporting_value",
        )
    except ObjectiveReferenceError as error:
        raise ReferenceProvenanceError(str(error)) from error
    if (
        reference.runtime_objective_transform == "negative_accuracy"
        and reference.reporting_objective_transform == "one_minus_accuracy"
    ):
        scale = 100.0 if scenario in {"lcbench", "nb301"} else 1.0
        expected_reporting_value = 1.0 + reference.value / scale
        if not math.isclose(reporting_value, expected_reporting_value, rel_tol=0.0, abs_tol=1e-12):
            raise ReferenceProvenanceError(
                f"Best-known reference {reference.task_id!r} reporting_value is inconsistent with "
                "its runtime value and declared transforms."
            )


def _validate_yahpo_provenance(
    reference: ObjectiveReference,
    *,
    allow_incomplete_best_known: bool,
) -> None:
    scenario = _validate_yahpo_identity(reference)
    _validate_yahpo_units_and_fidelity(reference)
    _validate_yahpo_provenance_status(
        reference,
        allow_incomplete_best_known=allow_incomplete_best_known,
    )
    _validate_yahpo_reporting_value(reference, scenario)


def _validate_best_known_provenance(
    reference: ObjectiveReference,
    required_keys: frozenset[str],
    *,
    allow_incomplete_best_known: bool,
) -> None:
    missing = sorted(key for key in required_keys if key not in reference.metadata)
    if missing:
        raise ReferenceProvenanceError(
            f"Best-known reference {reference.task_id!r} is missing required provenance fields: {missing}."
        )
    _validate_provenance_strings(reference, required_keys)
    _validate_provenance_seeds_and_budget(reference, required_keys)
    _validate_assumed_bound_provenance(reference)
    if "generation_date" in required_keys:
        _validate_generation_date(reference.metadata["generation_date"], task_id=reference.task_id)

    if reference.task_id.lower().startswith("yahpo/"):
        _validate_yahpo_provenance(
            reference,
            allow_incomplete_best_known=allow_incomplete_best_known,
        )


def _validate_provider_expectations(
    reference: ObjectiveReference,
    *,
    runtime_transform: str | None,
    reporting_transform: str | None,
    fidelity: Any | None,
) -> None:
    if runtime_transform is not None and reference.runtime_objective_transform != runtime_transform:
        raise ReferenceCompatibilityError(
            f"Reference runtime_objective_transform mismatch for {reference.task_id!r}: "
            f"{reference.runtime_objective_transform!r} != {runtime_transform!r}."
        )
    if reporting_transform is not None and reference.reporting_objective_transform != reporting_transform:
        raise ReferenceCompatibilityError(
            f"Reference reporting_objective_transform mismatch for {reference.task_id!r}: "
            f"{reference.reporting_objective_transform!r} != {reporting_transform!r}."
        )
    if fidelity is None:
        return
    frozen_expected = _deep_freeze(fidelity, field_name="expected_fidelity")
    if reference.fidelity != frozen_expected:
        raise ReferenceCompatibilityError(
            f"Reference fidelity mismatch for {reference.task_id!r}: "
            f"{_jsonable(reference.fidelity)!r} != {_jsonable(frozen_expected)!r}."
        )


class ManifestReferenceProvider:
    """Load provenance-complete objective references from JSON/CSV/YAML rows."""

    def __init__(
        self,
        source: str | Path | Mapping[str, Any] | Sequence[Mapping[str, Any]],
        *,
        required_best_known_metadata: Sequence[str] = tuple(DEFAULT_BEST_KNOWN_PROVENANCE_KEYS),
        expected_runtime_objective_transform: str | None = None,
        expected_reporting_objective_transform: str | None = None,
        expected_fidelity: Any | None = None,
        allow_incomplete_best_known: bool = False,
    ) -> None:
        required = frozenset(required_best_known_metadata)
        if any(not isinstance(key, str) or not key for key in required):
            raise ReferenceManifestError("Required provenance keys must be non-empty strings.")
        references: dict[str, ObjectiveReference] = {}
        for raw_row in _load_reference_rows(source):
            row = _normalize_reference_row(raw_row)
            try:
                reference = ObjectiveReference(**row)
            except TypeError as error:
                raise ReferenceManifestError(f"Malformed objective-reference row: {error}.") from error
            if reference.task_id in references:
                raise ReferenceManifestError(f"Duplicate objective reference for task {reference.task_id!r}.")
            if reference.kind == "best_known":
                task_required = required
                if reference.task_id.lower().startswith("yahpo/"):
                    task_required |= YAHPO_BEST_KNOWN_PROVENANCE_KEYS
                _validate_best_known_provenance(
                    reference,
                    task_required,
                    allow_incomplete_best_known=allow_incomplete_best_known,
                )
            _validate_provider_expectations(
                reference,
                runtime_transform=expected_runtime_objective_transform,
                reporting_transform=expected_reporting_objective_transform,
                fidelity=expected_fidelity,
            )
            references[reference.task_id] = reference
        if not references:
            raise ReferenceManifestError("An objective-reference manifest must contain at least one row.")
        self._references: Mapping[str, ObjectiveReference] = MappingProxyType(references)

    @property
    def references(self) -> Mapping[str, ObjectiveReference]:
        """Return an immutable task-to-reference view."""
        return self._references

    def get_reference(
        self,
        task_id: str,
        objective_function: Any,  # noqa: ARG002
        task_metadata: Mapping[str, Any] | None,
    ) -> ObjectiveReference:
        """Look up a task and validate it against the active runtime context."""
        try:
            reference = self._references[task_id]
        except KeyError as error:
            raise ReferenceLookupError(f"No objective reference is available for task {task_id!r}.") from error
        _validate_compatibility(reference, task_metadata)
        return reference


class CompositeReferenceProvider:
    """Dispatch references by the first, case-insensitive task namespace."""

    def __init__(self, providers: Mapping[str, ReferenceProvider]) -> None:
        normalized: dict[str, ReferenceProvider] = {}
        for namespace, provider in providers.items():
            key = _require_nonempty_string(namespace, field_name="provider namespace").lower().rstrip("/")
            if "/" in key:
                raise ObjectiveReferenceError(f"Provider namespace must be one task-ID segment, got {namespace!r}.")
            if key in normalized:
                raise ObjectiveReferenceError(f"Duplicate composite provider namespace {key!r}.")
            if not isinstance(provider, ReferenceProvider):
                raise TypeError(f"Provider for namespace {key!r} does not implement get_reference().")
            normalized[key] = provider
        if not normalized:
            raise ObjectiveReferenceError("CompositeReferenceProvider requires at least one provider.")
        self._providers: Mapping[str, ReferenceProvider] = MappingProxyType(normalized)

    def get_reference(
        self,
        task_id: str,
        objective_function: Any,
        task_metadata: Mapping[str, Any] | None,
    ) -> ObjectiveReference:
        """Dispatch a lookup without exposing the resulting value elsewhere."""
        namespace = task_id.partition("/")[0].lower()
        try:
            provider = self._providers[namespace]
        except KeyError as error:
            raise ReferenceLookupError(
                f"No objective-reference provider is registered for namespace {namespace!r}."
            ) from error
        return provider.get_reference(task_id, objective_function, task_metadata)


@dataclass(frozen=True, slots=True)
class ReferenceBreachContext:
    """Run context required for one auditable breach record."""

    run_id: str
    trial: int
    outer_seed: int | None
    inner_seed: int
    scenario: str
    instance: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_nonempty_string(self.run_id, field_name="run_id"))
        if isinstance(self.trial, bool) or not isinstance(self.trial, int) or self.trial < 0:
            raise ObjectiveReferenceError(f"trial must be a non-negative integer, got {self.trial!r}.")
        if self.outer_seed is not None and (isinstance(self.outer_seed, bool) or not isinstance(self.outer_seed, int)):
            raise ObjectiveReferenceError(f"outer_seed must be an integer or null, got {self.outer_seed!r}.")
        if isinstance(self.inner_seed, bool) or not isinstance(self.inner_seed, int):
            raise ObjectiveReferenceError(f"inner_seed must be an integer, got {self.inner_seed!r}.")
        object.__setattr__(self, "scenario", _require_nonempty_string(self.scenario, field_name="scenario"))
        object.__setattr__(self, "instance", _require_nonempty_string(self.instance, field_name="instance"))


@dataclass(frozen=True, slots=True)
class ReferenceBreachRecord:
    """Machine-readable evidence that a successful evaluation beat a reference."""

    timestamp: str
    run_id: str
    task_id: str
    reference_kind: ReferenceKind
    reference_value: float
    observed_value: float
    breach_magnitude: float
    tolerance: float
    trial: int
    outer_seed: int | None
    inner_seed: int
    scenario: str
    instance: str
    source: str
    source_hash: str
    benchmark_code_version: str
    benchmark_data_version: str
    runtime_objective_transform: str
    reporting_objective_transform: str
    fidelity: Any
    hard_error: bool
    review_required: bool = True

    @classmethod
    def create(
        cls,
        reference: ObjectiveReference,
        observed_value: float,
        context: ReferenceBreachContext,
    ) -> ReferenceBreachRecord:
        """Create a UTC-stamped record from an already-detected breach."""
        observed = _finite_float(observed_value, field_name="observed_value")
        magnitude = reference.value - observed
        if magnitude <= reference.tolerance:
            raise ObjectiveReferenceError("Cannot create a breach record for a value within reference tolerance.")
        return cls(
            timestamp=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            run_id=context.run_id,
            task_id=reference.task_id,
            reference_kind=reference.kind,
            reference_value=reference.value,
            observed_value=observed,
            breach_magnitude=magnitude,
            tolerance=reference.tolerance,
            trial=context.trial,
            outer_seed=context.outer_seed,
            inner_seed=context.inner_seed,
            scenario=context.scenario,
            instance=context.instance,
            source=reference.source,
            source_hash=reference.source_hash,
            benchmark_code_version=reference.benchmark_code_version,
            benchmark_data_version=reference.benchmark_data_version,
            runtime_objective_transform=reference.runtime_objective_transform,
            reporting_objective_transform=reference.reporting_objective_transform,
            fidelity=reference.fidelity,
            hard_error=reference.kind == "exact",
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record without mutable shared state."""
        return {field_name: _jsonable(getattr(self, field_name)) for field_name in self.__dataclass_fields__}


@runtime_checkable
class BreachRecorder(Protocol):
    """Persistence interface consumed by :func:`reference_regret`."""

    def record(self, record: ReferenceBreachRecord) -> None:
        """Persist one record durably without replacing earlier records."""


class JSONLReferenceBreachRecorder:
    """Append one canonical JSON object per line to a breach journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.exists() and not self.path.is_file():
            raise IsADirectoryError(f"Reference-breach journal is not a file: {self.path}")

    def record(self, record: ReferenceBreachRecord) -> None:
        """Append a record; never truncate or rewrite an existing journal."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(record.as_dict(), allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n").encode(
            "utf-8"
        )
        descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise OSError(f"Short write while appending reference breach to {self.path}.")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def reference_regret(
    reference: ObjectiveReference,
    observed_value: float,
    *,
    recorder: BreachRecorder,
    context: ReferenceBreachContext,
) -> float:
    """Return clipped minimization regret and persist material breaches.

    Non-finite observations represent failed/unusable trials and return
    ``+inf`` without creating a false breach.  A best-known breach is recorded
    and clipped to zero.  An exact breach is recorded first and then raised as
    a hard scientific error.  References are never updated in place.
    """
    if isinstance(observed_value, bool):
        raise ObjectiveReferenceError(f"observed_value must be numeric, got {observed_value!r}.")
    try:
        observed = float(observed_value)
    except (TypeError, ValueError) as error:
        raise ObjectiveReferenceError(f"observed_value must be numeric, got {observed_value!r}.") from error
    if not math.isfinite(observed):
        return math.inf

    if observed < reference.value - reference.tolerance:
        record = ReferenceBreachRecord.create(reference, observed, context)
        recorder.record(record)
        if reference.kind == "exact":
            raise ExactReferenceBreachError(record)
        return 0.0
    return max(observed - reference.value, 0.0)


__all__ = [
    "DEFAULT_BEST_KNOWN_PROVENANCE_KEYS",
    "YAHPO_BEST_KNOWN_PROVENANCE_KEYS",
    "BBOBExactReferenceProvider",
    "BreachRecorder",
    "CompositeReferenceProvider",
    "ExactReferenceBreachError",
    "JSONLReferenceBreachRecorder",
    "ManifestReferenceProvider",
    "ObjectiveReference",
    "ObjectiveReferenceError",
    "ObjectiveReferenceProvider",
    "OptBenchExactReferenceProvider",
    "ReferenceBreachContext",
    "ReferenceBreachRecord",
    "ReferenceCompatibilityError",
    "ReferenceKind",
    "ReferenceLookupError",
    "ReferenceManifestError",
    "ReferenceProvenanceError",
    "ReferenceProvider",
    "reference_regret",
]
