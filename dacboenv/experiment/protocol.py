"""Frozen instance-manifest and seed protocol helpers.

The helpers in this module deliberately do not alter environment behavior.  They
provide a small, auditable foundation for versioned task manifests and dedicated
validation/test seed streams.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from carps.utils.env_vars import CARPS_ROOT
from omegaconf import DictConfig, OmegaConf

MANIFEST_SCHEMA_VERSION = 1
"""Schema version of the version-controlled scientific instance manifests."""

YAHPO_SPLIT_MASTER_SEED = 3_670_740_481
"""Reserved master seed for a future installed-data YAHPO train/validation split."""

VALIDATION_INNER_MASTER_SEED = 3_670_740_482
"""Master seed used only to freeze validation inner seeds."""

TEST_INNER_MASTER_SEED = 3_670_740_483
"""Independent master seed used only to freeze test inner seeds."""

EXPECTED_NATIVE_BBOB_DIMENSIONS = (2, 4, 8, 16, 32)
BBOB_TRAIN_FUNCTIONS = (3, 6, 8, 13, 17, 21)
BBOB_VALIDATION_FUNCTIONS = (2, 7, 11, 16, 20)
BBOB_STRICT_TEST_FUNCTIONS = (1, 4, 5, 9, 10, 12, 14, 15, 18, 19, 22, 23, 24)
BBOB_STRESS_TEST_FUNCTIONS = (1, 5, 9, 12, 15, 19, 22, 24)

OFFICIAL_YAHPO_SO_INSTANCES: Mapping[str, tuple[str, ...]] = {
    "lcbench": ("167168", "189873", "189906"),
    "nb301": ("CIFAR10",),
    "rbv2_glmnet": ("375", "458"),
    "rbv2_ranger": ("16", "42"),
    "rbv2_rpart": ("14", "40499"),
    "rbv2_super": ("1053", "1457", "1063", "1479", "15", "1468"),
    "rbv2_xgboost": ("12", "1501", "16", "40499"),
}

_UINT32_LIMIT = 2**32
_BBOB_TASK_ID = re.compile(r"^bbob/(?P<dimension>\d+)/(?P<function_id>\d+)/(?P<instance_id>\d+)$")
_BBOB_CONFIG_FILENAME = re.compile(r"^cfg_(?P<dimension>\d+)_(?P<function_id>\d+)_(?P<instance_id>\d+)\.yaml$")
_YAHPO_SO_TASK_NAME = re.compile(r"^\s*name:\s*(yahpo/so/[^/\s]+/[^/\s]+/None)\s*$", re.MULTILINE)


class ManifestValidationError(ValueError):
    """Raised when a manifest violates the versioned protocol."""


class ManifestUnavailableError(RuntimeError):
    """Raised when code tries to use a deliberately non-runnable manifest."""


@dataclass(frozen=True)
class LegacyYahpoReference:
    """One checked empirical YAHPO reference from the repository CSV."""

    scenario: str
    instance: str
    target: str
    minimization_value: float

    @property
    def task_id(self) -> str:
        """Return the corresponding CARP-S single-objective task ID."""
        return f"yahpo/so/{self.scenario}/{self.instance}/None"

    @property
    def accuracy_scale(self) -> str:
        """Return the scale used by the checked legacy reference."""
        return "percent" if self.scenario in {"lcbench", "nb301"} else "fraction"

    @property
    def one_minus_accuracy(self) -> float:
        """Convert legacy ``-accuracy`` to the explicit ``1 - accuracy`` cost."""
        divisor = 100.0 if self.accuracy_scale == "percent" else 1.0
        return 1.0 + self.minimization_value / divisor


def frozen_inner_seeds(master_seed: int, count: int) -> tuple[int, ...]:
    """Derive a stable uint32 seed list directly from one dedicated master seed."""
    if isinstance(master_seed, bool) or not isinstance(master_seed, int) or not 0 <= master_seed < _UINT32_LIMIT:
        raise ValueError(f"master_seed must be a uint32 integer, got {master_seed!r}.")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"count must be a positive integer, got {count!r}.")
    state = np.random.SeedSequence(master_seed).generate_state(count, dtype=np.uint32)
    seeds = tuple(int(seed) for seed in state)
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"Master seed {master_seed} unexpectedly generated duplicate inner seeds.")
    return seeds


def expected_bbob_task_ids(
    functions: Sequence[int], dimensions: Sequence[int], native_instance: int
) -> tuple[str, ...]:
    """Build task IDs in deterministic dimension-major/function-major order."""
    return tuple(
        f"bbob/{dimension}/{function_id}/{native_instance}" for dimension in dimensions for function_id in functions
    )


def official_yahpo_so_task_ids() -> tuple[str, ...]:
    """Return the official YAHPO-SO final-test tasks in protocol order."""
    return tuple(
        f"yahpo/so/{scenario}/{instance}/None"
        for scenario, instances in OFFICIAL_YAHPO_SO_INSTANCES.items()
        for instance in instances
    )


def sealed_final_test_task_ids() -> frozenset[str]:
    """Return every intrinsically sealed BBOB and official YAHPO final-test task."""
    strict_bbob = expected_bbob_task_ids(BBOB_STRICT_TEST_FUNCTIONS, (2, 8, 16), 2)
    return frozenset((*strict_bbob, *official_yahpo_so_task_ids()))


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ManifestValidationError("Manifest mapping keys must all be strings.")
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ManifestValidationError(f"Unsupported value in manifest: {value!r}.")


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Hash canonical JSON content, excluding the self-referential hash field."""
    payload = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    canonical = json.dumps(
        _plain_value(payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_manifest_identity(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Expected schema_version {MANIFEST_SCHEMA_VERSION}, got {manifest.get('schema_version')!r}."
        )
    if not isinstance(manifest.get("id"), str) or not manifest["id"]:
        raise ManifestValidationError("Manifest id must be a non-empty string.")
    if manifest.get("domain") not in {"bbob", "yahpo", "mixed"}:
        raise ManifestValidationError(f"Unsupported manifest domain {manifest.get('domain')!r}.")
    if manifest.get("split") not in {"train", "validation", "test"}:
        raise ManifestValidationError(f"Unsupported manifest split {manifest.get('split')!r}.")
    if manifest.get("status") not in {"ready", "defined", "blocked"}:
        raise ManifestValidationError(f"Unsupported manifest status {manifest.get('status')!r}.")


def _validate_manifest_availability(manifest: Mapping[str, Any]) -> None:
    if not isinstance(manifest.get("runnable"), bool):
        raise ManifestValidationError("Manifest runnable must be a boolean.")
    if manifest["status"] != "ready" and manifest["runnable"]:
        raise ManifestValidationError("Only a ready manifest may be marked runnable.")
    if not manifest["runnable"]:
        blockers = manifest.get("blockers")
        if not isinstance(blockers, list) or not blockers or not all(isinstance(item, str) for item in blockers):
            raise ManifestValidationError("A non-runnable manifest must list explicit blockers.")


def _validate_manifest_contexts(manifest: Mapping[str, Any]) -> None:
    task_ids = manifest.get("task_ids")
    if not isinstance(task_ids, list) or not all(isinstance(task_id, str) for task_id in task_ids):
        raise ManifestValidationError("Manifest task_ids must be a list of strings.")
    if len(set(task_ids)) != len(task_ids):
        raise ManifestValidationError("Manifest task_ids must be unique.")
    if manifest["runnable"] and not task_ids:
        raise ManifestValidationError("A runnable manifest must contain at least one task.")

    inner_seeds = manifest.get("inner_seeds")
    if not isinstance(inner_seeds, list):
        raise ManifestValidationError("Manifest inner_seeds must be a list.")
    for seed in inner_seeds:
        if seed is None:
            continue
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < _UINT32_LIMIT:
            raise ManifestValidationError(f"Inner seed must be null or uint32, got {seed!r}.")
    fixed_seeds = [seed for seed in inner_seeds if seed is not None]
    if len(set(fixed_seeds)) != len(fixed_seeds):
        raise ManifestValidationError("Fixed inner seeds must be unique.")


def _validate_manifest_digest(manifest: Mapping[str, Any]) -> None:
    stored_hash = manifest.get("manifest_hash")
    if not isinstance(stored_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", stored_hash):
        raise ManifestValidationError("manifest_hash must be a lowercase SHA-256 digest.")
    computed_hash = manifest_hash(manifest)
    if stored_hash != computed_hash:
        raise ManifestValidationError(f"Manifest hash mismatch: stored {stored_hash}, computed {computed_hash}.")


def validate_manifest_structure(manifest: Mapping[str, Any]) -> None:
    """Validate common manifest fields and its canonical content hash."""
    _validate_manifest_identity(manifest)
    _validate_manifest_availability(manifest)
    _validate_manifest_contexts(manifest)
    _validate_manifest_digest(manifest)


def load_manifest(path: Path) -> dict[str, Any]:
    """Load a YAML manifest into plain Python values and validate its structure."""
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ManifestValidationError(f"Manifest {path} must contain a mapping.")
    manifest = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(manifest, dict):
        raise ManifestValidationError(f"Manifest {path} must contain a mapping.")
    validate_manifest_structure(manifest)
    return manifest


def require_runnable_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail clearly if an incomplete manifest is selected for execution."""
    validate_manifest_structure(manifest)
    if not manifest["runnable"]:
        blockers = "; ".join(str(item) for item in manifest["blockers"])
        raise ManifestUnavailableError(f"Manifest {manifest['id']!r} is not runnable: {blockers}")


def parse_bbob_task_id(task_id: str) -> tuple[int, int, int]:
    """Parse a native BBOB task ID into dimension, function, and instance."""
    match = _BBOB_TASK_ID.fullmatch(task_id)
    if match is None:
        raise ManifestValidationError(f"Invalid native BBOB task ID {task_id!r}.")
    return tuple(int(match.group(name)) for name in ("dimension", "function_id", "instance_id"))


def discover_native_bbob_configs(config_root: Path | None = None) -> dict[tuple[int, int, int], Path]:
    """Index checked-out CARP-S BBOB YAMLs and enforce the audited dimensions."""
    root = config_root or Path(CARPS_ROOT) / "configs" / "task" / "BBOB"
    if not root.is_dir():
        raise FileNotFoundError(f"CARP-S BBOB config directory does not exist: {root}")
    configs: dict[tuple[int, int, int], Path] = {}
    for path in sorted(root.glob("cfg_*.yaml")):
        match = _BBOB_CONFIG_FILENAME.fullmatch(path.name)
        if match is None:
            continue
        key = tuple(int(match.group(name)) for name in ("dimension", "function_id", "instance_id"))
        if key in configs:
            raise ManifestValidationError(f"Duplicate CARP-S BBOB config for {key}: {configs[key]} and {path}.")
        configs[key] = path
    dimensions = tuple(sorted({key[0] for key in configs}))
    if dimensions != EXPECTED_NATIVE_BBOB_DIMENSIONS:
        raise ManifestValidationError(
            "Checked-out CARP-S BBOB dimensions differ from the audited protocol: "
            f"expected {list(EXPECTED_NATIVE_BBOB_DIMENSIONS)}, found {list(dimensions)} below {root}."
        )
    return configs


def validate_native_bbob_manifest(manifest: Mapping[str, Any], config_root: Path | None = None) -> None:
    """Prove that every BBOB manifest entry has a native CARP-S YAML."""
    validate_manifest_structure(manifest)
    if manifest["domain"] != "bbob":
        raise ManifestValidationError(f"Expected a BBOB manifest, got {manifest['domain']!r}.")
    native_configs = discover_native_bbob_configs(config_root)
    missing: list[str] = []
    for task_id in manifest["task_ids"]:
        key = parse_bbob_task_id(task_id)
        if key not in native_configs:
            missing.append(task_id)
    if missing:
        raise ManifestValidationError(f"Manifest tasks have no native CARP-S YAML: {missing}.")


def bbob_function_ids(manifest: Mapping[str, Any]) -> frozenset[int]:
    """Return the BBOB function identities represented by a manifest."""
    return frozenset(parse_bbob_task_id(task_id)[1] for task_id in manifest["task_ids"])


def fixed_contexts(manifest: Mapping[str, Any]) -> frozenset[tuple[str, int]]:
    """Return task/inner-seed pairs for a frozen manifest."""
    if any(seed is None for seed in manifest["inner_seeds"]):
        raise ManifestValidationError(f"Manifest {manifest['id']!r} uses a dynamic seed stream.")
    return frozenset((task_id, seed) for task_id in manifest["task_ids"] for seed in manifest["inner_seeds"])


def discover_official_yahpo_so_configs(config_root: Path | None = None) -> dict[str, Path]:
    """Index the checked CARP-S YAHPO-SO configs by resolved task name."""
    root = config_root or Path(CARPS_ROOT) / "configs" / "task" / "YAHPO" / "SO"
    if not root.is_dir():
        raise FileNotFoundError(f"CARP-S YAHPO-SO config directory does not exist: {root}")
    configs: dict[str, Path] = {}
    for path in sorted(root.glob("cfg_*.yaml")):
        match = _YAHPO_SO_TASK_NAME.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise ManifestValidationError(f"Could not resolve a YAHPO-SO task name from {path}.")
        task_id = match.group(1)
        if task_id in configs:
            raise ManifestValidationError(f"Duplicate CARP-S YAHPO-SO config for {task_id!r}.")
        configs[task_id] = path
    return configs


def load_legacy_yahpo_references(path: Path) -> dict[str, LegacyYahpoReference]:
    """Load the checked empirical YAHPO minima and key them by CARP-S task ID."""
    references: dict[str, LegacyYahpoReference] = {}
    with path.open(encoding="utf-8", newline="") as file_handle:
        for row in csv.DictReader(file_handle):
            reference = LegacyYahpoReference(
                scenario=row["bench"],
                instance=row["instance"],
                target=row["metric"],
                minimization_value=float(row["f_min"]),
            )
            if reference.task_id in references:
                raise ManifestValidationError(f"Duplicate empirical reference for {reference.task_id!r}.")
            references[reference.task_id] = reference
    return references


def _validate_yahpo_coverage(
    expected_tasks: tuple[str, ...], configs: Mapping[str, Path], references: Mapping[str, LegacyYahpoReference]
) -> None:
    missing_configs = sorted(set(expected_tasks).difference(configs))
    missing_references = sorted(set(expected_tasks).difference(references))
    if missing_configs or missing_references:
        raise ManifestValidationError(
            f"Official YAHPO coverage is incomplete: missing configs={missing_configs}, "
            f"missing references={missing_references}."
        )


def _validate_yahpo_task_metadata(
    manifest: Mapping[str, Any],
    expected_tasks: tuple[str, ...],
    references: Mapping[str, LegacyYahpoReference],
) -> None:
    task_metadata = manifest.get("tasks")
    if not isinstance(task_metadata, list) or [item.get("task_id") for item in task_metadata] != list(expected_tasks):
        raise ManifestValidationError("Official YAHPO task metadata must cover task_ids in the same order.")
    for item in task_metadata:
        reference = references[item["task_id"]]
        if item.get("target") != reference.target:
            raise ManifestValidationError(f"Target mismatch for {item['task_id']!r}.")
        if item.get("objective_transform") != "one_minus_accuracy":
            raise ManifestValidationError(f"Objective transform mismatch for {item['task_id']!r}.")
        if item.get("accuracy_scale") != reference.accuracy_scale:
            raise ManifestValidationError(f"Accuracy scale mismatch for {item['task_id']!r}.")
        if not np.isclose(
            float(item.get("reference_cost")),
            reference.one_minus_accuracy,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ManifestValidationError(f"Reference cost mismatch for {item['task_id']!r}.")
        if not np.isclose(
            float(item.get("legacy_minimization_reference")),
            reference.minimization_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ManifestValidationError(f"Legacy reference mismatch for {item['task_id']!r}.")


def validate_official_yahpo_manifest(
    manifest: Mapping[str, Any], *, config_root: Path | None = None, reference_csv: Path
) -> None:
    """Validate official task/config/reference coverage without importing YAHPO data."""
    validate_manifest_structure(manifest)
    if manifest["domain"] != "yahpo" or manifest["split"] != "test":
        raise ManifestValidationError("The official YAHPO-SO manifest must be a YAHPO test manifest.")
    expected_tasks = official_yahpo_so_task_ids()
    if tuple(manifest["task_ids"]) != expected_tasks:
        raise ManifestValidationError("The official YAHPO-SO manifest does not match the frozen task protocol.")

    configs = discover_official_yahpo_so_configs(config_root)
    references = load_legacy_yahpo_references(reference_csv)
    _validate_yahpo_coverage(expected_tasks, configs, references)
    source = manifest.get("reference_source")
    if not isinstance(source, Mapping) or source.get("sha256") != file_sha256(reference_csv):
        raise ManifestValidationError("The official YAHPO reference-source hash is stale.")
    _validate_yahpo_task_metadata(manifest, expected_tasks, references)
