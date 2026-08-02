"""Installed-data YAHPO inventory, split, budget, and provenance helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from carps.objective_functions.yahpo import YAHPO_TASK_DATA_DIR
from yahpo_gym import BenchmarkSet
from yahpo_gym.local_config import local_config

from dacboenv.experiment.protocol import OFFICIAL_YAHPO_SO_INSTANCES, YAHPO_SPLIT_MASTER_SEED

YAHPO_TRAIN_COUNTS: Mapping[str, int] = {
    "lcbench": 12,
    "rbv2_glmnet": 12,
    "rbv2_rpart": 12,
    "rbv2_ranger": 12,
    "rbv2_xgboost": 12,
    "rbv2_super": 8,
}
YAHPO_VALIDATION_COUNTS: Mapping[str, int] = {
    "lcbench": 4,
    "rbv2_glmnet": 4,
    "rbv2_rpart": 4,
    "rbv2_ranger": 4,
    "rbv2_xgboost": 4,
    "rbv2_super": 4,
}
ALLOWED_TRAINING_BUDGET_MULTIPLIERS = (0.6, 0.8, 1.0)


class YAHPOCoverageError(RuntimeError):
    """Raised before a split when eligible references or instances are insufficient."""


@dataclass(frozen=True)
class YAHPOSplit:
    """One deterministic, official-test-disjoint train/validation split."""

    train_task_ids: tuple[str, ...]
    validation_task_ids: tuple[str, ...]
    master_seed: int


def yahpo_task_id(scenario: str, instance: str) -> str:
    """Return a canonical CARP-S YAHPO single-objective task ID."""
    return f"yahpo/so/{scenario}/{instance}/None"


def installed_yahpo_inventory(
    scenarios: Sequence[str] = tuple(YAHPO_TRAIN_COUNTS),
) -> dict[str, tuple[str, ...]]:
    """Enumerate installed instances without mutating the on-disk installation."""
    local_config._config = {"data_path": str(YAHPO_TASK_DATA_DIR)}
    inventory: dict[str, tuple[str, ...]] = {}
    for scenario in scenarios:
        benchmark = BenchmarkSet(scenario=scenario, multithread=False)
        inventory[scenario] = tuple(sorted(str(instance) for instance in benchmark.instances))
    return inventory


def official_yahpo_task_ids() -> frozenset[str]:
    """Return the sealed official test inventory."""
    return frozenset(
        yahpo_task_id(scenario, instance)
        for scenario, instances in OFFICIAL_YAHPO_SO_INSTANCES.items()
        for instance in instances
    )


def _scenario_seed(master_seed: int, scenario: str) -> int:
    digest = hashlib.sha256(scenario.encode("utf-8")).digest()
    scenario_word = int.from_bytes(digest[:4], byteorder="little")
    return int(np.random.SeedSequence([master_seed, scenario_word]).generate_state(1, dtype=np.uint32)[0])


def deterministic_yahpo_split(
    inventory: Mapping[str, Sequence[str]],
    provenance_complete_task_ids: Collection[str],
    *,
    master_seed: int = YAHPO_SPLIT_MASTER_SEED,
) -> YAHPOSplit:
    """Split only non-test instances with provenance-complete references.

    Coverage is checked before any IDs are returned; official test instances
    cannot silently fill a shortage.
    """
    eligible_references = set(provenance_complete_task_ids)
    sealed = official_yahpo_task_ids()
    train: list[str] = []
    validation: list[str] = []
    shortages: dict[str, dict[str, int]] = {}
    for scenario in YAHPO_TRAIN_COUNTS:
        candidates = [
            yahpo_task_id(scenario, instance)
            for instance in sorted(str(value) for value in inventory.get(scenario, ()))
            if yahpo_task_id(scenario, instance) not in sealed
            and yahpo_task_id(scenario, instance) in eligible_references
        ]
        required = YAHPO_TRAIN_COUNTS[scenario] + YAHPO_VALIDATION_COUNTS[scenario]
        if len(candidates) < required:
            shortages[scenario] = {"eligible": len(candidates), "required": required}
            continue
        rng = np.random.default_rng(_scenario_seed(master_seed, scenario))
        order = rng.permutation(len(candidates))
        selected = [candidates[int(index)] for index in order[:required]]
        n_train = YAHPO_TRAIN_COUNTS[scenario]
        train.extend(selected[:n_train])
        validation.extend(selected[n_train:])
    if shortages:
        raise YAHPOCoverageError(
            "Cannot freeze YAHPO train/validation manifests until provenance-complete reference coverage is "
            f"sufficient: {json.dumps(shortages, sort_keys=True)}"
        )
    if set(train) & set(validation) or (set(train) | set(validation)) & sealed:
        raise RuntimeError("Internal YAHPO split overlap error.")
    return YAHPOSplit(tuple(train), tuple(validation), int(master_seed))


def yahpo_data_identity(data_root: Path = Path(YAHPO_TASK_DATA_DIR)) -> dict[str, str]:
    """Return version/Git/config-space hashes for the installed data tree."""
    version_value = (data_root / "VERSION").read_text(encoding="utf-8").strip()
    if version_value.upper().startswith("VERSION:"):
        version_value = version_value.partition(":")[2].strip()
    head_path = data_root / ".git" / "HEAD"
    commit = "unavailable"
    if head_path.is_file():
        head = head_path.read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref_path = data_root / ".git" / head.partition(":")[2].strip()
            if ref_path.is_file():
                commit = ref_path.read_text(encoding="utf-8").strip()
        else:
            commit = head
    digest = hashlib.sha256()
    for path in sorted(data_root.glob("*/config_space.json")):
        digest.update(path.relative_to(data_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {
        "version": version_value,
        "git_commit": commit,
        "config_space_tree_sha256": digest.hexdigest(),
    }


def effective_training_budget(native_budget: int, multiplier: float, *, initial_design_size: int) -> int:
    """Apply an allowed multiplier while retaining a valid initial design."""
    if float(multiplier) not in ALLOWED_TRAINING_BUDGET_MULTIPLIERS:
        raise ValueError(
            f"YAHPO training budget multiplier must be one of {ALLOWED_TRAINING_BUDGET_MULTIPLIERS}, "
            f"got {multiplier!r}."
        )
    if native_budget <= 0 or initial_design_size <= 0:
        raise ValueError("native_budget and initial_design_size must be positive.")
    effective = int(np.ceil(native_budget * float(multiplier)))
    if effective < initial_design_size:
        raise ValueError(f"Effective budget {effective} is smaller than initial-design size {initial_design_size}.")
    return effective


def apply_yahpo_budget_multiplier(
    native_budget: int,
    multiplier: float,
    *,
    initial_design_size: int,
    split: str,
) -> int:
    """Change training budgets only; validation/test always return native budget."""
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"split must be train, validation, or test, got {split!r}.")
    if split != "train":
        if float(multiplier) != 1.0:
            raise ValueError(f"YAHPO {split} contexts must use full native budget (multiplier 1.0).")
        return int(native_budget)
    return effective_training_budget(native_budget, multiplier, initial_design_size=initial_design_size)


__all__ = [
    "ALLOWED_TRAINING_BUDGET_MULTIPLIERS",
    "YAHPO_TRAIN_COUNTS",
    "YAHPO_VALIDATION_COUNTS",
    "YAHPOCoverageError",
    "YAHPOSplit",
    "apply_yahpo_budget_multiplier",
    "deterministic_yahpo_split",
    "effective_training_budget",
    "installed_yahpo_inventory",
    "official_yahpo_task_ids",
    "yahpo_data_identity",
    "yahpo_task_id",
]
