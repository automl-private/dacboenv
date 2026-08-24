"""Versioned array contracts for offline dynamic-WEI learning."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

ALPHA_GRID = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)
ACTION_COUNT = 5
OFFLINE_FINAL_SCHEMA_VERSION = "dacbo-offline-final-v3"
BEHAVIOR_COMPONENT = "behavior_f5"
INITIAL_BRANCH_COMPONENT = "initial_same_state_q5"
MIDRUN_BRANCH_SCHEMA_VERSION = "dacbo-offline-branch-f5-v2"

BEHAVIOR_REQUIRED_ARRAYS = frozenset(
    {
        "global_state",
        "action_features",
        "next_global_state",
        "next_action_features",
        "action_index",
        "alpha",
        "reward",
        "terminated",
        "truncated",
        "behavior_probability",
        "behavior_log_probability",
        "task_id",
        "task_index",
        "domain_id",
        "scenario_id",
        "phase_bin",
        "episode_index",
        "episode_offsets",
        "seed",
        "policy_id",
        "bo_evaluations_before",
        "bo_evaluations_after",
        "realized_duration",
        "dataset_metadata_json",
    }
)

BRANCH_REQUIRED_ARRAYS = frozenset(
    {
        "global_state",
        "action_features",
        "action_alpha",
        "q5",
        "valid_action_mask",
        "tie_mask_q5",
        "top1_top2_gap_q5",
        "task_id",
        "domain_id",
        "scenario_id",
        "phase_bin",
        "source_policy_id",
        "source_state_digest",
        "source_replay_digest",
        "candidate_duplicate_groups",
        "branch_protocol_hash",
        "reference_metadata_json",
        "dataset_metadata_json",
    }
)


@dataclass(frozen=True, slots=True)
class DatasetIdentity:
    """Scientific identity embedded in every finalized dataset component."""

    split: str
    component: str
    manifest_hash: str
    source_dataset_sha256: str
    task_split_hash: str


def metadata_from_array(array: np.ndarray) -> dict[str, Any]:
    """Decode one scalar Unicode JSON metadata array."""
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError("dataset_metadata_json must be a scalar fixed-width string array.")
    value = json.loads(str(array.item()))
    if not isinstance(value, dict):
        raise ValueError("Dataset metadata must decode to a JSON object.")
    return value


def ensure_no_object_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    """Reject pickle-requiring arrays and ragged object encodings."""
    offenders = sorted(key for key, value in arrays.items() if np.asarray(value).dtype.hasobject)
    if offenders:
        raise ValueError(f"Offline datasets may not contain object arrays: {offenders}.")


def validate_behavior_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:  # noqa: C901, PLR0912
    """Validate shapes, dtypes, probabilities, and episode boundaries."""
    ensure_no_object_arrays(arrays)
    missing = sorted(BEHAVIOR_REQUIRED_ARRAYS - set(arrays))
    if missing:
        raise ValueError(f"Behavior dataset is missing arrays: {missing}.")
    metadata = metadata_from_array(np.asarray(arrays["dataset_metadata_json"]))
    if metadata.get("schema_version") != OFFLINE_FINAL_SCHEMA_VERSION:
        raise ValueError(f"Unsupported offline schema: {metadata.get('schema_version')!r}.")
    if metadata.get("component") != BEHAVIOR_COMPONENT:
        raise ValueError(f"Expected component {BEHAVIOR_COMPONENT!r}.")
    n = int(np.asarray(arrays["reward"]).shape[0])
    for key in BEHAVIOR_REQUIRED_ARRAYS - {"episode_offsets", "dataset_metadata_json"}:
        if np.asarray(arrays[key]).shape[0] != n:
            raise ValueError(f"Array {key!r} is not transition aligned.")
    if np.asarray(arrays["global_state"]).shape[1:] != (13,):
        raise ValueError("global_state must have trailing shape (13,).")
    if np.asarray(arrays["action_features"]).shape[1:] != (5, 4):
        raise ValueError("action_features must have trailing shape (5, 4).")
    if np.asarray(arrays["next_global_state"]).shape[1:] != (13,):
        raise ValueError("next_global_state must have trailing shape (13,).")
    if np.asarray(arrays["next_action_features"]).shape[1:] != (5, 4):
        raise ValueError("next_action_features must have trailing shape (5, 4).")
    for key in ("global_state", "action_features", "next_global_state", "next_action_features", "reward"):
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ValueError(f"Behavior array {key!r} contains non-finite values.")
    action = np.asarray(arrays["action_index"], dtype=np.int64)
    if np.any((action < 0) | (action >= len(ALPHA_GRID))):
        raise ValueError("Behavior action indices must be in [0, 4].")
    alpha = np.asarray(arrays["alpha"], dtype=np.float64)
    if not np.allclose(alpha, ALPHA_GRID[action], rtol=0.0, atol=1e-7):
        raise ValueError("Behavior alpha values do not match action indices.")
    probability = np.asarray(arrays["behavior_probability"], dtype=np.float64)
    log_probability = np.asarray(arrays["behavior_log_probability"], dtype=np.float64)
    if np.any(~np.isfinite(probability)) or np.any((probability <= 0) | (probability > 1)):
        raise ValueError("Behavior propensities must be finite and in (0, 1].")
    if not np.allclose(log_probability, np.log(probability), atol=1e-6, rtol=1e-6):
        raise ValueError("Behavior log probabilities do not match probabilities.")
    offsets = np.asarray(arrays["episode_offsets"], dtype=np.int64)
    minimum_offset_count = 2
    if (
        offsets.ndim != 1
        or offsets.size < minimum_offset_count
        or offsets[0] != 0
        or offsets[-1] != n
        or np.any(np.diff(offsets) <= 0)
    ):
        raise ValueError("episode_offsets must strictly partition all transitions.")
    terminal = np.logical_or(np.asarray(arrays["terminated"], dtype=bool), np.asarray(arrays["truncated"], dtype=bool))
    for start, stop in itertools.pairwise(offsets):
        if not bool(terminal[int(stop) - 1]) or terminal[int(start) : int(stop) - 1].any():
            raise ValueError("Each offline episode must terminate exactly on its final transition.")
    return metadata


def validate_branch_arrays(  # noqa: C901, PLR0912 - validation is intentionally exhaustive
    arrays: Mapping[str, np.ndarray], *, allow_initial_only: bool = True
) -> dict[str, Any]:
    """Validate a same-state all-action branch dataset."""
    ensure_no_object_arrays(arrays)
    missing = sorted(BRANCH_REQUIRED_ARRAYS - set(arrays))
    if missing:
        raise ValueError(f"Branch dataset is missing arrays: {missing}.")
    metadata = metadata_from_array(np.asarray(arrays["dataset_metadata_json"]))
    component = metadata.get("component")
    if component == INITIAL_BRANCH_COMPONENT and not allow_initial_only:
        raise ValueError("Initial-only Q5 labels are not accepted for this operation.")
    if component not in {INITIAL_BRANCH_COMPONENT, "midrun_same_state_q5_q10"}:
        raise ValueError(f"Unsupported branch component: {component!r}.")
    if component == INITIAL_BRANCH_COMPONENT and metadata.get("schema_version") != OFFLINE_FINAL_SCHEMA_VERSION:
        raise ValueError("Initial counterfactual data must use the finalized dataset schema.")
    if component == "midrun_same_state_q5_q10":
        if metadata.get("schema_version") != MIDRUN_BRANCH_SCHEMA_VERSION:
            raise ValueError("Mid-run branch data must use the current execution-semantics schema.")
        provenance_arrays = {"data_context_split", "environment_context_split"}
        missing_provenance = sorted(provenance_arrays - set(arrays))
        if missing_provenance:
            raise ValueError(f"Mid-run branch data is missing split provenance: {missing_provenance}.")
        data_splits = np.asarray(arrays["data_context_split"]).astype(str)
        environment_splits = np.asarray(arrays["environment_context_split"]).astype(str)
        expected = np.where(data_splits == "train", "train", np.where(data_splits == "dev", "validation", ""))
        if np.any(expected == "") or not np.array_equal(environment_splits, expected):
            raise ValueError("Mid-run branch data/environment context splits are inconsistent.")
    n = int(np.asarray(arrays["q5"]).shape[0])
    if np.asarray(arrays["q5"]).shape != (n, 5):
        raise ValueError("q5 must have shape (states, 5).")
    if "q10" in arrays and np.asarray(arrays["q10"]).shape != (n, 5):
        raise ValueError("q10 must have shape (states, 5).")
    for key in ("global_state", "action_features", "q5"):
        if not np.isfinite(np.asarray(arrays[key], dtype=np.float64)).all():
            raise ValueError(f"Branch array {key!r} contains non-finite values.")
    if "q10" in arrays and not np.isfinite(np.asarray(arrays["q10"], dtype=np.float64)).all():
        raise ValueError("Branch array 'q10' contains non-finite values.")
    for key in BRANCH_REQUIRED_ARRAYS - {"action_alpha", "dataset_metadata_json"}:
        if np.asarray(arrays[key]).shape[0] != n:
            raise ValueError(f"Branch array {key!r} is not state aligned.")
    if not np.array_equal(np.asarray(arrays["action_alpha"], dtype=np.float32), ALPHA_GRID):
        raise ValueError("Branch action identity must be the frozen WEI alpha grid.")
    for value in np.asarray(arrays["candidate_duplicate_groups"]):
        groups = json.loads(str(value))
        if (
            not isinstance(groups, list)
            or len(groups) != ACTION_COUNT
            or any(not isinstance(item, int) for item in groups)
        ):
            raise ValueError("candidate_duplicate_groups must encode five integer group identities.")
    return metadata
