"""Paper-grade evaluation determinism and canonical fingerprint utilities."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EVALUATION_PROTOCOL_VERSION = "evaluation_protocol_v2_deterministic"
PROCESS_DETERMINISM_CONTRACT = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def require_process_determinism() -> dict[str, str]:
    """Fail before environment creation unless the interpreter contract is exact."""
    actual = {name: os.environ.get(name) for name in PROCESS_DETERMINISM_CONTRACT}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in PROCESS_DETERMINISM_CONTRACT.items()
        if actual[name] != expected
    }
    if mismatches:
        detail = ", ".join(
            f"{name}={entry['actual']!r} (expected {entry['expected']!r})" for name, entry in mismatches.items()
        )
        raise RuntimeError(
            "Production evaluation process determinism preflight failed: "
            f"{detail}. Export these variables before starting Python; changing "
            "PYTHONHASHSEED after interpreter startup is ineffective."
        )
    return {name: str(value) for name, value in actual.items()}


def canonical_json(value: Any) -> str:
    """Serialize scientific values without reprs or mapping-order dependence."""
    return json.dumps(_canonical_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_sha256(value: Any) -> str:
    """Hash canonical UTF-8 JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """Hash a file without loading a model artifact into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_policy_seed(
    evaluation_master_seed: int,
    method_id: str,
    context_hash: str,
    *,
    outer_ppo_seed: int | None,
    replicate_id: int = 0,
) -> int:
    """Derive a policy-only stream which cannot alter BO construction."""
    digest = canonical_sha256(
        {
            "context_hash": context_hash,
            "evaluation_master_seed": int(evaluation_master_seed),
            "method_id": method_id,
            "outer_ppo_seed": outer_ppo_seed,
            "replicate_id": int(replicate_id),
            "stream": "policy_only",
        }
    )
    return int(digest[:8], 16)


def canonical_configuration(configuration: Any, configuration_space: Any) -> dict[str, Any]:
    """Represent active and inactive hyperparameters in stable space order."""
    active = dict(configuration)
    names = sorted(str(hyperparameter.name) for hyperparameter in configuration_space.values())
    return {name: _canonical_value(active[name]) if name in active else {"__inactive__": True} for name in names}


def runhistory_fingerprints(runhistory: Any, configuration_space: Any, initial_design_size: int) -> dict[str, Any]:
    """Fingerprint ordered configurations, costs, incumbents, and initial design."""
    configs: list[dict[str, Any]] = []
    costs: list[float] = []
    incumbent = np.inf
    incumbents: list[float] = []
    for trial_key, trial_value in runhistory._data.items():
        configuration = runhistory.get_config(trial_key.config_id)
        configs.append(canonical_configuration(configuration, configuration_space))
        cost = float(np.asarray(trial_value.cost, dtype=np.float64).reshape(-1)[0])
        costs.append(cost)
        if np.isfinite(cost):
            incumbent = min(incumbent, cost)
        incumbents.append(float(incumbent))
    if not 0 < initial_design_size <= len(configs):
        raise RuntimeError("Cannot fingerprint an incomplete or invalid initial design.")
    config_hashes = [canonical_sha256(configuration) for configuration in configs]
    return {
        "initial_design_hash": canonical_sha256(
            {"configuration_hashes": config_hashes[:initial_design_size], "costs": costs[:initial_design_size]}
        ),
        "initial_design_configuration_hashes": config_hashes[:initial_design_size],
        "initial_design_costs": costs[:initial_design_size],
        "first_model_based_candidate_hash": (
            config_hashes[initial_design_size] if len(config_hashes) > initial_design_size else None
        ),
        "evaluated_configuration_trajectory_hash": canonical_sha256(config_hashes),
        "incumbent_trajectory_hash": canonical_sha256(incumbents),
    }


def context_inventory_hash(contexts: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered serialized context inventory."""
    return canonical_sha256(list(contexts))


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_canonical_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Canonical scientific serialization rejects non-finite floats.")
        return {"__float64_hex__": np.float64(value).tobytes().hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported canonical value type {type(value).__name__}.")
