"""Import a broad CARP-S export into the explicit Stage-B matrix contract."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.experiment.evaluation_status import atomic_json, evaluation_cell_hash


def _config_identity(row: Any) -> dict[str, Any]:
    serialized = str(row.cfg_str).replace("\\n", "\n")
    method_match = re.search(r"^optimizer_id: (.+)$", serialized, re.MULTILINE)
    task_match = re.search(r"^  name: (.+)$", serialized, re.MULTILINE)
    seed_match = re.search(r"^seed: ([0-9]+)$", serialized, re.MULTILINE)
    budget_match = re.search(r"^    n_trials: ([0-9]+)$", serialized, re.MULTILINE)
    model_match = re.search(r"^    model: (.+)$", serialized, re.MULTILINE)
    if None in (method_match, task_match, seed_match, budget_match):
        raise ValueError(f"Could not parse exported CARP-S config identity: {row.cfg_fn}")
    assert method_match is not None
    assert task_match is not None
    assert seed_match is not None
    assert budget_match is not None
    method = method_match.group(1)
    task = task_match.group(1)
    seed = int(seed_match.group(1))
    model_path = None if model_match is None else model_match.group(1)
    return {
        "method_id": method,
        "task_id": task,
        "evaluation_seed": seed,
        "checkpoint_mode": "final" if model_path is not None else "none",
        "model_path_recorded": None if model_path is None else str(model_path),
        "evaluation_budget": int(budget_match.group(1)),
    }


def import_carps_matrix(  # noqa: C901, PLR0912, PLR0915
    evaluation_root: Path,
    output_root: Path,
    *,
    model_inventory: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Infer the complete rectangular matrix and materialize compatibility statuses.

    The original broad launch was a rectangular method/task/evaluation-seed
    matrix. Failed task composition never reached ``logs_cfg``; reconstructing
    the Cartesian inventory is therefore necessary to expose those failures.
    """
    cfg_path = evaluation_root / "logs_cfg.parquet"
    logs_path = evaluation_root / "logs.parquet"
    if not cfg_path.is_file() or not logs_path.is_file():
        raise FileNotFoundError("CARP-S matrix import requires logs_cfg.parquet and logs.parquet.")
    configs = pd.read_parquet(cfg_path)
    logs = pd.read_parquet(logs_path)
    identities = [_config_identity(row) for row in configs.itertuples(index=False)]
    methods = sorted({row["method_id"] for row in identities})
    tasks = sorted({row["task_id"] for row in identities})
    seeds = sorted({row["evaluation_seed"] for row in identities})
    model_hash_by_method: dict[str, str] = {}
    if model_inventory:
        for method in methods:
            if not method.startswith("ppo_"):
                continue
            matches = [
                item
                for item in model_inventory
                if f"seed{item['outer_ppo_seed']}" in method
                and ((item["action_family"] == "wei") == ("AWEI" in method))
                and ((item["training_domain"] == "mixed") == ("Imixed" in method))
            ]
            if len(matches) != 1:
                raise ValueError(f"Could not map broad-evaluation method {method!r} to exactly one Stage-B model.")
            model_hash_by_method[method] = str(matches[0]["model_sha256"])
    budget_by_task = {row["task_id"]: row["evaluation_budget"] for row in identities}
    observed_by_key = {
        (row["method_id"], row["task_id"], row["evaluation_seed"]): (row, int(configs.iloc[index].experiment_id))
        for index, row in enumerate(identities)
    }
    finished = logs.groupby("experiment_id", sort=False).n_trials.max().to_dict()
    # Recover the paired initial-design fingerprint from every completed
    # legacy method in a context.  CARP-S exports configuration lists and
    # costs losslessly; this representation is also emitted by the repair
    # worker and is independent of optimizer/policy identity.
    initial_hashes_by_context: dict[tuple[str, int], set[str]] = {}
    initial_size_by_context: dict[tuple[str, int], int] = {}
    identity_by_experiment = {experiment_id: identity for identity, experiment_id in observed_by_key.values()}
    for experiment_id, rows in logs.groupby("experiment_id", sort=False):
        identity = identity_by_experiment[int(experiment_id)]
        if int(finished.get(experiment_id, 0)) < int(identity["evaluation_budget"]):
            continue
        context = (identity["task_id"], identity["evaluation_seed"])
        # SMAC's max-ratio cap truncates to an integer (e.g. 267 -> 53),
        # rather than applying mathematical ceiling.
        initial_size = max(1, int(0.2 * int(identity["evaluation_budget"])))
        prefix = rows.sort_values("n_trials").head(initial_size)
        if len(prefix) != initial_size:
            continue
        values = []
        for configuration, cost in zip(prefix.trial_info__config, prefix.trial_value__cost, strict=True):
            parsed = ast.literal_eval(configuration) if isinstance(configuration, str) else configuration
            values.append({"config": parsed, "cost": float(cost)})
        initial_hashes_by_context.setdefault(context, set()).add(canonical_sha256(values))
        initial_size_by_context[context] = initial_size
    initial_hash_by_context: dict[tuple[str, int], str] = {}
    for context, hashes in initial_hashes_by_context.items():
        if len(hashes) != 1:
            raise RuntimeError(f"Legacy methods disagree on the initial design for context {context!r}: {hashes!r}")
        initial_hash_by_context[context] = hashes.pop()
    cells = []
    for method in methods:
        for task in tasks:
            for seed in seeds:
                checkpoint = "final" if method.startswith("ppo_") else "none"
                cells.append(
                    {
                        "method_id": method,
                        "task_id": task,
                        "evaluation_seed": seed,
                        "checkpoint_mode": checkpoint,
                        "model_sha256": model_hash_by_method.get(method),
                        "evaluation_budget": budget_by_task[task],
                        "initial_design_size": initial_size_by_context[(task, seed)],
                        "initial_design_hash_expected": initial_hash_by_context[(task, seed)],
                    }
                )
    expected = {
        "schema_version": "stageb-broad-evaluation-expected-v1",
        "cells": cells,
        "method_count": len(methods),
        "task_count": len(tasks),
        "evaluation_seed_count": len(seeds),
        "matrix_hash": canonical_sha256(cells),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    expected_path = output_root / "expected_protocol.json"
    atomic_json(expected_path, expected)

    status_root = output_root / "imported_status"
    success_count = 0
    incomplete_count = 0
    for cell in cells:
        key = (cell["method_id"], cell["task_id"], cell["evaluation_seed"])
        observed = observed_by_key.get(key)
        if observed is None:
            continue
        _identity, experiment_id = observed
        if int(finished.get(experiment_id, 0)) < int(cell["evaluation_budget"]):
            incomplete_count += 1
            continue
        cell_hash = evaluation_cell_hash(cell)
        status = {
            **cell,
            "cell_hash": cell_hash,
            "context_hash": canonical_sha256(
                {"task_id": cell["task_id"], "inner_seed": cell["evaluation_seed"], "budget": cell["evaluation_budget"]}
            ),
            "status": "success",
            "result_path": str(logs_path.resolve()),
            "result_sha256": None,
            "objective_evaluations_completed": int(cell["evaluation_budget"]),
            "initial_design_hash": cell["initial_design_hash_expected"],
            "legacy_carps_experiment_id": experiment_id,
        }
        atomic_json(status_root / cell_hash / "episode.status.json", status)
        success_count += 1
    summary = {
        "submitted_config_rows": len(configs),
        "expected_cells": len(cells),
        "imported_successes": success_count,
        "incomplete_submitted_cells": incomplete_count,
        "unsubmitted_or_failed_before_config_logging": len(cells) - success_count - incomplete_count,
    }
    atomic_json(output_root / "carps_import_summary.json", summary)
    return expected, summary


__all__ = ["import_carps_matrix"]
