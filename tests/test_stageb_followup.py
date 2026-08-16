"""Focused Stage-B matrix, status, selector, and planning tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from dacboenv.experiment.audit_evaluation_matrix import audit_matrix
from dacboenv.experiment.evaluation_status import atomic_json, episode_status, evaluation_cell_hash
from dacboenv.experiment.import_carps_evaluation_matrix import import_carps_matrix
from dacboenv.experiment.merge_evaluation_repairs import merge_repairs
from dacboenv.experiment.nonfeedback_registry import load_registry
from dacboenv.experiment.plan_evaluation_repairs import plan_repairs
from dacboenv.experiment.stageb_followup import (
    _write_generated_scripts,
    build_headroom_manifest,
    discover_stageb_runs,
)
from omegaconf import OmegaConf


def _cfg(method: str, task: str, seed: int, budget: int = 4) -> str:
    model = "  policy_kwargs:\\n    model: /model.zip\\n" if method.startswith("ppo_") else ""
    return (
        f"seed: {seed}\\ntask:\\n  name: {task}\\n  optimization_resources:\\n"
        f"    n_trials: {budget}\\noptimizer:\\n{model}optimizer_id: {method}\\n"
    )


def test_carps_import_exposes_failed_before_logging_and_missing_cells(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    methods = ("ppo_AWEI-Iyahpo-seed0", "static-wei_alpha_discrete-action0")
    tasks = ("bbob/2/1/0", "yahpo/so/rbv2_xgboost/12/None")
    configs = []
    experiment = 0
    # Omit the learned/BBOB cell and the targeted static/YAHPO cell.
    for method in methods:
        for task in tasks:
            if (method.startswith("ppo_") and task.startswith("bbob/")) or (
                method.startswith("static") and task.startswith("yahpo/")
            ):
                continue
            configs.append(
                {"cfg_fn": f"{experiment}/config.yaml", "cfg_str": _cfg(method, task, 5), "experiment_id": experiment}
            )
            experiment += 1
    pd.DataFrame(configs).to_parquet(evaluation / "logs_cfg.parquet", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": row["experiment_id"],
                "n_trials": trial,
                "trial_info__config": f"[{trial}]",
                "trial_value__cost": float(trial),
            }
            for row in configs
            for trial in range(1, 5)
        ]
    ).to_parquet(evaluation / "logs.parquet", index=False)
    output = tmp_path / "followup"
    expected, summary = import_carps_matrix(evaluation, output)
    audit = audit_matrix(output / "imported_status", output / "expected_protocol.json")
    assert expected["method_count"] == 2
    assert summary["expected_cells"] == 4
    assert audit["successful_cells"] == 2
    assert audit["missing_cells"] == 2
    assert all(cell["initial_design_hash_expected"] for cell in expected["cells"])
    repairs = plan_repairs(output / "imported_status", output / "expected_protocol.json")
    assert repairs["repair_cell_count"] == 2


def test_atomic_failed_status_reraises_and_never_writes_success(tmp_path: Path) -> None:
    status = tmp_path / "episode.status.json"
    cell = {
        "method_id": "learned",
        "task_id": "yahpo/so/missing/1/None",
        "evaluation_seed": 3,
        "checkpoint_mode": "final",
        "model_sha256": "hash",
    }
    with (
        pytest.raises(RuntimeError, match="missing reference"),
        episode_status(status, cell=cell, context_hash="context", result_path=tmp_path / "result.json"),
    ):
        raise RuntimeError("missing reference")
    payload = json.loads(status.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["exception_type"] == "RuntimeError"
    assert not (tmp_path / "result.json").exists()


def test_failed_episode_status_propagates_nonzero_from_fresh_process(tmp_path: Path) -> None:
    status = tmp_path / "episode.status.json"
    source = f"""from pathlib import Path
from dacboenv.experiment.evaluation_status import episode_status
p = Path({str(status)!r})
cell = {{'method_id': 'broken', 'task_id': 'yahpo/so/missing/1/None',
        'evaluation_seed': 0, 'checkpoint_mode': 'none', 'model_sha256': None}}
with episode_status(p, cell=cell, context_hash='context', result_path=p.with_name('result.json')):
    raise RuntimeError('missing reference')
"""
    python = Path.cwd() / ".env" / "bin" / "python"
    completed = subprocess.run([str(python), "-c", source], check=False)  # noqa: S603
    assert completed.returncode != 0
    assert json.loads(status.read_text())["status"] == "failed"


def test_headroom_plan_uses_final_and_one_job_per_context_phase() -> None:
    registry = load_registry(Path("artifacts/stageb_followup/nonfeedback_selector_registry.json"))
    inventory = []
    for domain in ("yahpo", "mixed"):
        inventory.append(
            {
                "run_id": f"{domain}-wei-seed0",
                "run_root": f"/{domain}",
                "outer_ppo_seed": 0,
                "training_domain": domain,
                "action_family": "wei",
                "interaction_frequency": 1,
                "config_path": "dacboenv/configs/training/yahpo_ppo_pilot.yaml",
                "model_path": "/model.zip",
                "model_sha256": "model",
                "normalization_path": None,
                "normalization_sha256": None,
            }
        )
    # Supply the only field read from config through a small resolved fixture.
    for item in inventory:
        item["config_path"] = str(Path("dacboenv/configs/training/yahpo_ppo_pilot.yaml").resolve())
    manifest = build_headroom_manifest(inventory, Path.cwd(), registry)
    assert manifest["checkpoint_mode"] == "final"
    assert all(row["branch_actions"] == [0, 1, 2, 3, 4] for row in manifest["jobs"])
    assert all(row["branch_horizons"] == [1, 5, 10] for row in manifest["jobs"])
    assert all(row["checkpoint_mode"] == "final" for row in manifest["jobs"])
    assert all(row["evaluation_budget"] > 0 for row in manifest["jobs"])
    assert manifest["estimated_objective_calls_upper"] > 0
    assert manifest["job_count"] == (24 + 44) * 3


def test_selector_registry_is_complete_deployable_and_self_hashed() -> None:
    registry = load_registry(Path("artifacts/stageb_followup/nonfeedback_selector_registry.json"))
    assert len(registry["entries"]) == 6
    assert all(entry["deployable"] for entry in registry["entries"])
    assert all("privileged" not in entry["selected_nonfeedback_class"] for entry in registry["entries"])


def test_discovers_exactly_twelve_final_stageb_runs(tmp_path: Path) -> None:
    for domain in ("yahpo", "mixed"):
        for family in ("wei", "af_selection"):
            for seed in range(3):
                run = tmp_path / domain / family / str(seed)
                final_step = 80
                action_id = "WEI-discrete-f1" if family == "wei" else "AF-select-f1"
                tasks = (
                    ["yahpo/so/lcbench/126025/None"]
                    if domain == "yahpo"
                    else ["bbob/4/2/1", "yahpo/so/lcbench/126025/None"]
                )
                cfg = OmegaConf.create(
                    {
                        "seed": seed,
                        "action_space_id": action_id,
                        "observation_space_id": "structured" if family == "wei" else "structured-af-selection",
                        "training_instances": {"domain": domain, "manifest_hash": f"train-{domain}"},
                        "experiment": {
                            "total_timesteps": final_step,
                            "vecnormalize": False,
                            "validation": {"manifest_hash": f"validation-{domain}"},
                        },
                        "optimizer": {"gamma": 1.0},
                        "dacboenv": {
                            "task_ids": tasks,
                            "interaction_frequency": 1,
                            "reward_keys": ["reference_regret_improvement"],
                        },
                    }
                )
                config = run / ".hydra" / "config.yaml"
                config.parent.mkdir(parents=True)
                OmegaConf.save(cfg, config)
                checkpoint = run / "validation" / "frequent" / "checkpoints" / "step_80_model.zip"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(f"{domain}-{family}-{seed}".encode())
                (checkpoint.parent.parent / "history.json").write_text(
                    json.dumps(
                        {
                            "checkpoints": [
                                {
                                    "training_step": final_step,
                                    "model_path": str(checkpoint),
                                    "normalization_path": None,
                                    "scores": {"balanced": 0.0},
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                (run / "protocol_metadata.json").write_text(
                    json.dumps({"source_revision": "training-revision"}), encoding="utf-8"
                )
    inventory = discover_stageb_runs(tmp_path)
    assert len(inventory) == 12
    assert {tuple(row["identity"]) for row in inventory} == {
        (domain, family, seed)
        for domain in ("yahpo", "mixed")
        for family in ("wei", "af_selection")
        for seed in range(3)
    }


def test_generated_followup_scripts_need_only_cli_flags_and_pass_bash_n(tmp_path: Path) -> None:
    followup = tmp_path / "followup"
    followup.mkdir()
    _write_generated_scripts(Path.cwd(), Path(".env/bin/python").resolve(), followup)
    for script in followup.glob("*.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)  # noqa: S603, S607
        text = script.read_text(encoding="utf-8")
        assert "PYTHONHASHSEED=0" in text
        assert "DACBO_" not in text


def test_merge_indexes_shared_legacy_result_without_copying_it_per_cell(tmp_path: Path) -> None:
    original = tmp_path / "original"
    repairs = tmp_path / "repairs"
    output = tmp_path / "merged"
    shared_result = tmp_path / "logs.parquet"
    shared_result.write_bytes(b"one shared CARP-S result")
    cell = {
        "method_id": "static",
        "task_id": "bbob/2/1/0",
        "evaluation_seed": 1,
        "checkpoint_mode": "none",
        "model_sha256": None,
    }
    cell_hash = evaluation_cell_hash(cell)
    atomic_json(
        original / cell_hash / "episode.status.json",
        {
            **cell,
            "cell_hash": cell_hash,
            "context_hash": "context",
            "status": "success",
            "result_path": str(shared_result),
            "result_sha256": None,
        },
    )
    expected = tmp_path / "expected.json"
    atomic_json(expected, {"cells": [cell]})
    manifest = merge_repairs(original, repairs, expected, output)
    assert manifest["audit"]["complete"]
    status = json.loads((output / "cells" / cell_hash / "episode.status.json").read_text())
    assert status["source_result_path"] == str(shared_result)
    assert list((output / "cells" / cell_hash).iterdir()) == [output / "cells" / cell_hash / "episode.status.json"]
