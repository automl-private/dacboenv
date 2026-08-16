"""Contracts for the packaged CARP-S GP-statistics campaign."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from dacboenv.env.observations.gp_hyperparameters import (
    GP_HP_CHANGE_DIM,
    GP_HP_SUMMARY_DIM,
    GPHyperparameterDiagnostics,
    GPHyperparameterFeatureBundle,
)
from dacboenv.experiment.gp_statistics_campaign import (
    GPHyperparameterTrajectoryCallback,
    build_inventory,
    discover_packaged_tasks,
    job_decision,
    write_job_status,
)


def test_inventory_uses_only_bbob_instance_zero_and_every_packaged_yahpo_so(tmp_path: Path) -> None:
    tasks = discover_packaged_tasks()
    bbob = [task for task in tasks if task["task_group"] == "BBOB"]
    yahpo = [task for task in tasks if task["task_group"] == "YAHPO/SO"]
    assert len(bbob) == 24
    assert len(yahpo) == 20
    assert all(task["task_id"].endswith("/0") for task in bbob)
    assert {task["task_id"].split("/")[2] for task in yahpo} == {
        "lcbench",
        "nb301",
        "rbv2_glmnet",
        "rbv2_ranger",
        "rbv2_rpart",
        "rbv2_super",
        "rbv2_xgboost",
    }

    inventory = build_inventory(tmp_path)
    assert inventory["job_count"] == 88
    assert inventory["unique_task_count"] == 44
    assert {job["seed"] for job in inventory["jobs"]} == {0, 1}
    assert all(job["default_n_trials"] > 0 for job in inventory["jobs"])
    assert len({(job["task_id"], job["seed"]) for job in inventory["jobs"]}) == 88


def test_callback_records_one_complete_fixed_shape_bundle(tmp_path: Path) -> None:
    output = tmp_path / "gp_statistics.jsonl"
    callback = GPHyperparameterTrajectoryCallback(str(output), "bbob/2/1/0", 0)
    bundle = GPHyperparameterFeatureBundle(
        summary=np.zeros(GP_HP_SUMMARY_DIM, dtype=np.float32),
        change=np.zeros(GP_HP_CHANGE_DIM, dtype=np.float32),
        raw=np.zeros(64, dtype=np.float32),
        raw_mask=np.zeros(64, dtype=np.float32),
        raw_roles=np.zeros((64, 4), dtype=np.float32),
        state_key=(7, "theta-hash"),
    )
    callback._provider = SimpleNamespace(  # type: ignore[assignment]
        features=lambda _smbo: bundle,
        diagnostics=GPHyperparameterDiagnostics(),
    )
    smbo = SimpleNamespace(runhistory=SimpleNamespace(finished=7))
    info = SimpleNamespace(config={"x1": 2.0, "x0": 1.0})
    callback.on_ask_end(smbo, info)
    callback.on_end(smbo)

    record = json.loads(output.read_text(encoding="utf-8"))
    assert record["task_id"] == "bbob/2/1/0"
    assert record["runhistory_finished"] == 7
    assert record["candidate"] == {"x0": 1.0, "x1": 2.0}
    assert len(record["summary"]) == GP_HP_SUMMARY_DIM
    assert len(record["change"]) == GP_HP_CHANGE_DIM
    assert len(record["raw"]) == 64
    assert np.asarray(record["raw_roles"]).shape == (64, 4)
    assert json.loads((tmp_path / "gp_statistics_callback_status.json").read_text())["status"] == "success"


def test_successful_job_is_skipped_but_identity_conflict_fails_closed(tmp_path: Path) -> None:
    inventory = build_inventory(tmp_path)
    row = inventory["jobs"][0]
    assert job_decision(row, inventory["manifest_hash"]) == "run"
    stats = Path(row["output_directory"]) / "gp_statistics.jsonl"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}\n", encoding="utf-8")
    write_job_status(row, inventory["manifest_hash"], "success", exit_code=0)
    assert job_decision(row, inventory["manifest_hash"]) == "skip"

    status_path = Path(row["output_directory"]) / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["seed"] = 999
    status_path.write_text(json.dumps(status), encoding="utf-8")
    assert job_decision(row, inventory["manifest_hash"]) == "corrupt"
