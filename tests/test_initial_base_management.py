"""Hydra preparation and conservative archive compatibility checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.selective_wei.initial_collection import read_verified
from dacboenv.selective_wei.initial_data_reuse import audit_initial_data_reuse


def test_initial_campaign_hydra_prepare_and_status(tmp_path: Path) -> None:
    """Prepare two explicit engineering contexts without running a BO job."""
    manifest = {"task_splits": {"train": ["bbob/2/3/0"], "dev": ["yahpo/so/lcbench/167200/None"], "holdout": []}}
    manifest["manifest_hash"] = canonical_sha256(manifest)
    source = tmp_path / "partition.json"
    source.write_text(json.dumps(manifest))
    output = tmp_path / "campaign"
    command = [sys.executable, "-m", "dacboenv.experiment.initial_base_campaign", f"output_root={output}"]
    environment = dict(os.environ)
    environment.update(
        PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )
    prepared = subprocess.run(  # noqa: S603 - fixed local module and synthetic fixture paths
        [*command, "operation=prepare", f"partition_manifest={source}", "seeds=[0]"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    frozen = read_verified(output / "manifest.json")
    assert len(frozen["jobs"]) == 2
    assert {row["recipe"]["split"] for row in frozen["jobs"]} == {"train", "dev"}
    assert frozen["features"]["version"] == "initial-mdgu-v2"
    status = subprocess.run(  # noqa: S603 - read-only campaign audit
        [*command, "operation=status"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert status.returncode == 0, status.stdout + status.stderr
    assert read_verified(output / "status.json")["missing_indices"] == [0, 1]
    preflight = subprocess.run(  # noqa: S603 - compose-only actual Hydra worker path
        [*command, "operation=preflight"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert preflight.returncode == 0, preflight.stdout + preflight.stderr
    assert '"objective_evaluations": 0' in preflight.stdout


def test_transition_archive_cannot_fabricate_initial_features(tmp_path: Path) -> None:
    """Even GP-labelled transition arrays do not prove raw initialization exists."""
    path = tmp_path / "fixture.npz"
    for split in ("train", "test"):
        np.savez_compressed(
            path,
            dataset_metadata_json=np.asarray(json.dumps({"context_split": split})),
            observations__gp_hp_summary=np.zeros((1, 24)),
        )
        audit = audit_initial_data_reuse(path)
        assert not audit["training_eligible"]
        assert not audit["raw_D0_available"]
        assert not audit["compatible_initial_features"]
