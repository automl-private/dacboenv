"""Actual Hydra subprocess tests for supervised initial-base artifact lifecycle."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from dacboenv.selective_wei.initial_base_model import load_initial_base
from dacboenv.selective_wei.initial_collection import read_verified, write_verified
from dacboenv.selective_wei.initial_context import InitialContext


def test_initial_base_hydra_cli_resume_and_relocation(tmp_path: Path) -> None:
    """Fit through Hydra, reject overwrites/changed recipes, and resume relocatably."""
    source = tmp_path / "dataset"
    source.mkdir()
    context = InitialContext(
        "engineering",
        ("dimension", "spread"),
        (2.0, 1.0),
        (True, True),
        ("M", "D"),
        2,
        12,
        "design",
        "costs",
        "model",
        "training-data",
        "protocol",
        0,
    )
    for split in ("train", "dev"):
        rows = [
            {
                "recipe": {"split": split, "task_id": f"synthetic-{split}-{index}"},
                "context": asdict(context),
                "targets": {"terminal_paired_scaled": [1.0, 1.0, 0.0, -1.0, 1.0]},
            }
            for index in range(4)
        ]
        write_verified(
            source / f"initial_base_{split}.json",
            {
                "split": split,
                "target_semantics": "full_static_terminal_loss",
                "rows": rows,
                "partition_manifest_hash": "synthetic-partition",
                "features": {},
            },
        )
    output = tmp_path / "run"
    command = [
        sys.executable,
        "-m",
        "dacboenv.experiment.fit_initial_base",
        f"dataset_root={source}",
        f"output_root={output}",
        "features=base_initial_data",
        "model=ridge",
    ]
    environment = dict(os.environ)
    environment.update(
        PYTHONHASHSEED="0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"
    )

    def run(extra: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - fixed local module and generated fixture paths
            command + extra, env=environment, capture_output=True, text=True, timeout=120, check=False
        )

    first = run([])
    assert first.returncode == 0, first.stdout + first.stderr
    model = load_initial_base(output / "base_bundle.json")
    assert model.select(context).action == 3
    assert run([]).returncode != 0
    resumed = run(["resume=true"])
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert run(["resume=true", "seed=1"]).returncode != 0
    moved = tmp_path / "relocated"
    shutil.copytree(output, moved)
    assert run([f"output_root={moved}", "resume=true"]).returncode == 0
    assert np.array_equal(load_initial_base(moved / "base_bundle.json").predict([context]), model.predict([context]))
    assert read_verified(output / "training_complete.json")["status"] == "complete"
    exported = subprocess.run(  # noqa: S603 - generated engineering paths and fixed local module
        [
            sys.executable,
            "-m",
            "dacboenv.experiment.collect_initial_base_policies",
            f"run_root={output}",
            f"output_root={tmp_path / 'export'}",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert exported.returncode == 0, exported.stdout + exported.stderr
    inventory = read_verified(tmp_path / "export" / "initial_base_policy_inventory.json")
    assert inventory["policies"][0]["deployment_kind"] == "initial_base_only"
