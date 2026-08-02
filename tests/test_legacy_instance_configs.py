"""Fail-closed coverage for pre-manifest instance aliases."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from dacboenv.dacboenv import DACBOEnv
from omegaconf import OmegaConf


def test_every_legacy_instance_alias_is_explicitly_deprecated() -> None:
    config_root = Path("dacboenv/configs/instances")
    configs = sorted(config_root.glob("*.yaml"))
    assert configs
    for path in configs:
        cfg = OmegaConf.load(path)
        assert cfg.dacboenv.deprecated_legacy_instance_config == cfg.instance_set_id, path


def test_legacy_instance_alias_fails_before_environment_setup() -> None:
    with pytest.raises(ValueError, match="Legacy configs/instances composition"):
        DACBOEnv(
            task_ids=["bbob/2/3/0"],
            deprecated_legacy_instance_config="bbob2d_fid3_3seeds",
        )


@pytest.mark.parametrize("target", ["legacy-carps", "legacy-smac", "legacy-optbench", "metabo"])
def test_legacy_make_targets_fail_before_external_or_environment_changes(target: str) -> None:
    make_executable = shutil.which("make")
    assert make_executable is not None
    result = subprocess.run(  # noqa: S603
        [make_executable, "-s", target],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "deprecated pre-manifest" in result.stderr
