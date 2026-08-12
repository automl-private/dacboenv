"""CARP-S baseline launcher contracts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "scripts" / "eval_structured_baselines.sh"


def test_baseline_launcher_uses_carps_with_frozen_deterministic_contexts(tmp_path: Path) -> None:
    """Dry-run commands must pair every method through native CARP-S contexts."""
    environment = os.environ.copy()
    environment.update(
        {
            "DACBO_BASELINE_DRY_RUN": "1",
            "DACBO_BASELINE_RUN_DIR": str(tmp_path / "results"),
            "DACBO_EVALUATION_SET": "bbob_validation",
        }
    )
    completed = subprocess.run(  # noqa: S603
        ["/bin/bash", str(LAUNCHER)],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    output = completed.stdout
    assert output.count("-m carps.run") == 9
    assert "dacboenv.experiment.baseline" not in output
    assert "dacboenv.experiment.default_smac" not in output
    assert "+task/BBOB=cfg_4_2_1" in output
    assert "cfg_8_20_1" in output
    assert "seed=1349011988" in output
    assert "1294824964" in output
    assert "dacboenv.context_split=validation" in output
    assert "RoundRobinInstanceSelector" in output
    assert "instance_selector_class.offset=0" in output
    assert r"\$\{optimizer_id\}" in output
    assert "36ed3fb56ddc141069b1efad21f4f2ee51d98fed5a0ebaf8c1cdc0d3fcfec196" in output


def test_baseline_launcher_exports_process_determinism_contract() -> None:
    """The launcher must establish determinism before starting Python."""
    source = LAUNCHER.read_text(encoding="utf-8")
    source_position = source.index('source "${script_directory}/evaluation_determinism_env.sh"')
    python_position = source.index("${python_bin} -c")
    assert source_position < python_position
