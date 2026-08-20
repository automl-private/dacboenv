"""Final-checkpoint selection without validation history."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from omegaconf import OmegaConf

from dacboenv.experiment.checkpoint_selection import select_checkpoint


def _write_valid_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("placeholder.txt", "final-model")


def test_final_selection_uses_training_complete_root_model(tmp_path: Path) -> None:
    """The final model must not depend on frequent-validation artifacts."""
    run_root = tmp_path / "run"
    hydra_root = run_root / ".hydra"
    hydra_root.mkdir(parents=True)

    config = OmegaConf.create(
        {
            "experiment": {
                "total_timesteps": 6144,
                "vecnormalize": False,
            }
        }
    )
    OmegaConf.save(config, hydra_root / "config.yaml")

    model_path = run_root / "model.zip"
    _write_valid_zip(model_path)
    (run_root / "training_complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "num_timesteps": 6144,
                "expected_final_timesteps": 6144,
                "model_path": str(model_path),
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert not (run_root / "validation" / "frequent" / "history.json").exists()

    selected = select_checkpoint(run_root, "final")

    assert selected.mode == "final"
    assert selected.model_path == model_path.resolve()
    assert selected.training_step == 6144
    assert selected.expected_final_step == 6144
    assert selected.selection_metric == "training_complete_root_model"
    assert selected.panel_id is None
    assert selected.panel_hash is None
    assert selected.normalization_path is None
