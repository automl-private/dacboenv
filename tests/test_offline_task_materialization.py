"""Generated CARP-S task configs retain global Hydra package semantics."""

from __future__ import annotations

from pathlib import Path

import pytest
from dacboenv.experiment import prepare_offline_carps_evaluation as preparation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def test_regenerated_yahpo_task_preserves_global_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regeneration keeps task fields global and leaves interpolations unresolved."""
    monkeypatch.setattr(preparation.carps, "__file__", str(tmp_path / "carps" / "__init__.py"))
    task_id = "yahpo/so/lcbench/fixture/None"
    task_config = OmegaConf.create({"task": {"name": task_id, "seed": "${seed}"}})
    monkeypatch.setattr(preparation, "get_task_config", lambda _task: task_config)
    root = tmp_path / "generated"
    destination = root / "task" / "YAHPO" / "SO" / "offline_lcbench_fixture.yaml"
    for _ in range(2):
        groups = preparation._task_configs({task_id}, root)
        assert groups["YAHPO/SO"] == ["offline_lcbench_fixture"]
        rendered = destination.read_text(encoding="utf-8")
        assert rendered.startswith("# @package _global_\n\n")
        assert "${seed}" in rendered
        with initialize_config_dir(config_dir=str(root), version_base=None):
            config = compose(overrides=["+task/YAHPO/SO=offline_lcbench_fixture", "+seed=7"])
        assert config.task.name == task_id
        assert config.task.seed == 7
        assert "YAHPO" not in config.task
