"""Contracts for loading CARPS optimizer configurations."""

from __future__ import annotations

from pathlib import Path

from dacboenv.utils.carps_optimizer import get_task_config, load_optimizer_config


def test_load_optimizer_config_accepts_documented_yaml_path(tmp_path: Path) -> None:
    """A direct YAML filename must not require the generated CARPS index."""
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text("optimizer:\n  name: direct\n", encoding="utf-8")

    config = load_optimizer_config(str(config_path))

    assert config.optimizer.name == "direct"


def test_optimizer_base_defaults_are_overridden_by_child(tmp_path: Path) -> None:
    """Hydra-style child values must take precedence over base defaults."""
    (tmp_path / "base.yaml").write_text(
        "optimizer:\n  inherited: true\n  learning_rate: 0.1\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "optimizer.yaml"
    config_path.write_text(
        "defaults:\n  - base\noptimizer:\n  learning_rate: 0.01\n",
        encoding="utf-8",
    )

    config = load_optimizer_config(str(config_path))

    assert config.optimizer.inherited
    assert config.optimizer.learning_rate == 0.01
    assert "defaults" not in config


def test_carps_1_1_optimizer_lookup_without_packaged_index() -> None:
    """Named optimizers remain usable when CARPS only ships YAML configs."""
    config = load_optimizer_config("SMAC3-BlackBoxFacade")

    assert config.optimizer.smac_cfg.smac_class == "smac.facade.blackbox_facade.BlackBoxFacade"


def test_carps_1_1_non_bbob_task_lookup_without_packaged_index() -> None:
    """The legacy non-BBOB task interface does not require a writable cache."""
    config = get_task_config("dummy")

    assert config.task.name == "dummy"
