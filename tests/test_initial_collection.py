"""Bounded five-arm CARP-S continuation pairing tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from dacboenv.experiment.real_env import real_structured_bbob_smoke_env, real_structured_yahpo_env
from dacboenv.selective_wei.initial_collection import collect_static_context, read_verified
from dacboenv.selective_wei.initial_features import InitialFeatureSettings
from dacboenv.utils import carps_optimizer
from omegaconf import OmegaConf


@pytest.mark.parametrize("task_id", ["bbob/2/3/0", "yahpo/so/lcbench/167200/None", "yahpo/so/rbv2_xgboost/41216/None"])
def test_real_five_arm_pairing_and_reverse_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task_id: str) -> None:
    """All static arms share z0; reversed order and complete reuse are invariant."""
    original = carps_optimizer.get_task_config

    def tiny_task(task_id: str):
        cfg = OmegaConf.create(OmegaConf.to_container(original(task_id), resolve=False))
        cfg.task.optimization_resources.n_trials = 7
        return cfg

    monkeypatch.setattr(carps_optimizer, "get_task_config", tiny_task)

    def factory():
        if task_id.startswith("bbob/"):
            return real_structured_bbob_smoke_env(task_id, 199, interaction_frequency=5)
        return real_structured_yahpo_env(
            task_id,
            199,
            initial_design_n_configs=2,
            context_split="train",
            interaction_frequency=5,
            reference_table="dacboenv/experiment/analysis/yahpo_best_known_references.json",
        )

    recipe = {"split": "train", "task": task_id, "seed": 199, "engineering_budget": 7}
    settings = InitialFeatureSettings(probe_count=8)
    first = collect_static_context(factory, tmp_path / "forward", recipe=recipe, settings=settings)
    reverse = collect_static_context(
        factory, tmp_path / "reverse", recipe=recipe, settings=settings, arm_order=(4, 3, 2, 1, 0)
    )
    assert first["context_digest"] == reverse["context_digest"]
    assert first["targets"] == reverse["targets"]
    reused = collect_static_context(factory, tmp_path / "forward", recipe=recipe, settings=settings)
    assert reused == first
    assert len(read_verified(tmp_path / "forward" / "arm_0.json")["records"]) == 7
