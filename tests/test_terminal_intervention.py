"""Bounded real engineering test of explicitly terminal intervention labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from dacboenv.experiment.collect_snapshots import completed_evaluations, observation_digest
from dacboenv.experiment.real_env import real_structured_bbob_smoke_env
from dacboenv.experiment.snapshot_branch import BOSnapshot
from dacboenv.selective_wei.initial_collection import write_verified
from dacboenv.selective_wei.initial_context import InitialBaseDecision
from dacboenv.selective_wei.terminal_intervention import terminal_interventions
from dacboenv.utils import carps_optimizer
from omegaconf import OmegaConf


def test_real_override_then_base_terminal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Five actions return to original a0, finish T and telescope exactly."""
    original = carps_optimizer.get_task_config

    def tiny_task(task_id: str) -> Any:
        config = OmegaConf.create(OmegaConf.to_container(original(task_id), resolve=False))
        config.task.optimization_resources.n_trials = 17
        return config

    monkeypatch.setattr(carps_optimizer, "get_task_config", tiny_task)

    def factory(task: str, seed: int) -> Any:
        return real_structured_bbob_smoke_env(task, seed, interaction_frequency=5)

    env = factory("bbob/2/3/0", 205)
    try:
        env.reset()
        context = env.get_initial_context({"probe_count": 8})
        env.freeze_initial_base(InitialBaseDecision(context.digest, 2, "engineering-base", 0))
        initial = env.get_incumbent_cost()
        observation, _, _, _, _ = env.step(2)
        reference = env.objective_reference
        assert reference is not None
        snapshot = BOSnapshot(
            "bbob/2/3/0",
            205,
            action_history=(2,),
            interaction_frequency=5,
            completed_evaluations=completed_evaluations(env),
            observation_hash=observation_digest(observation),
            initial_anchor_json=json.dumps(env.portable_initial_anchor()),
            reference_value=float(reference.value),
            initial_design_incumbent=initial,
        )
        result = terminal_interventions(snapshot, factory, horizon=5)
        assert result["target_semantics"] == "override_then_base_terminal_advantage"
        assert result["terminal_advantages"][2] == 0
        assert np.isfinite(result["terminal_advantages"]).all()
        for branch in result["branches"]:
            assert branch["action_suffix"] == [branch["action"], 2]
            assert len(branch["records"]) == 17
        write_verified(tmp_path / "terminal_intervention.json", result)
    finally:
        env.close()
