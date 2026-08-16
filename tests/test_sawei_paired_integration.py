"""Tiny real BBOB/SMAC pairing smoke for native SAWEI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from dacboenv.experiment.evaluation_runner import make_dacbo_method_runner
from dacboenv.experiment.paired_evaluator import (
    SAWEI,
    STATIC_ACTION_PREFIX,
    EvaluationContext,
    MethodRegistry,
    evaluate_registered_methods,
)
from dacboenv.experiment.protocol import manifest_hash
from dacboenv.experiment.real_env import real_sawei_bbob_env, real_sawei_env, real_structured_bbob_smoke_env
from dacboenv.policy.sawei import SAWEIPolicy
from dacboenv.policy.static import StaticParameterPolicy
from dacboenv.utils import carps_optimizer as carps_optimizer_module
from omegaconf import DictConfig


def test_real_sawei_and_static_share_exact_reduced_bbob_context(tmp_path, monkeypatch) -> None:
    native_get_task_config = carps_optimizer_module.get_task_config

    def tiny_task(task_id: str) -> DictConfig:
        cfg = native_get_task_config(task_id)
        cfg.task.optimization_resources.n_trials = 7
        return cfg

    monkeypatch.setattr(carps_optimizer_module, "get_task_config", tiny_task)
    task_id = "bbob/2/3/0"
    inner_seed = 919
    manifest = {
        "schema_version": 1,
        "id": "paired-sawei-real-smoke",
        "domain": "bbob",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "task_ids": [task_id],
        "inner_seeds": [inner_seed],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id=task_id,
        native_instance="0",
        inner_seed=inner_seed,
        evaluation_budget=7,
        reference_kind="exact",
        reference_value=20.91,
        objective_transform="identity",
        manifest_hash=manifest["manifest_hash"],
    )
    registry = MethodRegistry(n_static_actions=5)
    static_method = f"{STATIC_ACTION_PREFIX}2"
    method_specs = {
        static_method: (
            lambda env, _context, _method: StaticParameterPolicy(env, 2),
            real_structured_bbob_smoke_env,
            "wei",
        ),
        SAWEI: (
            lambda env, _context, _method: SAWEIPolicy(env, window_size=2),
            lambda selected_task, seed: real_sawei_bbob_env(
                selected_task,
                seed,
                output_directory=tmp_path / SAWEI,
                initial_design_n_configs=2,
            ),
            "wei_continuous",
        ),
    }
    for method_name, (policy_factory, env_factory, action_family) in method_specs.items():
        registry.register_runner(
            method_name,
            make_dacbo_method_runner(
                env_factory=env_factory,
                policy_factory=policy_factory,
                action_family=action_family,
                checkpoint_type="none",
                outer_ppo_seed=None,
                trace_directory=tmp_path / "traces",
                code_commit="real-smoke",
                policy_seed=17,
                policy_metadata={"sawei_delta": 0.1, "sawei_window_size": 2},
            ),
        )

    records = evaluate_registered_methods(
        manifest,
        [context],
        [static_method, SAWEI],
        registry,
    )

    assert len(records) == 2
    assert all(record.evaluation_budget == 7 for record in records)
    assert all(record.context_key == context.key for record in records)
    assert all(0.0 <= record.episode_return <= 1.0 for record in records)
    assert {record.method for record in records} == {static_method, SAWEI}
    traces = [path for path in (tmp_path / "traces").glob("*.json") if not path.name.endswith(".status.json")]
    assert len(traces) == 2
    assert len(list((tmp_path / "traces").glob("*.status.json"))) == 2
    trace_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in traces]
    assert len({tuple(trace["incumbent_trajectory"][:2]) for trace in trace_payloads}) == 1
    static_record = next(record for record in records if record.method == static_method)
    sawei_record = next(record for record in records if record.method == SAWEI)
    assert len(static_record.action_histogram) == 5
    assert sawei_record.action_histogram == ()


def test_real_yahpo_sawei_factory_uses_non_test_best_known_reward() -> None:
    env = real_sawei_env(
        "yahpo/so/lcbench/3945/None",
        1943,
        initial_design_n_configs=2,
        reference_table=Path("dacboenv/experiment/analysis/yahpo_best_known_references.json").resolve(),
        allow_incomplete_reference=True,
    )
    try:
        observation, info = env.reset()
        assert env._objective_reference is not None
        assert env._objective_reference.kind == "best_known"
        assert env.observation_space.contains(observation)
        assert all("reference" not in key.lower() for key in info)
        policy = SAWEIPolicy(env, window_size=2)
        action = policy(observation)
        next_observation, reward, _terminated, _truncated, step_info = env.step(action)
        assert env.observation_space.contains(next_observation)
        assert np.isfinite(float(reward))
        assert all("reference" not in key.lower() for key in step_info)
    finally:
        env.close()
