"""Focused protocol-v2 determinism and RNG-isolation tests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from dacboenv.experiment.evaluation_determinism import (
    EVALUATION_PROTOCOL_VERSION,
    PROCESS_DETERMINISM_CONTRACT,
    canonical_sha256,
    derive_policy_seed,
    require_process_determinism,
)
from dacboenv.experiment.evaluation_runner import run_dacbo_episode
from dacboenv.experiment.paired_evaluator import EvaluationContext, EvaluationMethod
from dacboenv.experiment.real_env import real_structured_bbob_env
from dacboenv.policy.static import StaticParameterPolicy


def _context(inner_seed: int = 17) -> EvaluationContext:
    return EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id="bbob/2/3/0",
        native_instance="0",
        inner_seed=inner_seed,
        evaluation_budget=77,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash="a" * 64,
        manifest_id="tiny-nontest-v1",
        evaluation_protocol_version=EVALUATION_PROTOCOL_VERSION,
    )


def test_context_hash_excludes_method_outer_seed_and_output_state() -> None:
    context = _context()
    assert context.context_hash == canonical_sha256(asdict(context))
    first = derive_policy_seed(123, "uniform_random", context.context_hash, outer_ppo_seed=1)
    replay = derive_policy_seed(123, "uniform_random", context.context_hash, outer_ppo_seed=1)
    other_policy = derive_policy_seed(123, "uniform_random", context.context_hash, outer_ppo_seed=2)
    assert first == replay
    assert other_policy != first
    assert _context(18).context_hash != context.context_hash


def test_process_contract_rejects_missing_hash_seed_before_work(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROCESS_DETERMINISM_CONTRACT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("PYTHONHASHSEED")
    with pytest.raises(RuntimeError, match="before starting Python"):
        require_process_determinism()


def test_process_contract_accepts_exact_exported_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in PROCESS_DETERMINISM_CONTRACT.items():
        monkeypatch.setenv(name, value)
    assert require_process_determinism() == PROCESS_DETERMINISM_CONTRACT


def test_fresh_process_context_hash_is_repeatable() -> None:
    code = "from dacboenv.experiment.evaluation_determinism import canonical_sha256; print(canonical_sha256({'x': 1}))"
    environment = {**os.environ, **PROCESS_DETERMINISM_CONTRACT}
    python = str(Path.cwd() / ".env/bin/python")
    first = subprocess.check_output([python, "-c", code], env=environment, text=True)  # noqa: S603
    second = subprocess.check_output([python, "-c", code], env=environment, text=True)  # noqa: S603
    assert first == second


def test_real_af_posterior_mean_constant_policy_matches_static_exactly() -> None:
    """Equivalent learned/static AF-selection action zero has one trajectory."""
    context = replace(_context(), reference_value=20.91)

    def factory(task_id: str, inner_seed: int, *, interaction_frequency: int = 1):
        return real_structured_bbob_env(
            task_id,
            inner_seed,
            "af_selection",
            context_split="validation",
            interaction_frequency=interaction_frequency,
        )

    def policy_factory(env, _context, _method):
        return StaticParameterPolicy(env, 0)

    traces = [
        run_dacbo_episode(
            context,
            EvaluationMethod(name, requires_trained_model=name.startswith("learned")),
            env_factory=factory,
            policy_factory=policy_factory,
            action_family="af_selection",
            checkpoint_type="smoke",
            outer_ppo_seed=outer_seed,
            code_commit="test-evaluation-revision",
            policy_seed=17,
        )
        for name, outer_seed in (("learned_constant_posterior_mean", 9), ("static_posterior_mean", None))
    ]
    fields = (
        "initial_design_hash",
        "first_model_based_candidate_hash",
        "evaluated_configuration_trajectory_hash",
        "incumbent_trajectory_hash",
        "action_trajectory_hash",
    )
    assert all(traces[0].fingerprints[field] == traces[1].fingerprints[field] for field in fields)
    assert traces[0].record.final_incumbent == traces[1].record.final_incumbent
