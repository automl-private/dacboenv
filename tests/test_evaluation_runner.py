"""Execution tests for the concrete unified evaluator adapter."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from dacboenv.env.reward import normalized_reference_regret_potential
from dacboenv.experiment import (
    evaluation_runner as evaluation_runner_module,
    unified_evaluator as unified_evaluator_module,
)
from dacboenv.experiment.default_smac import DefaultSMACResult
from dacboenv.experiment.evaluation_runner import make_dacbo_method_runner
from dacboenv.experiment.paired_evaluator import (
    BEST_VALIDATION_STATIC,
    DEFAULT_SMAC,
    LEARNED_VALIDATION_SELECTED,
    MARGINAL_RANDOM_CONTROL,
    MODAL_STATIC_CLONE,
    SAWEI,
    STATIC_ACTION_PREFIX,
    EvaluationContext,
    MethodRegistry,
    evaluate_registered_methods,
)
from dacboenv.experiment.protocol import manifest_hash
from dacboenv.experiment.real_env import real_structured_bbob_smoke_env
from dacboenv.experiment.unified_evaluator import (
    ProductionEvaluationUnavailableError,
    build_production_registry,
    inspect_stage_a_run,
)
from dacboenv.reference import ObjectiveReference
from dacboenv.utils import carps_optimizer as carps_optimizer_module
from gymnasium.spaces import Discrete
from omegaconf import DictConfig


@dataclass
class _TrialValue:
    cost: float


class _RunHistory:
    def __init__(self) -> None:
        self._data: OrderedDict[int, _TrialValue] = OrderedDict()


class _TinyPairedEnv:
    """Deterministic six-evaluation environment with two initial trials."""

    def __init__(self, task_id: str, seed: int) -> None:
        self.task_id = task_id
        self.seed = seed
        self.current_task_id = task_id
        self.current_seed = seed
        self._n_trials = 6
        self.interaction_frequency = 1
        self.action_space = Discrete(2)
        selector = SimpleNamespace(_initial_design_configs=[object(), object()])
        self._smac_instance = SimpleNamespace(
            runhistory=_RunHistory(),
            intensifier=SimpleNamespace(config_selector=selector),
        )
        self._objective_reference = ObjectiveReference(
            task_id=task_id,
            value=0.0,
            kind="exact",
            runtime_objective_transform="identity",
            reporting_objective_transform="identity",
            fidelity="not_applicable",
            source="tiny-test",
            source_hash="0" * 64,
            benchmark_code_version="test",
            benchmark_data_version="test",
            tolerance=0.0,
        )
        self.closed = False

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self._smac_instance.runhistory._data.clear()
        self._smac_instance.runhistory._data[0] = _TrialValue(10.0)
        self._smac_instance.runhistory._data[1] = _TrialValue(8.0)
        return np.asarray([0.0]), {"task_id": self.task_id, "inner_seed": self.seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        step = len(self._smac_instance.runhistory._data)
        previous = min(value.cost for value in self._smac_instance.runhistory._data.values())
        improvement = 1.0 if int(action) == step % 2 else 0.25
        current = previous - improvement
        self._smac_instance.runhistory._data[step] = _TrialValue(current)
        reward = normalized_reference_regret_potential(current, 0.0, 8.0) - normalized_reference_regret_potential(
            previous, 0.0, 8.0
        )
        return np.asarray([float(step)]), reward, step + 1 == self._n_trials, False, {}

    def close(self) -> None:
        self.closed = True


class _AlternatingPolicy:
    def __init__(self, offset: int) -> None:
        self.offset = offset
        self.step = 0

    def __call__(self, observation: Any) -> int:  # noqa: ARG002
        action = (self.step + self.offset) % 2
        self.step += 1
        return action

    def set_seed(self, seed: int | None) -> None:  # noqa: ARG002
        return None


def _manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "tiny-validation",
        "domain": "bbob",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "task_ids": ["bbob/2/3/0"],
        "inner_seeds": [11],
        "blockers": [],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def test_static_and_sawei_callbacks_share_one_paired_execution_path(tmp_path) -> None:
    """Tiny SAWEI-named smoke proves pairing/trace wiring, not real performance."""
    manifest = _manifest()
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id="bbob/2/3/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=6,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash=manifest["manifest_hash"],
    )
    registry = MethodRegistry(n_static_actions=1)
    envs: list[_TinyPairedEnv] = []

    def env_factory(task_id: str, seed: int) -> _TinyPairedEnv:
        env = _TinyPairedEnv(task_id, seed)
        envs.append(env)
        return env

    for method_name, offset in ((f"{STATIC_ACTION_PREFIX}0", 0), (SAWEI, 1)):
        registry.register_runner(
            method_name,
            make_dacbo_method_runner(
                env_factory=env_factory,
                policy_factory=lambda _env, _context, _method, offset=offset: _AlternatingPolicy(offset),
                action_family="wei",
                checkpoint_type="none",
                outer_ppo_seed=None,
                trace_directory=tmp_path / "traces",
                code_commit="deadbeef",
                policy_seed=7,
                policy_metadata={"smoke": "synthetic_execution_wiring"},
            ),
        )

    records = evaluate_registered_methods(
        manifest,
        [context],
        [f"{STATIC_ACTION_PREFIX}0", SAWEI],
        registry,
    )

    assert len(records) == 2
    assert {record.context_key for record in records} == {context.key}
    assert all(record.evaluation_budget == 6 for record in records)
    assert all(np.isfinite(record.anytime_auc) for record in records)
    assert all(env.closed for env in envs)
    trace_files = [path for path in (tmp_path / "traces").glob("*.json") if not path.name.endswith(".status.json")]
    assert len(trace_files) == 2
    assert len(list((tmp_path / "traces").glob("*.status.json"))) == 2


def test_production_registry_executes_static_random_and_validation_controls(tmp_path) -> None:
    manifest = _manifest()
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id="bbob/2/3/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=6,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash=manifest["manifest_hash"],
    )
    controls_path = tmp_path / "controls.json"
    controls_path.write_text(
        f"""{{
  "source_method": "learned_validation_selected",
  "source_split": "validation",
  "source_action_family": "wei",
  "source_checkpoint": "best",
  "source_outer_ppo_seed": 7,
  "source_validation_manifest_hash": "{manifest["manifest_hash"]}",
  "source_code_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_record_count": 1,
  "source_action_counts": [1, 3],
  "source_action_frequencies": [0.25, 0.75],
  "modal_action": 1
}}\n""",
        encoding="utf-8",
    )
    static_selection_path = tmp_path / "best-static.json"
    static_selection_path.write_text(
        f"""{{
  "source_split": "validation",
  "source_action_family": "wei",
  "source_validation_manifest_hash": "{manifest["manifest_hash"]}",
  "source_code_commit": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "source_metric": "normalized_final_regret",
  "source_static_scores": {{"0": 1.0, "1": 0.5}},
  "method": "best_validation_static",
  "selected_action": 0
}}\n""",
        encoding="utf-8",
    )
    methods = [
        f"{STATIC_ACTION_PREFIX}0",
        "uniform_random",
        MODAL_STATIC_CLONE,
        MARGINAL_RANDOM_CONTROL,
        BEST_VALIDATION_STATIC,
    ]
    registry = build_production_registry(
        methods,
        env_factory=_TinyPairedEnv,
        action_family="wei",
        trace_directory=tmp_path / "traces",
        control_provenance_path=controls_path,
        static_selection_provenance_path=static_selection_path,
        n_actions=2,
        policy_seed=13,
    )

    records = evaluate_registered_methods(manifest, [context], methods, registry)

    assert {record.method for record in records} == set(methods)
    assert len({record.context_key for record in records}) == 1
    assert next(record for record in records if record.method == MODAL_STATIC_CLONE).outer_ppo_seed == 7


def test_unified_runner_rejects_reference_mismatch_before_policy_execution(tmp_path) -> None:
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id="bbob/2/3/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=6,
        reference_kind="exact",
        reference_value=1.0,
        objective_transform="identity",
        manifest_hash=_manifest()["manifest_hash"],
    )
    policy_created = False

    def policy_factory(_env, _context, _method):
        nonlocal policy_created
        policy_created = True
        return _AlternatingPolicy(0)

    runner = make_dacbo_method_runner(
        env_factory=_TinyPairedEnv,
        policy_factory=policy_factory,
        action_family="wei",
        checkpoint_type="none",
        outer_ppo_seed=None,
        trace_directory=tmp_path / "traces",
    )

    with pytest.raises(RuntimeError, match="reference value differs"):
        runner(context, MethodRegistry(n_static_actions=1).method(f"{STATIC_ACTION_PREFIX}0"))

    assert not policy_created


def test_production_registry_rejects_unimplemented_method_before_environment_creation(tmp_path) -> None:
    calls = 0

    def env_factory(_task_id: str, _seed: int) -> _TinyPairedEnv:
        nonlocal calls
        calls += 1
        return _TinyPairedEnv(_task_id, _seed)

    with pytest.raises(ProductionEvaluationUnavailableError, match="sawei") as error:
        build_production_registry(
            [f"{STATIC_ACTION_PREFIX}0", SAWEI],
            env_factory=env_factory,
            action_family="wei",
            trace_directory=tmp_path / "traces",
            n_actions=2,
        )

    assert calls == 0
    assert "method-specific" in error.value.readiness["unavailable"][SAWEI]


def test_production_registry_registers_sawei_only_with_method_specific_factory(tmp_path) -> None:
    registry = build_production_registry(
        [SAWEI],
        env_factory=_TinyPairedEnv,
        sawei_env_factory=_TinyPairedEnv,
        action_family="wei",
        trace_directory=tmp_path / "traces",
        n_actions=2,
    )

    assert callable(registry.runner(SAWEI))


def test_production_registry_adapts_native_default_smac_without_dacbo_env(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    context = EvaluationContext(
        domain="bbob",
        scenario_or_function="3",
        dimension=2,
        task_id="bbob/2/3/0",
        native_instance="0",
        inner_seed=11,
        evaluation_budget=6,
        reference_kind="exact",
        reference_value=0.0,
        objective_transform="identity",
        manifest_hash=manifest["manifest_hash"],
    )
    monkeypatch.setattr(
        evaluation_runner_module,
        "run_default_smac_episode",
        lambda _task_id, _seed, **_kwargs: DefaultSMACResult(
            task_id="bbob/2/3/0",
            inner_seed=11,
            initial_incumbent=8.0,
            final_incumbent=4.0,
            reference_value=0.0,
            initial_regret=8.0,
            final_regret=4.0,
            normalized_final_regret=0.5,
            normalized_anytime_auc=0.75,
            telescoping_return=0.1,
            bo_evaluations=6,
            runtime_seconds=0.25,
            incumbent_trajectory=(10.0, 8.0, 7.0, 6.0, 5.0, 4.0),
        ),
    )
    env_calls = 0

    def unused_env_factory(_task_id: str, _seed: int) -> _TinyPairedEnv:
        nonlocal env_calls
        env_calls += 1
        return _TinyPairedEnv(_task_id, _seed)

    registry = build_production_registry(
        [DEFAULT_SMAC],
        env_factory=unused_env_factory,
        action_family="wei",
        trace_directory=tmp_path / "traces",
        n_actions=2,
    )
    records = evaluate_registered_methods(manifest, [context], [DEFAULT_SMAC], registry)

    assert env_calls == 0
    assert len(records) == 1
    assert records[0].method == DEFAULT_SMAC
    assert records[0].action_family == "native_smac"
    assert records[0].anytime_auc == pytest.approx(0.75)
    assert records[0].action_histogram == ()
    assert (tmp_path / "traces" / "default_smac__bbob__2__3__0__seed11.json").is_file()


def test_production_registry_preflights_yahpo_native_default_smac(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_id = "yahpo/so/lcbench/3945/None"
    manifest = {
        "schema_version": 1,
        "id": "tiny-yahpo-validation",
        "domain": "yahpo",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "task_ids": [task_id],
        "inner_seeds": [11],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)
    context = EvaluationContext(
        domain="yahpo",
        scenario_or_function="lcbench",
        dimension=None,
        task_id=task_id,
        native_instance="3945",
        inner_seed=11,
        evaluation_budget=6,
        reference_kind="best_known",
        reference_value=-97.0,
        objective_transform="negative_accuracy",
        manifest_hash=manifest["manifest_hash"],
    )
    reference = ObjectiveReference(
        task_id=task_id,
        value=-97.0,
        kind="best_known",
        runtime_objective_transform="negative_accuracy",
        reporting_objective_transform="one_minus_accuracy",
        fidelity="fixed_maximum",
        source="test",
        source_hash="0" * 64,
        benchmark_code_version="test",
        benchmark_data_version="test",
        tolerance=0.0,
    )
    monkeypatch.setattr(
        evaluation_runner_module,
        "run_default_smac_episode",
        lambda _task_id, _seed, **_kwargs: DefaultSMACResult(
            task_id=task_id,
            inner_seed=11,
            initial_incumbent=-95.0,
            final_incumbent=-96.0,
            reference_value=-97.0,
            initial_regret=2.0,
            final_regret=1.0,
            normalized_final_regret=0.5,
            normalized_anytime_auc=0.75,
            telescoping_return=0.1,
            bo_evaluations=6,
            runtime_seconds=0.25,
            incumbent_trajectory=(-94.0, -95.0, -95.2, -95.5, -95.8, -96.0),
        ),
    )
    registry = build_production_registry(
        [DEFAULT_SMAC],
        env_factory=_TinyPairedEnv,
        action_family="wei",
        trace_directory=tmp_path / "traces",
        default_smac_references={task_id: reference},
        n_actions=2,
    )

    records = evaluate_registered_methods(manifest, [context], [DEFAULT_SMAC], registry)

    assert records[0].domain == "yahpo"
    assert records[0].reference_kind == "best_known"


def test_stage_a_run_inventory_requires_complete_checkpoint_specific_normalization(tmp_path) -> None:
    run_root = tmp_path / "run"
    (run_root / ".hydra").mkdir(parents=True)
    (run_root / "validation").mkdir()
    (run_root / ".hydra" / "config.yaml").write_text(
        "seed: 4\naction_space_id: WEI-discrete-f1\n"
        "dacboenv:\n  interaction_frequency: 1\n"
        "experiment:\n  vecnormalize: true\n",
        encoding="utf-8",
    )
    (run_root / "model.zip").write_bytes(b"model")
    (run_root / "validation" / "best_model.zip").write_bytes(b"model")

    with pytest.raises(FileNotFoundError, match="best_balanced_vecnormalize"):
        inspect_stage_a_run(run_root)

    (run_root / "vecnormalize.pkl").write_bytes(b"norm")
    (run_root / "validation" / "best_balanced_vecnormalize.pkl").write_bytes(b"norm")
    artifacts = inspect_stage_a_run(run_root)

    assert artifacts.outer_seed == 4
    assert artifacts.vecnormalize is True
    assert artifacts.action_family == "wei"
    assert artifacts.interaction_frequency == 1

    with pytest.raises(ProductionEvaluationUnavailableError, match="action family differs"):
        build_production_registry(
            [LEARNED_VALIDATION_SELECTED],
            env_factory=_TinyPairedEnv,
            action_family="lcb_quantile",
            trace_directory=tmp_path / "mismatch-traces",
            run_root=run_root,
            n_actions=2,
        )
    with pytest.raises(ProductionEvaluationUnavailableError, match="interaction frequency differs"):
        build_production_registry(
            [LEARNED_VALIDATION_SELECTED],
            env_factory=_TinyPairedEnv,
            action_family="wei",
            interaction_frequency=5,
            trace_directory=tmp_path / "frequency-mismatch-traces",
            run_root=run_root,
            n_actions=2,
        )


def test_cli_environment_factory_propagates_manifest_context_split(monkeypatch) -> None:
    received: list[tuple[str, int, str, str, int]] = []

    def factory(
        task_id: str,
        seed: int,
        action_family: str,
        *,
        context_split: str,
        interaction_frequency: int,
    ) -> object:
        received.append((task_id, seed, action_family, context_split, interaction_frequency))
        return object()

    monkeypatch.setattr(
        unified_evaluator_module.importlib,
        "import_module",
        lambda _module: SimpleNamespace(factory=factory),
    )
    bound = unified_evaluator_module._load_environment_factory("fake:factory", "wei", "test")

    bound("bbob/2/3/0", 11, interaction_frequency=5)

    assert received == [("bbob/2/3/0", 11, "wei", "test", 5)]


def test_yahpo_preflight_requires_explicit_reference_table() -> None:
    context = EvaluationContext(
        domain="yahpo",
        scenario_or_function="lcbench",
        dimension=None,
        task_id="yahpo/so/lcbench/3945/None",
        native_instance="3945",
        inner_seed=11,
        evaluation_budget=126,
        reference_kind="best_known",
        reference_value=-97.0,
        objective_transform="negative_accuracy",
        manifest_hash="a" * 64,
    )

    with pytest.raises(ValueError, match="requires --reference-table"):
        unified_evaluator_module._preflight_yahpo_references(None, [context])


def test_real_structured_evaluator_factory_uses_exact_reference_reward(monkeypatch) -> None:
    native_get_task_config = carps_optimizer_module.get_task_config

    def tiny_task(task_id: str) -> DictConfig:
        cfg = native_get_task_config(task_id)
        cfg.task.optimization_resources.n_trials = 7
        return cfg

    monkeypatch.setattr(carps_optimizer_module, "get_task_config", tiny_task)
    env = real_structured_bbob_smoke_env("bbob/2/3/0", 193)
    try:
        observation, _info = env.reset()
        assert env.observation_space.contains(observation)
        assert env._objective_reference is not None
        assert env._objective_reference.kind == "exact"
        assert env._objective_reference.value == pytest.approx(20.91)
        _next_observation, reward, _terminated, _truncated, _step_info = env.step(0)
        assert np.isfinite(reward)
    finally:
        env.close()
