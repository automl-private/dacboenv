"""Focused contracts for the structured BO-MDP state and reward."""

from __future__ import annotations

from collections import Counter, OrderedDict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    Integer,
    OrdinalHyperparameter,
)
from sklearn.tree import _tree

# SMAC 2.3.1 still imports this public alias, which was removed by newer
# scikit-learn releases. Keep the project-local test environment importable
# without changing production code.
if not hasattr(_tree, "DTYPE"):
    _tree.DTYPE = np.float32

from carps.optimizers.smac20 import SMAC3Optimizer
from dacboenv.dacboenv import DACBOEnv
from dacboenv.env import observation as observation_module
from dacboenv.env.action import WEIDiscreteActionSpace, WEITempoRLActionSpace
from dacboenv.env.observation import (
    ACTION_FEATURE_DEFAULT,
    ACTION_FEATURE_INDEX,
    GLOBAL_STATE_DEFAULT,
    GLOBAL_STATE_INDEX,
    GLOBAL_STATE_NAMES,
    STRUCTURED_OBSERVATION_NAMES,
    WEI_ALPHA_LEVELS,
    ObservationSpace,
    _configuration_distance,
    _synchronize_model,
    calculate_action_features,
    calculate_global_state,
)
from dacboenv.env.reward import (
    LEGACY_REWARDS,
    DACBOReward,
    _initial_design_location_and_scale,
    calc_reference_free_improvement,
)
from dacboenv.features.signal.ubr import model_fitted
from dacboenv.optimizer import DACBOEnvOptimizer
from dacboenv.policy.noop import NoOpPolicy
from dacboenv.utils.reference_performance import ReferencePerformance
from dacboenv.utils.weighted_expected_improvement import WEI
from smac.main.config_selector import ConfigSelector
from smac.runhistory.enumerations import StatusType

from dacboenv import optimizer as optimizer_module


class FakeRunHistory:
    """Minimal insertion-ordered run history used by reward and state tests."""

    def __init__(self, costs: list[float], *, finished: int | None = None) -> None:
        self.set_costs(costs)
        self.finished = len(costs) if finished is None else finished

    def __len__(self) -> int:
        return len(self._data)

    def set_costs(self, costs: list[float]) -> None:
        """Replace the ordered trial values."""
        self._data = OrderedDict((trial, SimpleNamespace(cost=cost)) for trial, cost in enumerate(costs))


def make_fake_smbo(
    costs: list[float],
    *,
    n_initial: int,
    n_trials: int = 20,
    finished: int | None = None,
) -> SimpleNamespace:
    """Build only the SMBO attributes exercised by these contracts."""
    acquisition_model = SimpleNamespace(
        _kernel=SimpleNamespace(hyperparameters=[]),
    )
    selector = SimpleNamespace(
        _initial_design_configs=[object() for _ in range(n_initial)],
        _model=object(),
        _acquisition_function=SimpleNamespace(model=acquisition_model),
    )
    intensifier = SimpleNamespace(config_selector=selector, _config_selector=selector)
    return SimpleNamespace(
        runhistory=FakeRunHistory(costs, finished=finished),
        intensifier=intensifier,
        _intensifier=intensifier,
        _scenario=SimpleNamespace(n_trials=n_trials, seed=0),
    )


def potential(incumbent: float, location: float, scale: float) -> float:
    """Evaluate the reference-free reward potential."""
    return float(-np.arcsinh((incumbent - location) / scale))


def test_reference_free_reward_is_positive_affine_invariant() -> None:
    """Positive rescaling and translation must not change a transition reward."""
    costs = [10.0, 14.0, 8.0, 11.0, 6.0]
    original = calc_reference_free_improvement(make_fake_smbo(costs, n_initial=3))

    slope = 7.5
    intercept = -103.0
    transformed = [slope * cost + intercept for cost in costs]
    transformed_reward = calc_reference_free_improvement(
        make_fake_smbo(transformed, n_initial=3),
    )

    assert original > 0.0
    assert transformed_reward == pytest.approx(original)


def test_reference_free_reward_telescopes_over_an_episode() -> None:
    """The undiscounted rewards equal the final minus initial potential."""
    initial_costs = [10.0, 14.0, 8.0, 11.0]
    later_costs = [9.0, 7.0, np.nan, 7.5, 4.0, 5.0]
    all_costs = initial_costs + later_costs

    rewards = [
        calc_reference_free_improvement(
            make_fake_smbo(all_costs[:end], n_initial=len(initial_costs)),
        )
        for end in range(len(initial_costs) + 1, len(all_costs) + 1)
    ]

    location, scale = _initial_design_location_and_scale(np.asarray(initial_costs))
    initial_incumbent = min(initial_costs)
    final_incumbent = min(cost for cost in all_costs if np.isfinite(cost))
    expected_return = potential(final_incumbent, location, scale) - potential(
        initial_incumbent,
        location,
        scale,
    )

    assert all(reward >= 0.0 for reward in rewards)
    assert sum(rewards) == pytest.approx(expected_return)


def test_reference_free_reward_only_emits_for_a_new_incumbent() -> None:
    """Initial-design, failed, and non-improving trials must return zero."""
    assert calc_reference_free_improvement(
        make_fake_smbo([10.0, 8.0, 12.0], n_initial=3),
    ) == pytest.approx(0.0)
    assert calc_reference_free_improvement(
        make_fake_smbo([10.0, 8.0, 12.0, np.nan], n_initial=3),
    ) == pytest.approx(0.0)
    assert calc_reference_free_improvement(
        make_fake_smbo([10.0, 8.0, 12.0, 9.0], n_initial=3),
    ) == pytest.approx(0.0)


def test_reference_free_reward_ignores_failed_finite_crash_costs() -> None:
    """SMAC's finite crash penalty must not become the incumbent or baseline."""
    smbo = make_fake_smbo([], n_initial=3)
    smbo.runhistory._data = OrderedDict(
        [
            (0, SimpleNamespace(cost=-100.0, status=StatusType.CRASHED)),
            (1, SimpleNamespace(cost=10.0, status=StatusType.SUCCESS)),
            (2, SimpleNamespace(cost=12.0, status=StatusType.SUCCESS)),
            (3, SimpleNamespace(cost=9.0, status=StatusType.SUCCESS)),
        ],
    )
    smbo.runhistory.finished = 4

    assert calc_reference_free_improvement(smbo) > 0.0


def test_structured_observation_shapes_and_feature_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structured arrays expose the documented dimensions and column order."""
    smbo = make_fake_smbo([4.0, 3.0], n_initial=2, n_trials=20)
    expected_components = {
        "calculate_budget_density": 0.2,
        "calculate_log_effective_dimension": 0.3,
        "calculate_categorical_fraction": 0.4,
        "calculate_integer_ordinal_fraction": 0.5,
        "calculate_has_conditionals": 1.0,
        "calculate_normalized_noise": 0.6,
        "calculate_parameter_age": 0.1,
        "calculate_short_progress": 0.2,
        "calculate_long_progress": 0.3,
        "calculate_stagnation_age": 0.4,
        "calculate_calibration_error": 0.5,
    }
    for function_name, value in expected_components.items():
        monkeypatch.setattr(
            observation_module,
            function_name,
            lambda _smbo, _memory=None, value=value: value,
        )

    global_state = calculate_global_state(smbo, {"alpha": [0.75]})
    expected_global_state = np.asarray(
        [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 0.6, 0.75, 0.1, 0.2, 0.3, 0.4, 0.5],
        dtype=np.float32,
    )

    assert GLOBAL_STATE_NAMES == (
        "budget_percentage",
        "rho_B",
        "d_eff",
        "p_cat",
        "p_int",
        "has_conditionals",
        "normalized_noise",
        "previous_alpha",
        "a_age",
        "p_ts",
        "p_tl",
        "stagnation_age",
        "calibration_error",
    )
    np.testing.assert_allclose(global_state, expected_global_state)
    assert global_state.shape == (13,)
    assert global_state.dtype == np.float32

    monkeypatch.setattr(observation_module, "model_fitted", lambda _model: False)
    action_features = calculate_action_features(smbo)
    assert action_features.shape == (5, 4)
    assert action_features.dtype == np.float32
    np.testing.assert_array_equal(
        action_features[:, ACTION_FEATURE_INDEX["alpha"]],
        WEI_ALPHA_LEVELS,
    )
    np.testing.assert_array_equal(action_features, ACTION_FEATURE_DEFAULT)


def test_mixed_gower_distance_handles_all_supported_hp_types() -> None:
    """Novelty must normalize mixed, log-scaled, and inactive dimensions."""
    configspace = ConfigurationSpace()
    parent = Categorical("model", ["a", "b"])
    log_float = Float("rate", (1e-3, 1e3), log=True)
    integer = Integer("count", (1, 5))
    ordinal = OrdinalHyperparameter("quality", ["low", "middle", "high"])
    conditional = Float("conditional", (0.0, 1.0))
    configspace.add([parent, log_float, integer, ordinal, conditional])
    configspace.add(EqualsCondition(conditional, parent, "b"))

    lower = Configuration(
        configspace,
        values={
            "model": "a",
            "rate": 1e-3,
            "count": 1,
            "quality": "low",
        },
    )
    upper = Configuration(
        configspace,
        values={
            "model": "b",
            "rate": 1e3,
            "count": 5,
            "quality": "high",
            "conditional": 0.5,
        },
    )
    hyperparameters = list(configspace.values())

    assert _configuration_distance(lower, lower, hyperparameters) == pytest.approx(
        0.0,
    )
    assert _configuration_distance(lower, upper, hyperparameters) == pytest.approx(
        1.0,
    )


def test_observation_space_is_idempotent_per_finished_trial(  # noqa: PLR0915
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reads must neither advance memories nor expose cached arrays."""
    smbo = make_fake_smbo([4.0], n_initial=2, n_trials=20, finished=1)
    space = ObservationSpace(smbo, keys=["global_state", "action_features"])
    calls: Counter[str] = Counter()

    def calibration(_smbo: Any) -> float:
        calls["calibration"] += 1
        return 0.2

    def alpha(_smbo: Any) -> float:
        calls["alpha"] += 1
        return 0.25

    def synchronize(_smbo: Any) -> None:
        calls["synchronize"] += 1

    def global_state(fake_smbo: Any, memory: dict[str, list[float]]) -> np.ndarray:
        values = GLOBAL_STATE_DEFAULT.copy()
        values[GLOBAL_STATE_INDEX["budget_percentage"]] = fake_smbo.runhistory.finished / fake_smbo._scenario.n_trials
        values[GLOBAL_STATE_INDEX["previous_alpha"]] = memory["alpha"][-1]
        values[GLOBAL_STATE_INDEX["calibration_error"]] = memory["calibration"][-1]
        return values

    space._register_to_memory["calibration"] = calibration
    space._register_to_memory["alpha"] = alpha
    for observation_type in space._observation_types:
        if observation_type.name == "global_state":
            observation_type.compute = global_state
        elif observation_type.name == "action_features":
            observation_type.compute = lambda _smbo, _memory: ACTION_FEATURE_DEFAULT.copy()
    monkeypatch.setattr(observation_module, "_synchronize_model", synchronize)

    initial = space.get_observation()
    repeated_initial = space.get_observation()
    assert initial["global_state"].shape == (13,)
    assert initial["action_features"].shape == (5, 4)
    assert space.space.contains(initial)
    np.testing.assert_array_equal(initial["global_state"], repeated_initial["global_state"])
    assert calls == Counter()

    smbo.runhistory.finished = 2
    first = space.get_observation()
    second = space.get_observation()
    assert space.space.contains(first)
    np.testing.assert_array_equal(first["global_state"], second["global_state"])
    assert calls == Counter({"calibration": 1, "synchronize": 1, "alpha": 1})
    assert len(space._memory["calibration"]) == 1
    assert len(space._memory["alpha"]) == 1

    first["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] = 1.0
    defensive_copy = space.get_observation()
    assert defensive_copy["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] == pytest.approx(0.1)
    assert calls == Counter({"calibration": 1, "synchronize": 1, "alpha": 1})

    smbo.runhistory.finished = 3
    next_trial = space.get_observation()
    assert next_trial["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] == pytest.approx(0.15)
    assert calls == Counter({"calibration": 2, "synchronize": 2, "alpha": 2})

    space.reset()
    after_reset = space.get_initial_observation()
    repeated_after_reset = space.get_initial_observation()
    assert after_reset["global_state"][GLOBAL_STATE_INDEX["budget_percentage"]] == pytest.approx(0.15)
    np.testing.assert_array_equal(
        after_reset["global_state"],
        repeated_after_reset["global_state"],
    )
    assert calls == Counter({"calibration": 3, "synchronize": 3, "alpha": 3})


def test_model_fitted_accepts_mcmc_trained_state() -> None:
    """MCMC-like models use the shared ``_is_trained`` fitted-state contract."""
    assert model_fitted(SimpleNamespace(_is_trained=True))
    assert not model_fitted(SimpleNamespace(_is_trained=False))
    assert not model_fitted(None)


def test_structured_sync_preserves_batched_selector_semantics() -> None:
    """Observation reads must not force a batched selector to retrain early."""
    smbo = make_fake_smbo([4.0, 3.0], n_initial=2)
    selector = smbo.intensifier.config_selector
    selector._retrain_after = 2
    train_calls: list[tuple[np.ndarray, np.ndarray]] = []
    selector._model = SimpleNamespace(
        train=lambda X, Y: train_calls.append((X, Y)),
    )
    selector._collect_data = lambda: (
        np.ones((2, 1)),
        np.ones((2, 1)),
        [object(), object()],
    )

    _synchronize_model(smbo)

    assert train_calls == []


def test_structured_observations_are_explicit_opt_in() -> None:
    """The legacy default excludes all new state keys; the group has exactly two."""
    smbo = make_fake_smbo([], n_initial=2, finished=0)

    legacy_space = ObservationSpace(smbo, keys=None)
    assert STRUCTURED_OBSERVATION_NAMES.isdisjoint(legacy_space._keys)
    assert STRUCTURED_OBSERVATION_NAMES.isdisjoint(legacy_space.space.spaces)

    structured_space = ObservationSpace(
        smbo,
        keys=["global_state", "action_features"],
    )
    assert structured_space._keys == ["global_state", "action_features"]
    assert set(structured_space.space.spaces) == {"global_state", "action_features"}


def test_reward_manager_is_idempotent_per_finished_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated reward reads reuse the same transition result and computation."""
    smbo = make_fake_smbo([10.0, 12.0, 8.0, 6.0], n_initial=3)
    reward_manager = DACBOReward(smbo, keys=["reference_free_improvement"])
    reward_type = reward_manager._reward_types[0]
    original_compute = reward_type.compute
    calls = 0

    def counting_compute(fake_smbo: Any, reference: float | None) -> float:
        nonlocal calls
        calls += 1
        return float(original_compute(fake_smbo, reference))

    monkeypatch.setattr(reward_type, "compute", counting_compute)

    first = reward_manager.get_reward()
    repeated = reward_manager.get_reward()
    assert first > 0.0
    assert repeated == pytest.approx(first)
    assert calls == 1

    smbo.runhistory.set_costs([10.0, 12.0, 8.0, 6.0, 5.0])
    smbo.runhistory.finished = 5
    next_transition = reward_manager.get_reward()
    assert next_transition > 0.0
    assert calls == 2


def test_reference_free_reward_is_explicit_opt_in() -> None:
    """Adding the new reward must not change DACBOReward's legacy default."""
    reward = DACBOReward(make_fake_smbo([10.0, 8.0], n_initial=2))

    assert reward._keys == [reward_type.name for reward_type in LEGACY_REWARDS]
    assert "reference_free_improvement" not in reward._keys


def test_tempo_action_space_rejects_empty_axes() -> None:
    """Neither Tempo duration nor alpha axes may be empty."""
    with pytest.raises(ValueError, match="step_durations must not be empty"):
        WEITempoRLActionSpace(
            object(),
            step_durations=[],
            param_levels=WEI_ALPHA_LEVELS.tolist(),
        )

    with pytest.raises(ValueError, match="param_levels must not be empty"):
        WEITempoRLActionSpace(
            object(),
            step_durations=[1],
            param_levels=[],
        )
    with pytest.raises(TypeError, match="must be integers"):
        WEITempoRLActionSpace(
            object(),
            step_durations=[1.5],  # type: ignore[list-item]
            param_levels=[0.5],
        )
    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        WEITempoRLActionSpace(
            object(),
            step_durations=[1],
            param_levels=[1.5],
        )


def test_action_features_validate_tempo_alpha_grid() -> None:
    """Action-conditioned rows must map one-to-one to the discrete alpha axis."""
    env = DACBOEnv.__new__(DACBOEnv)
    env._dacbo_observation_space = SimpleNamespace(_keys=["action_features"])
    env._action_space = WEIDiscreteActionSpace(
        object(),
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )
    env._validate_action_feature_space()

    env._action_space = WEITempoRLActionSpace(
        object(),
        step_durations=[1],
        param_levels=[0.0, 0.5, 1.0],
    )
    with pytest.raises(ValueError, match="require WEI alpha levels"):
        env._validate_action_feature_space()

    env._action_space = object()
    with pytest.raises(ValueError, match="requires WEIDiscreteActionSpace"):
        env._validate_action_feature_space()


def test_exact_discrete_action_maps_indices_to_alpha_levels() -> None:
    """The five categorical actions must address the exact configured alphas."""
    selector = object.__new__(ConfigSelector)
    selector._acquisition_function = SimpleNamespace(_alpha=None)
    smbo = SimpleNamespace(
        _intensifier=SimpleNamespace(_config_selector=selector),
    )
    action_space = WEIDiscreteActionSpace(
        smbo,
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )

    assert action_space.space.n == 5
    for action_idx, alpha in enumerate(WEI_ALPHA_LEVELS):
        action_space.update_optimizer(action_idx)
        assert selector._acquisition_function._alpha == pytest.approx(alpha)

    with pytest.raises(ValueError, match="outside"):
        action_space.update_optimizer(5)


def test_fixed_interaction_frequency_sums_rewards_and_stops_at_terminal() -> None:
    """A discrete alpha is applied once and held over the configured BO steps."""
    env = DACBOEnv.__new__(DACBOEnv)
    env._action_space = WEIDiscreteActionSpace(
        object(),
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )
    env._interaction_frequency = 5
    env._episode_reward = 0.0
    env._episode_length = 0
    substep_results = iter(
        [
            ({"trial": np.asarray([1])}, 0.5, False, False, {}),
            ({"trial": np.asarray([2])}, 1.5, True, False, {}),
        ],
    )
    updates: list[int] = []
    env.update_optimizer = lambda action: updates.append(int(action))
    env._step = lambda *, action: next(substep_results)  # noqa: ARG005
    env.get_n_finished_trials = lambda: 2

    observation, reward, terminated, truncated, info = env.step(3)

    assert updates == [3]
    assert reward == pytest.approx(2.0)
    assert terminated
    assert not truncated
    np.testing.assert_array_equal(observation["trial"], np.asarray([2]))
    assert info == {
        "bo_evaluations": 2,
        "policy_decisions": 1,
        "episode": {"r": 2.0, "l": 1},
    }


def test_bo_budget_exhaustion_is_terminal_not_a_timeout() -> None:
    """SB3 must not bootstrap beyond the finite BO budget."""
    env = DACBOEnv.__new__(DACBOEnv)
    runhistory = SimpleNamespace(finished=0)
    smbo = SimpleNamespace(
        ask=object,
        tell=lambda _trial_info, _trial_value: setattr(runhistory, "finished", 1),
        _runner=SimpleNamespace(
            run_wrapper=lambda trial_info: (trial_info, SimpleNamespace(cost=1.0)),
        ),
        _scenario=SimpleNamespace(n_trials=1),
        runhistory=runhistory,
    )
    env._smac_instance = smbo
    env._requires_reference_performance = False
    env._terminate_after_reference_performance_reached = False
    env._evaluation_mode = False
    env._episode_reward = 0.0
    env._episode_length = 0
    env._action_space = object()
    env._interaction_frequency = 1
    env._instance = (0, "bbob/2/1/0")
    env.current_threshold = None
    env.update_optimizer = lambda _action: None
    env.get_incumbent_cost = lambda: 1.0
    env.get_observation = dict
    env.get_reward = lambda: 0.25
    env.get_n_finished_trials = lambda: runhistory.finished

    _obs, reward, terminated, truncated, info = env.step(action=0)

    assert reward == pytest.approx(0.25)
    assert terminated
    assert not truncated
    assert info == {
        "bo_evaluations": 1,
        "policy_decisions": 1,
        "episode": {"r": 0.25, "l": 1},
    }


def test_interaction_frequency_validation() -> None:
    """Only positive integer fixed frequencies are accepted."""
    base_kwargs = {
        "task_ids": ["bbob/2/1/0"],
        "inner_seeds": [0],
    }
    with pytest.raises(ValueError, match="must be > 0"):
        DACBOEnv(**base_kwargs, interaction_frequency=0)
    with pytest.raises(TypeError, match="positive integer"):
        DACBOEnv(**base_kwargs, interaction_frequency=1.5)


def test_prepared_reset_is_reused_once_without_aliasing() -> None:
    """CARPS space discovery must not make SB3 repeat the initial design."""
    env = DACBOEnv.__new__(DACBOEnv)
    prepared_observation = {"state": np.asarray([1.0], dtype=np.float32)}
    env._prepared_reset_result = (prepared_observation, {"prepared": True}, None)

    observation, info = env.reset()

    assert info == {"prepared": True}
    np.testing.assert_array_equal(observation["state"], prepared_observation["state"])
    assert observation["state"] is not prepared_observation["state"]
    assert env._prepared_reset_result is None


def test_tempo_step_sums_rewards_and_stops_at_terminal() -> None:
    """A Tempo action returns its accumulated reward and stops its hold early."""
    env = DACBOEnv.__new__(DACBOEnv)
    env._action_space = WEITempoRLActionSpace(
        object(),
        step_durations=[1, 5],
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )
    env._episode_reward = 0.0
    env._episode_length = 0
    action = np.asarray([1, 3], dtype=np.int64)
    substep_results = iter(
        [
            ({"trial": np.asarray([1])}, 1.25, False, False, {"substep": 1}),
            ({"trial": np.asarray([2])}, 2.5, True, False, {"substep": 2}),
        ],
    )
    applied_actions: list[np.ndarray] = []
    policy_updates: list[np.ndarray] = []

    def fake_substep(*, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        applied_actions.append(action.copy())
        return next(substep_results)

    env.update_optimizer = lambda selected_action: policy_updates.append(
        selected_action.copy(),
    )
    env._step = fake_substep
    env.get_n_finished_trials = lambda: 2
    observation, reward, terminated, truncated, info = env.step(action)

    assert reward == pytest.approx(3.75)
    assert terminated
    assert not truncated
    assert info == {
        "substep": 2,
        "bo_evaluations": 2,
        "policy_decisions": 1,
        "episode": {"r": 3.75, "l": 1},
    }
    np.testing.assert_array_equal(observation["trial"], np.asarray([2]))
    assert len(policy_updates) == 1
    np.testing.assert_array_equal(policy_updates[0], action)
    assert len(applied_actions) == 2
    for applied_action in applied_actions:
        np.testing.assert_array_equal(applied_action, action)


def test_external_tempo_optimizer_updates_once_per_policy_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Held Tempo actions must not be reapplied on every external ask."""
    optimizer = object.__new__(DACBOEnvOptimizer)
    optimizer._solver = SimpleNamespace(
        runhistory=[object()],
        intensifier=SimpleNamespace(
            config_selector=SimpleNamespace(_initial_design_configs=[object()]),
        ),
        optimizer=SimpleNamespace(_callbacks=[]),
    )
    action_space = WEITempoRLActionSpace(
        object(),
        step_durations=[1, 3],
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )
    selected_action = np.asarray([1, 3], dtype=np.int64)
    policy_calls: list[Any] = []
    updates: list[np.ndarray] = []
    optimizer._policy = lambda state: policy_calls.append(state) or selected_action.copy()
    optimizer._state = object()
    optimizer._dacboenv = SimpleNamespace(
        _action_space=action_space,
        update_optimizer=lambda action: updates.append(action.copy()),
    )
    optimizer._skip_duration = 0
    optimizer._actionfile = "unused.jsonl"

    sentinel = object()
    monkeypatch.setattr(SMAC3Optimizer, "ask", lambda _self: sentinel)
    monkeypatch.setattr(optimizer_module, "dump_logs", lambda *_args: None)

    assert [optimizer.ask() for _ in range(3)] == [sentinel] * 3
    assert len(policy_calls) == 1
    assert len(updates) == 1
    np.testing.assert_array_equal(updates[0], selected_action)

    assert optimizer.ask() is sentinel
    assert len(policy_calls) == 2
    assert len(updates) == 2


def test_external_discrete_optimizer_accepts_noop_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NoOpPolicy's None action must leave SMAC unchanged and remain loggable."""
    optimizer = object.__new__(DACBOEnvOptimizer)
    optimizer._solver = SimpleNamespace(
        runhistory=[object()],
        intensifier=SimpleNamespace(
            config_selector=SimpleNamespace(_initial_design_configs=[object()]),
        ),
        optimizer=SimpleNamespace(_callbacks=[]),
    )
    acquisition_function = WEI()
    action_space = WEIDiscreteActionSpace(
        SimpleNamespace(
            _intensifier=SimpleNamespace(
                _config_selector=SimpleNamespace(
                    _acquisition_function=acquisition_function,
                ),
            ),
        ),
        param_levels=WEI_ALPHA_LEVELS.tolist(),
    )
    updates: list[Any] = []
    optimizer._state = object()
    optimizer._dacboenv = SimpleNamespace(
        _action_space=action_space,
        interaction_frequency=1,
        update_optimizer=updates.append,
    )
    optimizer._policy = NoOpPolicy(env=optimizer._dacboenv)
    optimizer._skip_duration = 0
    optimizer._actionfile = "unused.jsonl"

    sentinel = object()
    action_logs: list[dict[str, Any]] = []
    monkeypatch.setattr(SMAC3Optimizer, "ask", lambda _self: sentinel)
    monkeypatch.setattr(
        optimizer_module,
        "dump_logs",
        lambda logs, _path: action_logs.append(logs),
    )

    assert optimizer.ask() is sentinel
    assert updates == []
    assert acquisition_function._alpha == pytest.approx(0.5)
    assert action_logs == [
        {
            "action_type": action_space._action.name,
            "n_trials": 1,
            "action": None,
        },
    ]


def test_random_seed_reference_query_uses_stable_mean() -> None:
    """Random-mode episodes compare against the mean over reference seeds."""
    reference = object.__new__(ReferencePerformance)
    reference.perf_df = pd.DataFrame(
        {
            "optimizer_id": ["ref", "ref", "ref", "other"],
            "task_id": ["task", "task", "other-task", "task"],
            "seed": [1, 2, 1, 1],
            "trial_value__cost_inc": [2.0, 4.0, 100.0, -100.0],
        },
    )

    assert reference.query_cost("ref", "task", seed=None) == pytest.approx(3.0)
    assert reference.query_cost("ref", "task", seed=2) == pytest.approx(4.0)
