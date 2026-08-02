"""Gymnasium reset-seeding contracts for DACBOEnv."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import dacboenv.dacboenv as dacboenv_module
import numpy as np
import pytest
from dacboenv.dacboenv import DACBOEnv
from dacboenv.env.instance import RandomInstanceSelector
from dacboenv.task import DACBOObjectiveFunction
from gymnasium.spaces import Box, Dict, Discrete
from gymnasium.utils.env_checker import check_reset_seed_determinism


class ProbeActionSpace:
    """Minimal action wrapper needed by ``DACBOEnv.reset``."""

    def __init__(self, smac_instance: object, **kwargs: Any) -> None:  # noqa: ARG002
        self.space = Discrete(2)

    def update_optimizer(self, action: object) -> None:
        """Satisfy the DACBO action-space protocol."""


class ProbeObservationSpace:
    """Expose the selected inner seed as a valid observation."""

    def __init__(
        self,
        smac_instance: object,
        keys: list[str] | None,  # noqa: ARG002
        action_space: object | None = None,  # noqa: ARG002
    ) -> None:
        self._smac_instance = smac_instance
        self._keys: list[str] = []
        self._observation_space: dict[str, object] = {}
        self.space = Dict(
            {
                "inner_seed": Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float64,
                ),
            },
        )

    def reset(self) -> None:
        """Satisfy the DACBO observation-space protocol."""

    def get_initial_observation(self) -> dict[str, np.ndarray]:
        """Return an observation tied to the selected BO seed."""
        return {
            "inner_seed": np.asarray(
                [self._smac_instance._scenario.seed],
                dtype=np.float64,
            ),
        }


@pytest.fixture
def probe_env_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]]:
    """Replace CARPS/SMAC with a cheap recorder for reset tests."""
    build_calls: list[tuple[str, int]] = []

    def build_probe_optimizer(
        *,
        optimizer_id: str | None,  # noqa: ARG001
        task_id: str,
        seed: int,
        optimizer_cfg: object,  # noqa: ARG001
    ) -> SimpleNamespace:
        build_calls.append((task_id, seed))
        config_selector = SimpleNamespace(_initial_design_configs=[])
        smbo = SimpleNamespace(
            _scenario=SimpleNamespace(
                seed=seed,
                n_trials=1,
                count_objectives=lambda: 1,
            ),
            intensifier=SimpleNamespace(config_selector=config_selector),
        )
        return SimpleNamespace(solver=SimpleNamespace(optimizer=smbo))

    monkeypatch.setattr(
        dacboenv_module,
        "build_carps_optimizer",
        build_probe_optimizer,
    )
    monkeypatch.setattr(
        dacboenv_module,
        "ObservationSpace",
        ProbeObservationSpace,
    )
    monkeypatch.setattr(
        dacboenv_module,
        "DACBOReward",
        lambda *_args, **_kwargs: object(),
    )

    def make_probe_env(**kwargs: Any) -> tuple[DACBOEnv, list[tuple[str, int]]]:
        build_calls.clear()
        action_space_class = kwargs.pop("action_space_class", ProbeActionSpace)
        env = DACBOEnv(
            action_space_class=action_space_class,
            observation_keys=[],
            evaluation_mode=True,
            **kwargs,
        )
        return env, build_calls

    return make_probe_env


def test_reset_seed_restarts_outer_selection_without_replacing_inner_seeds(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """Equal Gym seeds reproduce contexts while fixed BO seeds stay fixed."""
    env, build_calls = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=[7, 11],
        instance_selector_class=RandomInstanceSelector,
        seed=999,
    )

    env.reset(seed=123)
    first_sequence = [env.instance]
    env.reset()
    first_sequence.append(env.instance)

    env.reset(seed=123)
    repeated_sequence = [env.instance]
    env.reset()
    repeated_sequence.append(env.instance)

    assert repeated_sequence == first_sequence
    assert env.np_random_seed == 123
    assert env.instance_set.seeds == [7, 11]
    assert all(inner_seed in {7, 11} for _, inner_seed in build_calls)
    assert all(
        inner_seed == instance[0]
        for (_, inner_seed), instance in zip(
            build_calls,
            first_sequence * 2,
            strict=True,
        )
    )


def test_reset_passes_gymnasium_seed_determinism_check(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """The Gym RNG is seeded by the outer seed rather than the BO seed."""
    env, _ = probe_env_factory(
        task_ids=["task"],
        inner_seeds=[7],
        seed=0,
    )

    check_reset_seed_determinism(env)
    assert env.current_seed == 7
    assert env.np_random_seed == 456


def test_outer_seed_reproduces_generated_fallback_inner_seeds(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """Unspecified BO seeds are derived reproducibly from the Gym seed."""
    first_env, _ = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=None,
        instance_selector_class=RandomInstanceSelector,
        seed=None,
    )
    first_env.reset(seed=41)
    first_instance = first_env.instance
    first_fallback_seeds = first_env.instance_set.seeds.copy()

    second_env, _ = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=None,
        instance_selector_class=RandomInstanceSelector,
        seed=None,
    )
    second_env.reset(seed=41)

    assert second_env.instance_set.seeds == first_fallback_seeds
    assert second_env.instance == first_instance


def test_dynamic_inner_seed_stream_is_fresh_and_replayable(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """The ``None`` marker draws one reproducible fresh BO seed per episode."""
    env, build_calls = probe_env_factory(
        task_ids=["task-a", "task-b", "task-c"],
        inner_seeds=[None],
        instance_selector_class=RandomInstanceSelector,
        seed=999,
    )

    env.reset(seed=123)
    first_sequence = [env.instance]
    for _ in range(7):
        env.reset()
        first_sequence.append(env.instance)
    first_build_calls = build_calls.copy()

    env.reset(seed=123)
    repeated_sequence = [env.instance]
    for _ in range(7):
        env.reset()
        repeated_sequence.append(env.instance)
    repeated_build_calls = build_calls[len(first_build_calls) :]

    assert repeated_sequence == first_sequence
    assert repeated_build_calls == first_build_calls
    assert len({inner_seed for inner_seed, _task_id in first_sequence}) == len(first_sequence)
    assert env.instance_set.seeds == [None]
    assert first_build_calls == [(task_id, inner_seed) for inner_seed, task_id in first_sequence]
    assert all(inner_seed is not None for inner_seed, _task_id in first_sequence)


def test_dynamic_inner_seed_stream_is_independent_of_instance_selector(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """Task-selection draws cannot perturb the per-episode BO-seed stream."""

    def collect_inner_seeds(env: DACBOEnv) -> list[int]:
        env.reset(seed=73)
        seeds = [env.current_seed]
        for _ in range(7):
            env.reset()
            seeds.append(env.current_seed)
        return seeds

    round_robin_env, _ = probe_env_factory(
        task_ids=["task-a", "task-b", "task-c"],
        inner_seeds=[None],
        seed=999,
    )
    round_robin_seeds = collect_inner_seeds(round_robin_env)

    random_env, _ = probe_env_factory(
        task_ids=["task-a", "task-b", "task-c"],
        inner_seeds=[None],
        instance_selector_class=RandomInstanceSelector,
        seed=999,
    )
    random_seeds = collect_inner_seeds(random_env)

    assert random_seeds == round_robin_seeds


def test_action_space_choice_cannot_change_training_context_trace(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """All controller families receive the same task/inner-seed trace."""

    def collect(action_space_class: type[ProbeActionSpace]) -> list[tuple[int, str]]:
        env, _ = probe_env_factory(
            task_ids=["task-a", "task-b", "task-c"],
            inner_seeds=[None],
            instance_selector_class=RandomInstanceSelector,
            action_space_class=action_space_class,
            seed=999,
        )
        env.reset(seed=73)
        trace = [env.instance]
        for _ in range(11):
            env.reset()
            trace.append(env.instance)
        env.close()
        return trace

    action_spaces = {
        name: type(f"Probe{name.title()}ActionSpace", (ProbeActionSpace,), {})
        for name in ("wei", "lcb", "ucb", "af_selection")
    }
    traces = {name: collect(action_space_class) for name, action_space_class in action_spaces.items()}

    assert traces["wei"] == traces["lcb"] == traces["ucb"] == traces["af_selection"]
    assert len({inner_seed for inner_seed, _task_id in traces["wei"]}) == len(traces["wei"])


@pytest.mark.parametrize("inner_seeds", [[None, 7], [None, None]])
def test_dynamic_inner_seed_marker_must_be_a_singleton(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
    inner_seeds: list[int | None],
) -> None:
    """Mixed or repeated null markers have ambiguous episode semantics."""
    with pytest.raises(ValueError, match=r"exactly \[None\]"):
        probe_env_factory(
            task_ids=["task"],
            inner_seeds=inner_seeds,
            seed=1,
        )


def test_dynamic_inner_seed_marker_bypasses_fixed_seed_mapping() -> None:
    """CARPS external seeds cannot exhaust a streaming inner-seed marker."""
    objective = SimpleNamespace(_internal_seeds=[None], _seed_map={})

    assert DACBOObjectiveFunction._get_internal_seed(objective, 3) is None
    assert DACBOObjectiveFunction._get_internal_seed(objective, 7) is None
    assert objective._seed_map == {}


def test_dynamic_inner_seed_is_reported_after_carps_rollout() -> None:
    """CARPS trial metadata records the resolved seed rather than the marker."""
    objective = SimpleNamespace(
        _env=SimpleNamespace(current_seed=12345),
        _internal_seeds=[None],
        target_function=lambda **_kwargs: (1.0, {}),
    )
    trial_info = SimpleNamespace(
        config=object(),
        budget=None,
        instance=None,
        seed=7,
        cutoff=None,
    )

    trial_value = DACBOObjectiveFunction._evaluate(objective, trial_info)

    assert trial_value.additional_info["internal_seed"] == 12345


def test_outer_seed_does_not_replace_an_explicit_instance_set(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """A later fixed BO context remains independent of the Gym seed."""
    env, _ = probe_env_factory(
        task_ids=["task"],
        inner_seeds=None,
        seed=None,
    )
    env.instance_set = ([101], ["task"])

    env.reset(seed=41)

    assert env.instance_set.seeds == [101]
    assert env.current_seed == 101


def test_matching_seed_reuses_prepared_episode_without_rewinding_selector(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """SB3's first seeded reset consumes the prepared episode exactly once."""
    env, build_calls = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=[7],
        seed=19,
    )

    env.prepare_for_first_reset()
    prepared_observation = env._prepared_reset_result
    observation, _ = env.reset(seed=19)

    assert len(build_calls) == 1
    assert prepared_observation is not None
    np.testing.assert_array_equal(
        observation["inner_seed"],
        prepared_observation[0]["inner_seed"],
    )
    assert env.np_random_seed == 19

    env.reset()
    assert build_calls == [("task-a", 7), ("task-b", 7)]


def test_new_seed_discards_incompatible_prepared_episode(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """A cached episode prepared under another seed cannot mask reseeding."""
    env, build_calls = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=[7],
        seed=19,
    )

    env.prepare_for_first_reset()
    env.reset(seed=20)

    assert len(build_calls) == 2
    assert env._prepared_reset_result is None
    assert env.np_random_seed == 20


def test_prepared_reset_consumes_one_dynamic_inner_seed(
    probe_env_factory: Callable[..., tuple[DACBOEnv, list[tuple[str, int]]]],
) -> None:
    """Space preparation exposes the real generated context without skipping ahead."""
    env, build_calls = probe_env_factory(
        task_ids=["task-a", "task-b"],
        inner_seeds=[None],
        seed=19,
    )

    env.prepare_for_first_reset()
    prepared_instance = env.instance
    prepared_observation = env._prepared_reset_result
    observation, _ = env.reset(seed=19)

    assert prepared_observation is not None
    assert len(build_calls) == 1
    assert env.instance == prepared_instance
    assert env.instance == (env.current_seed, env.current_task_id)
    assert env.current_seed is not None
    np.testing.assert_array_equal(
        observation["inner_seed"],
        prepared_observation[0]["inner_seed"],
    )

    env.reset()
    second_instance = env.instance
    assert len(build_calls) == 2
    assert second_instance != prepared_instance
    assert second_instance == (env.current_seed, env.current_task_id)

    env.reset(seed=19)
    assert env.instance == prepared_instance
