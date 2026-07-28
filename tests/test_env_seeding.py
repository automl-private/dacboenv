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
        env = DACBOEnv(
            action_space_class=ProbeActionSpace,
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
