"""Bounded real initial-context lifecycle checks."""

from __future__ import annotations

from dacboenv.experiment.real_env import real_structured_bbob_smoke_env


def test_real_initial_features_are_frozen_and_reset() -> None:
    """Later refits cannot mutate the initial feature vector."""
    env = real_structured_bbob_smoke_env("bbob/2/3/0", 198, interaction_frequency=5)
    try:
        env.reset()
        settings = {"probe_count": 8}
        initial = env.get_initial_context(settings)
        assert initial.n_initial == 2
        assert set(initial.groups) == {"M", "D", "G", "U"}
        assert initial.values[initial.names.index("actual_n0")] == 2
        env.step(2)
        assert env.get_initial_context(settings) is initial
        env.reset()
        repeated = env.get_initial_context(settings)
        assert repeated is not initial
        assert repeated.digest == initial.digest
    finally:
        env.close()
