"""State-independent baseline policy contracts."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from dacboenv.policy.random import MarginalRandomPolicy
from gymnasium.spaces import Discrete


def test_marginal_random_policy_matches_rates_and_replays_seed() -> None:
    env = SimpleNamespace(action_space=Discrete(5))
    probabilities = [0.05, 0.1, 0.15, 0.2, 0.5]
    policy = MarginalRandomPolicy(env, probabilities)
    policy.set_seed(17)
    first = [policy(None) for _ in range(20_000)]
    policy.set_seed(17)

    assert [policy(None) for _ in range(20_000)] == first
    np.testing.assert_allclose(np.bincount(first, minlength=5) / len(first), probabilities, atol=0.01)


@pytest.mark.parametrize(
    "probabilities",
    [[0.5, 0.5], [0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, -0.4]],
)
def test_marginal_random_policy_rejects_invalid_rates(probabilities: list[float]) -> None:
    with pytest.raises(ValueError, match=r"Expected|Marginal"):
        MarginalRandomPolicy(SimpleNamespace(action_space=Discrete(5)), probabilities)
