"""Audited conversion of fixed-f5 transitions to SB3 DictReplayBuffer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from stable_baselines3.common.buffers import DictReplayBuffer

from dacboenv.offline.dataset import BehaviorDataset
from dacboenv.rl.double_dqn import DoubleDQN

if TYPE_CHECKING:
    from gymnasium import spaces
    from stable_baselines3 import DQN


def hierarchical_prefill_order(
    dataset: BehaviorDataset,
    seed: int,
    *,
    eligible_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Interleave domain/scenario/action/phase strata without replacement."""
    rng = np.random.default_rng(seed)
    arrays = dataset.arrays
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    eligible = np.arange(len(dataset), dtype=np.int64) if eligible_indices is None else np.asarray(eligible_indices)
    for index in eligible.tolist():
        domain = int(arrays["domain_id"][index])
        scenario = int(arrays["scenario_id"][index]) if domain else -1
        key = (domain, scenario, int(arrays["action_index"][index]), int(arrays["phase_bin"][index]))
        groups.setdefault(key, []).append(index)
    shuffled = {}
    for key, values in groups.items():
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        shuffled[key] = list(array)
    order = []
    keys = sorted(shuffled)
    while any(shuffled.values()):
        for key in keys:
            if shuffled[key]:
                order.append(shuffled[key].pop())
    result = np.asarray(order, dtype=np.int64)
    if result.size != len(eligible) or np.unique(result).size != len(eligible):
        raise RuntimeError("Hierarchical prefill must include every transition exactly once.")
    return np.asarray(result, dtype=np.int64)  # type: ignore[no-any-return]


def prefill_dict_replay_buffer(
    replay_buffer: DictReplayBuffer,
    dataset: BehaviorDataset,
    *,
    seed: int,
    maximum_transitions: int | None = None,
    eligible_indices: np.ndarray | None = None,
) -> dict[str, int]:
    """Insert exact f5 transitions with SB3 timeout semantics."""
    order = hierarchical_prefill_order(dataset, seed, eligible_indices=eligible_indices)
    if maximum_transitions is not None:
        order = order[:maximum_transitions]
    if len(order) > replay_buffer.buffer_size:
        raise ValueError("Replay buffer is smaller than requested offline prefill.")
    n_envs = int(replay_buffer.n_envs)
    usable = len(order) - len(order) % n_envs
    if usable == 0:
        raise ValueError(f"Offline prefill needs at least {n_envs} transitions for this vector replay buffer.")
    dropped = len(order) - usable
    order = order[:usable]
    for start in range(0, len(order), n_envs):
        rows = [dataset[int(index)] for index in order[start : start + n_envs]]
        replay_buffer.add(
            {
                "global_state": np.stack([row.global_state for row in rows]),
                "action_features": np.stack([row.action_features for row in rows]),
            },
            {
                "global_state": np.stack([row.next_global_state for row in rows]),
                "action_features": np.stack([row.next_action_features for row in rows]),
            },
            np.asarray([[row.action_index] for row in rows], dtype=np.int64),
            np.asarray([row.reward for row in rows], dtype=np.float32),
            np.asarray([row.terminated or row.truncated for row in rows], dtype=np.float32),
            [
                {
                    # A BO-budget truncation is terminal for this finite dataset;
                    # do not apply SB3's continuing-task TimeLimit bootstrap.
                    "TimeLimit.truncated": False,
                    "offline_source_truncated": row.truncated,
                }
                for row in rows
            ],
        )
    return {
        "inserted": len(order),
        "unique": int(np.unique(order).size),
        "dropped_for_vector_alignment": dropped,
        "buffer_size": int(replay_buffer.size()) * n_envs,
    }


@dataclass(frozen=True, slots=True)
class OfflineOnlineMixSchedule:
    """Linear audited target fraction for alternating offline updates."""

    initial_fraction: float = 0.5
    final_fraction: float = 0.1
    decay_steps: int = 100_000

    def fraction(self, step: int) -> float:
        """Return the requested offline fraction at one update."""
        if not 0 <= self.final_fraction <= self.initial_fraction <= 1:
            raise ValueError("Offline mixture fractions must satisfy 0 <= final <= initial <= 1.")
        progress = min(max(step, 0) / max(self.decay_steps, 1), 1.0)
        return self.initial_fraction + progress * (self.final_fraction - self.initial_fraction)

    def use_offline(self, step: int, seed: int) -> bool:
        """Choose a deterministic alternating update from step identity."""
        return bool(np.random.default_rng(np.random.SeedSequence([seed, step])).random() < self.fraction(step))


def configure_offline_replay(
    model: DQN,
    *,
    dataset_path: Path,
    seed: int,
    maximum_transitions: int | None,
    mixture: OfflineOnlineMixSchedule | None = None,
    domain: str = "mixed",
) -> dict[str, Any]:
    """Prefill the main buffer or attach a separate audited offline buffer."""
    dataset = BehaviorDataset(dataset_path)
    if dataset.split != "train":
        raise ValueError("Replay prefill accepts only finalized offline training transitions.")
    if domain not in {"mixed", "bbob", "yahpo"}:
        raise ValueError("Offline replay domain must be mixed, bbob, or yahpo.")
    eligible = None
    if domain != "mixed":
        domain_id = 0 if domain == "bbob" else 1
        eligible = np.flatnonzero(dataset.arrays["domain_id"] == domain_id)
    if mixture is None:
        if not isinstance(model.replay_buffer, DictReplayBuffer):
            raise TypeError("Offline Dict-observation prefill requires an initialized DictReplayBuffer.")
        result = prefill_dict_replay_buffer(
            model.replay_buffer,
            dataset,
            seed=seed,
            maximum_transitions=maximum_transitions,
            eligible_indices=eligible,
        )
        return {"mode": "main_buffer_prefill", **result}
    if not isinstance(model, DoubleDQN):
        raise TypeError("A scheduled offline/online replay mixture currently requires DoubleDQN.")
    available = len(dataset) if eligible is None else len(eligible)
    capacity = available if maximum_transitions is None else min(available, maximum_transitions)
    offline_buffer = DictReplayBuffer(
        max(capacity, 1),
        cast("spaces.Dict", model.observation_space),
        model.action_space,
        device=model.device,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=False,
    )
    result = prefill_dict_replay_buffer(
        offline_buffer,
        dataset,
        seed=seed,
        maximum_transitions=maximum_transitions,
        eligible_indices=eligible,
    )
    model.configure_offline_replay(offline_buffer, mixture, seed=seed)
    return {"mode": "scheduled_separate_buffer", **result}
