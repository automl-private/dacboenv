"""Deterministic hierarchical sampling for imbalanced offline data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dacboenv.offline.branch_dataset import BranchDataset
    from dacboenv.offline.dataset import BehaviorDataset


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    """Target composition for one behavior batch."""

    batch_size: int = 512
    bbob_fraction: float = 0.5
    positive_fraction: float = 0.25
    ordinary_fraction: float = 0.75
    balance_actions: bool = True
    balance_phases: bool = True
    balance_yahpo_scenarios: bool = True


@dataclass(frozen=True, slots=True)
class BranchSamplerConfig:
    """Target composition for one all-action branch-state batch."""

    batch_size: int = 512
    bbob_fraction: float = 0.5
    high_gap_fraction: float = 0.25
    high_gap_threshold: float = 1e-3
    balance_phases: bool = True
    balance_yahpo_scenarios: bool = True


class HierarchicalBatchSampler:
    """Sample reproducibly while reporting unavoidable replacement."""

    def __init__(
        self,
        dataset: BehaviorDataset,
        config: SamplerConfig,
        seed: int,
        *,
        eligible_indices: np.ndarray | None = None,
    ) -> None:
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if not np.isclose(config.positive_fraction + config.ordinary_fraction, 1.0):
            raise ValueError("ordinary_fraction and positive_fraction must sum to one.")
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.eligible_indices = (
            np.arange(len(dataset), dtype=np.int64)
            if eligible_indices is None
            else np.asarray(eligible_indices, dtype=np.int64)
        )
        if self.eligible_indices.size == 0:
            raise ValueError("Hierarchical sampler has no eligible transitions.")
        self.last_composition: dict[str, object] = {}

    def _strata(self, indices: np.ndarray) -> dict[tuple[int, int, int, int], np.ndarray]:
        arrays = self.dataset.arrays
        result: dict[tuple[int, int, int, int], list[int]] = {}
        for index in indices.tolist():
            domain = int(arrays["domain_id"][index])
            scenario = int(arrays["scenario_id"][index]) if domain == 1 and self.config.balance_yahpo_scenarios else -1
            action = int(arrays["action_index"][index]) if self.config.balance_actions else -1
            phase = int(arrays["phase_bin"][index]) if self.config.balance_phases else -1
            result.setdefault((domain, scenario, action, phase), []).append(index)
        return {key: np.asarray(value, dtype=np.int64) for key, value in result.items()}

    def _draw_balanced(self, pool: np.ndarray, count: int) -> tuple[np.ndarray, int]:
        strata = self._strata(pool)
        if not strata:
            raise ValueError("Cannot sample from an empty offline stratum.")
        keys = sorted(strata)
        allocation: np.ndarray = np.full(len(keys), count // len(keys), dtype=int)
        allocation[: count % len(keys)] += 1
        draws, replacements = [], 0
        for key, requested in zip(keys, allocation, strict=True):
            candidates = strata[key]
            replacement = requested > len(candidates)
            replacements += max(requested - len(candidates), 0)
            if requested:
                draws.append(self.rng.choice(candidates, size=requested, replace=replacement))
        return np.concatenate(draws), replacements

    def _draw_domains(self, pool: np.ndarray, count: int) -> tuple[np.ndarray, int]:
        domain = np.asarray(self.dataset.arrays["domain_id"])
        bbob_count = round(count * self.config.bbob_fraction)
        allocations = ((0, bbob_count), (1, count - bbob_count))
        draws, replacements = [], 0
        for domain_id, requested in allocations:
            if requested == 0:
                continue
            candidates = pool[domain[pool] == domain_id]
            if candidates.size == 0 and requested:
                raise ValueError(f"Requested domain {domain_id} but its sampling pool is empty.")
            current, repeated = self._draw_balanced(candidates, requested)
            draws.append(current)
            replacements += repeated
        return np.concatenate(draws) if draws else np.empty(0, dtype=np.int64), replacements

    def sample(self) -> np.ndarray:
        """Return one shuffled deterministic batch and its recorded composition."""
        all_indices = self.eligible_indices
        positive = all_indices[np.asarray(self.dataset.arrays["reward"])[all_indices] > 0]
        positive_count = round(self.config.batch_size * self.config.positive_fraction)
        ordinary_count = self.config.batch_size - positive_count
        ordinary_draw, ordinary_replacement = self._draw_domains(all_indices, ordinary_count)
        positive_draw, positive_replacement = self._draw_domains(positive, positive_count)
        draw = np.concatenate((ordinary_draw, positive_draw))
        self.rng.shuffle(draw)
        arrays = self.dataset.arrays
        actions = Counter(np.asarray(arrays["action_index"])[draw].tolist())
        phases = Counter(np.asarray(arrays["phase_bin"])[draw].tolist())
        scenarios = Counter(np.asarray(arrays["scenario_id"])[draw].tolist())
        self.last_composition = {
            "bbob_fraction": float(np.mean(arrays["domain_id"][draw] == 0)),
            "yahpo_fraction": float(np.mean(arrays["domain_id"][draw] == 1)),
            "positive_reward_fraction": float(np.mean(arrays["reward"][draw] > 0)),
            "action_frequencies": {str(key): value / len(draw) for key, value in sorted(actions.items())},
            "phase_frequencies": {str(key): value / len(draw) for key, value in sorted(phases.items())},
            "scenario_frequencies": {str(key): value / len(draw) for key, value in sorted(scenarios.items())},
            "replacement_draws": ordinary_replacement + positive_replacement,
            "effective_unique_rows": int(np.unique(draw).size),
        }
        return np.asarray(draw, dtype=np.int64)  # type: ignore[no-any-return]


class BranchBatchSampler:
    """Deterministically mix ordinary and informative branch states.

    The sampler reports replacement and unique-state counts so a sparse
    high-gap stratum cannot be duplicated silently.
    """

    def __init__(
        self,
        dataset: BranchDataset,
        config: BranchSamplerConfig,
        seed: int,
        *,
        eligible_indices: np.ndarray | None = None,
    ) -> None:
        self.dataset = dataset
        self.config = config
        self.rng = np.random.default_rng(seed)
        self.eligible_indices = (
            np.arange(len(dataset), dtype=np.int64)
            if eligible_indices is None
            else np.asarray(eligible_indices, dtype=np.int64)
        )
        if self.eligible_indices.size == 0:
            raise ValueError("Branch sampler has no eligible states.")
        if not 0 <= config.high_gap_fraction <= 1 or not 0 <= config.bbob_fraction <= 1:
            raise ValueError("Branch sampler fractions must lie in [0, 1].")
        self.last_composition: dict[str, object] = {}

    def _strata(self, pool: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
        arrays = self.dataset.arrays
        groups: dict[tuple[int, int, int], list[int]] = {}
        for index in pool.tolist():
            domain = int(arrays["domain_id"][index])
            scenario = int(arrays["scenario_id"][index]) if domain == 1 and self.config.balance_yahpo_scenarios else -1
            phase = int(arrays["phase_bin"][index]) if self.config.balance_phases else -1
            groups.setdefault((domain, scenario, phase), []).append(index)
        return {key: np.asarray(values, dtype=np.int64) for key, values in groups.items()}

    def _draw_balanced(self, pool: np.ndarray, count: int) -> tuple[np.ndarray, int]:
        if count == 0:
            return np.empty(0, dtype=np.int64), 0
        if pool.size == 0:
            raise ValueError("Requested a non-empty branch batch from an empty stratum.")
        groups = self._strata(pool)
        keys = sorted(groups)
        allocation: np.ndarray = np.full(len(keys), count // len(keys), dtype=np.int64)
        allocation[: count % len(keys)] += 1
        draws: list[np.ndarray] = []
        replacements = 0
        for key, requested in zip(keys, allocation, strict=True):
            candidates = groups[key]
            replacement = requested > len(candidates)
            replacements += max(int(requested) - len(candidates), 0)
            if requested:
                draws.append(self.rng.choice(candidates, int(requested), replace=replacement))
        return np.concatenate(draws), replacements

    def _draw(self, pool: np.ndarray, count: int) -> tuple[np.ndarray, int]:
        """Draw with an explicit BBOB/YAHPO allocation before subgroup balance."""
        domains = np.asarray(self.dataset.arrays["domain_id"])
        bbob_count = round(count * self.config.bbob_fraction)
        pieces: list[np.ndarray] = []
        replacements = 0
        for domain_id, requested in ((0, bbob_count), (1, count - bbob_count)):
            if requested == 0:
                continue
            candidates = pool[domains[pool] == domain_id]
            part, repeated = self._draw_balanced(candidates, requested)
            pieces.append(part)
            replacements += repeated
        return np.concatenate(pieces), replacements

    def sample(self) -> np.ndarray:
        """Draw one deterministic batch with an explicit high-gap component."""
        gaps = np.asarray(self.dataset.arrays["top1_top2_gap_q5"], dtype=np.float64)
        high_pool = self.eligible_indices[gaps[self.eligible_indices] > self.config.high_gap_threshold]
        high_count = round(self.config.batch_size * self.config.high_gap_fraction)
        ordinary_count = self.config.batch_size - high_count
        ordinary, ordinary_repeated = self._draw(self.eligible_indices, ordinary_count)
        high, high_repeated = self._draw(high_pool, high_count)
        batch = np.concatenate((ordinary, high))
        self.rng.shuffle(batch)
        domains = np.asarray(self.dataset.arrays["domain_id"])[batch]
        self.last_composition = {
            "bbob_fraction": float(np.mean(domains == 0)),
            "yahpo_fraction": float(np.mean(domains == 1)),
            "high_gap_fraction": float(np.mean(gaps[batch] > self.config.high_gap_threshold)),
            "replacement_draws": ordinary_repeated + high_repeated,
            "effective_unique_states": int(np.unique(batch).size),
        }
        return np.asarray(batch, dtype=np.int64)  # type: ignore[no-any-return]
