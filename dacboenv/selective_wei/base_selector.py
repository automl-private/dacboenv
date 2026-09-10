"""Training-only contextual base selectors for selective WEI."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from dacboenv.selective_wei.context import canonical_hash, context_from_task_id, domain_context_key, exact_context_key
from dacboenv.selective_wei.schemas import ACTION_COUNT, ALPHA_GRID, BaseDecision, Horizon, StaticContext

if TYPE_CHECKING:
    from numpy.typing import NDArray

BaseKind = Literal[
    "global_static", "context_mean", "context_lcb", "context_cvar", "context_stochastic", "context_phase"
]
MINIMUM_LCB_TASKS = 2


@dataclass(frozen=True, slots=True)
class BaseSelectorRegistry:
    """Versioned task-balanced contextual selector artifact."""

    schema_version: str
    selector_id: str
    kind: BaseKind
    source_split: str
    task_ids: tuple[str, ...]
    task_list_hash: str
    horizon: Horizon
    aggregation_rule: str
    fallback_hierarchy: tuple[str, ...]
    selected_actions: dict[str, int]
    action_values: dict[str, tuple[float, ...]]
    action_probabilities: dict[str, tuple[float, ...]]
    bootstrap_metadata: dict[str, Any]
    fit_manifest_hash: str
    fit_data_hash: str
    code_revision: str
    schema_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


class BaseSelector(ABC):
    """Common interface for frozen deployment base policies."""

    @abstractmethod
    def select(self, context: StaticContext) -> BaseDecision:
        """Select a base action using only stable deployment context."""


class FittedBaseSelector(BaseSelector):
    """Apply a fitted registry with exact, domain, then global fallback."""

    def __init__(self, registry: BaseSelectorRegistry) -> None:
        self.registry = registry

    def select(self, context: StaticContext) -> BaseDecision:
        """Select the frozen action and report which fallback matched."""
        include_phase = self.registry.kind == "context_phase"
        keys = (
            (exact_context_key(context, include_phase=include_phase), "exact"),
            (domain_context_key(context), "domain"),
            ("global", "global"),
        )
        for key, fallback in keys:
            if key in self.registry.selected_actions:
                action = self.registry.selected_actions[key]
                probabilities = self.registry.action_probabilities.get(key)
                values = self.registry.action_values.get(key)
                emergency = key == "global" and self.registry.bootstrap_metadata.get("global_is_emergency", False)
                return BaseDecision(
                    action=action,
                    alpha=ALPHA_GRID[action],
                    selector_id=self.registry.selector_id,
                    matched_context=key,
                    fallback_level="emergency" if emergency else fallback,  # type: ignore[arg-type]
                    horizon=self.registry.horizon,
                    fit_manifest_hash=self.registry.fit_manifest_hash,
                    fit_data_hash=self.registry.fit_data_hash,
                    action_probabilities=probabilities,
                    unique_task_support=self.registry.bootstrap_metadata.get("unique_task_support", {}).get(key),
                    expected_base_value=(
                        None if values is None else float(np.dot(probabilities or _one_hot(action), values))
                    ),
                )
        raise RuntimeError("Base selector registry has no global fallback.")


def _one_hot(action: int) -> tuple[float, ...]:
    return tuple(float(index == action) for index in range(ACTION_COUNT))


def _context_for_task(task_id: str, phase_bin: int | None = None) -> StaticContext:
    return context_from_task_id(task_id, phase_bin=phase_bin)


def _task_action_table(
    q_values: NDArray[np.floating],
    task_ids: NDArray[np.str_],
    row_indices: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Average within task first so long trajectories do not dominate."""
    selected_tasks = sorted(set(task_ids[row_indices].tolist()))
    return np.asarray(
        [q_values[row_indices[task_ids[row_indices] == task]].mean(axis=0) for task in selected_tasks],
        dtype=np.float64,
    )


def _lower_tail(values: NDArray[np.float64], level: float) -> NDArray[np.float64]:
    count = max(1, math.ceil(level * len(values)))
    return np.asarray(np.sort(values, axis=0)[:count].mean(axis=0), dtype=np.float64)


def _statistic(
    task_values: NDArray[np.float64],
    kind: BaseKind,
    *,
    kappa: float,
    cvar_level: float,
) -> NDArray[np.float64]:
    means = task_values.mean(axis=0)
    if kind == "context_lcb":
        se = task_values.std(axis=0, ddof=1) / np.sqrt(len(task_values)) if len(task_values) > 1 else np.full(5, np.inf)
        return np.asarray(means - kappa * se, dtype=np.float64)
    if kind == "context_cvar":
        return _lower_tail(task_values, cvar_level)
    return np.asarray(means, dtype=np.float64)


def fit_base_selector(  # noqa: PLR0913, C901
    *,
    q_values: NDArray[np.floating],
    task_ids: NDArray[np.str_],
    phase_bins: NDArray[np.integer] | None,
    kind: BaseKind,
    horizon: Horizon,
    source_split: str,
    fit_manifest_hash: str,
    fit_data_hash: str,
    code_revision: str,
    kappa: float = 1.0,
    cvar_level: float = 0.2,
    stochastic_temperature: float = 0.05,
    emergency_action: int = 2,
) -> FittedBaseSelector:
    """Fit a task-balanced base without validation or privileged features."""
    q = np.asarray(q_values, dtype=np.float64)
    tasks = np.asarray(task_ids).astype(str)
    if q.shape != (len(tasks), ACTION_COUNT) or not np.isfinite(q).all():
        raise ValueError("Base selector inputs must be finite arrays with shape (states, 5).")
    if source_split != "train":
        raise ValueError("Base selectors may only be fit from the offline train-task split.")
    if not len(tasks) or emergency_action not in range(ACTION_COUNT):
        raise ValueError("Base fitting requires tasks and a valid emergency action.")
    if kind == "context_phase" and phase_bins is None:
        raise ValueError("Context-plus-phase ablation requires phase bins.")
    if not 0.0 < cvar_level <= 1.0 or stochastic_temperature <= 0:
        raise ValueError("CVaR level and stochastic temperature must be positive.")
    group_indices: dict[str, list[int]] = {"global": list(range(len(tasks)))}
    for index, task in enumerate(tasks):
        context = _context_for_task(task, None if phase_bins is None else int(phase_bins[index]))
        if kind != "global_static":
            group_indices.setdefault(domain_context_key(context), []).append(index)
            group_indices.setdefault(exact_context_key(context, include_phase=kind == "context_phase"), []).append(
                index
            )
    actions: dict[str, int] = {}
    values_by_key: dict[str, tuple[float, ...]] = {}
    probabilities: dict[str, tuple[float, ...]] = {}
    support: dict[str, int] = {}
    unsupported: list[str] = []
    for key, indices in sorted(group_indices.items()):
        task_values = _task_action_table(q, tasks, np.asarray(indices, dtype=np.int64))
        support[key] = len(task_values)
        if kind == "context_lcb" and len(task_values) < MINIMUM_LCB_TASKS:
            unsupported.append(key)
            continue
        score = _statistic(task_values, kind, kappa=kappa, cvar_level=cvar_level)
        mean = task_values.mean(axis=0)
        if kind == "context_stochastic":
            logits = (score - score.max()) / stochastic_temperature
            probability = np.exp(logits) / np.exp(logits).sum()
            action = int(np.flatnonzero(probability == probability.max())[0])
        else:
            action = int(np.flatnonzero(score == score.max())[0])
            probability = np.asarray(_one_hot(action))
        actions[key] = action
        values_by_key[key] = tuple(float(value) for value in mean)
        probabilities[key] = tuple(float(value) for value in probability)
    if "global" not in actions:
        actions["global"] = emergency_action
        probabilities["global"] = _one_hot(emergency_action)
        values_by_key["global"] = tuple(
            float(value) for value in _task_action_table(q, tasks, np.arange(len(tasks))).mean(axis=0)
        )
    task_list = tuple(sorted(set(tasks.tolist())))
    core = {
        "schema_version": "dacbo-selective-base-registry-v1",
        "selector_id": f"B-{kind}-h{horizon}",
        "kind": kind,
        "source_split": source_split,
        "task_ids": task_list,
        "task_list_hash": canonical_hash(task_list),
        "horizon": horizon,
        "aggregation_rule": "mean-within-task-then-context",
        "fallback_hierarchy": ("exact", "domain", "global"),
        "selected_actions": actions,
        "action_values": values_by_key,
        "action_probabilities": probabilities,
        "bootstrap_metadata": {
            "unique_task_support": support,
            "unsupported_contexts": unsupported,
            "emergency_action": emergency_action,
            "global_is_emergency": "global" in unsupported,
            "unit": "task",
            "kappa": kappa if kind == "context_lcb" else None,
            "cvar_level": cvar_level if kind == "context_cvar" else None,
            "stochastic_temperature": stochastic_temperature if kind == "context_stochastic" else None,
        },
        "fit_manifest_hash": fit_manifest_hash,
        "fit_data_hash": fit_data_hash,
        "code_revision": code_revision,
    }
    schema_hash = canonical_hash({"fields": sorted(core), "schema_version": core["schema_version"]})
    registry = BaseSelectorRegistry(**core, schema_hash=schema_hash)  # type: ignore[arg-type]
    return FittedBaseSelector(registry)


def save_base_selector(selector: FittedBaseSelector, path: Path) -> str:
    """Atomically save a fitted selector and return its content hash."""
    payload = selector.registry.to_dict()
    payload["registry_hash"] = canonical_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return str(payload["registry_hash"])


def load_base_selector(path: Path, *, expected_hash: str | None = None) -> FittedBaseSelector:
    """Load and verify a frozen base-selector registry."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("registry_hash"))
    if canonical_hash(payload) != stored or (expected_hash is not None and stored != expected_hash):
        raise ValueError("Base selector registry hash mismatch.")
    return FittedBaseSelector(BaseSelectorRegistry(**payload))
