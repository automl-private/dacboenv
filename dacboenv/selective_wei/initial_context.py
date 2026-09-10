"""Immutable initial anchors and pure finite-probe posterior diagnostics.

These contracts do not fit a surrogate or generate an optimizer candidate.
The caller must verify the authoritative initialization/model fingerprints
before constructing an anchor. Raw provenance is never a predictor feature.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.schemas import ALPHA_GRID

INITIAL_CONTEXT_VERSION = "dacbo-initial-context-v1"


@dataclass(frozen=True, slots=True)
class InitialContext:
    """One immutable, mask-aware initial feature vector and its provenance."""

    episode_id: str
    names: tuple[str, ...]
    values: tuple[float, ...]
    available: tuple[bool, ...]
    groups: tuple[str, ...]
    n_initial: int
    total_budget: int
    ordered_design_digest: str
    ordered_cost_digest: str
    model_digest: str
    training_data_digest: str
    diagnostic_protocol_digest: str
    diagnostic_seed: int
    schema_version: str = INITIAL_CONTEXT_VERSION

    def __post_init__(self) -> None:
        """Reject incomplete, nonfinite, mutable, or inconsistent anchors."""
        fields = (self.names, self.values, self.available, self.groups)
        if any(not isinstance(value, tuple) for value in fields):
            raise TypeError("Initial feature fields must be immutable tuples.")
        if not self.names or len({len(value) for value in fields}) != 1:
            raise ValueError("Initial feature names, values, masks, and groups must align.")
        if len(set(self.names)) != len(self.names) or not np.isfinite(self.values).all():
            raise ValueError("Initial features require unique names and finite values.")
        if any(group not in {"M", "D", "G", "U"} for group in self.groups):
            raise ValueError("Unknown initial feature group.")
        if self.n_initial <= 0 or self.total_budget < self.n_initial:
            raise ValueError("Initial evaluation count must lie in (0,T].")
        if self.schema_version != INITIAL_CONTEXT_VERSION:
            raise ValueError("Unsupported initial context schema.")
        if not all(
            (
                self.episode_id,
                self.ordered_design_digest,
                self.ordered_cost_digest,
                self.model_digest,
                self.training_data_digest,
                self.diagnostic_protocol_digest,
            )
        ):
            raise ValueError("Initial context requires complete verification fingerprints.")

    @property
    def schema_hash(self) -> str:
        """Hash feature interpretation, not episode identity or task labels."""
        return canonical_hash(
            {
                "version": self.schema_version,
                "names": self.names,
                "groups": self.groups,
                "protocol": self.diagnostic_protocol_digest,
            }
        )

    @property
    def digest(self) -> str:
        """Hash the complete anchor for replay identity checks."""
        return canonical_hash(asdict(self))

    @property
    def has_controlled_budget(self) -> bool:
        """Report whether any post-design action can be executed."""
        return self.total_budget > self.n_initial

    def predictor_input(self, groups: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of features and masks without raw provenance."""
        keep = np.asarray([group in groups for group in self.groups])
        return np.asarray(self.values)[keep].copy(), np.asarray(self.available)[keep].copy()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitialContext:
        """Restore a serialized anchor without permitting mutable fields."""
        data = dict(payload)
        for key in ("names", "values", "available", "groups"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class InitialBaseDecision:
    """The episode's once-selected alpha; overrides never replace this anchor."""

    context_digest: str
    action: int
    base_semantic_hash: str
    selection_seed: int
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        """Validate action and semantic artifact identity."""
        if self.action not in range(5) or not self.context_digest or not self.base_semantic_hash:
            raise ValueError("Initial base decision requires a valid action and hashes.")

    @property
    def alpha(self) -> float:
        """Return the alpha on the frozen five-action grid."""
        return ALPHA_GRID[self.action]


class InitialAnchorCache:
    """Cache once per episode; reject later-GP reselection and bad restores."""

    def __init__(self) -> None:
        self.context: InitialContext | None = None
        self.decision: InitialBaseDecision | None = None

    def reset(self) -> None:
        """Forget both the initial features and the episode's base action."""
        self.context = None
        self.decision = None

    def freeze(self, context: InitialContext) -> InitialContext:
        """Freeze idempotently, refusing to overwrite an episode anchor."""
        if self.context is not None and self.context.digest != context.digest:
            raise ValueError("Initial anchor changed; explicit episode reset is required.")
        self.context = context
        return context

    def select(self, decision: InitialBaseDecision) -> InitialBaseDecision:
        """Accept exactly one action for a verified nonterminal initial state."""
        if self.context is None or decision.context_digest != self.context.digest:
            raise ValueError("Base decision does not match the initial anchor.")
        if not self.context.has_controlled_budget:
            raise ValueError("No controlled budget; no initial base decision is permitted.")
        if self.decision is not None and self.decision != decision:
            raise ValueError("The episode's base action cannot be reselected.")
        self.decision = decision
        return decision

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete immutable anchor state for verified replay."""
        payload = {
            "context": None if self.context is None else asdict(self.context),
            "decision": None if self.decision is None else asdict(self.decision),
        }
        return {**payload, "digest": canonical_hash(payload)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InitialAnchorCache:
        """Restore only internally consistent, unchanged state."""
        core = {key: payload[key] for key in ("context", "decision")}
        if canonical_hash(core) != payload["digest"]:
            raise ValueError("Initial anchor serialization digest mismatch.")
        cache = cls()
        if core["context"] is not None:
            cache.freeze(InitialContext.from_dict(core["context"]))
        if core["decision"] is not None:
            cache.select(InitialBaseDecision(**core["decision"]))
        return cache


def initial_output_scale(costs: np.ndarray, *, floor: float = 1e-12) -> tuple[float, str]:
    """Use D0 IQR, then standard deviation, then a declared unit fallback."""
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or floor <= 0:
        raise ValueError("Initial scale requires finite observed costs and a positive floor.")
    iqr = float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
    if iqr > floor:
        return iqr, "iqr"
    std = float(values.std())
    return (std, "standard_deviation") if std > floor else (1.0, "constant_unit_fallback")


def finite_probe_ubr(
    means: np.ndarray,
    variances: np.ndarray,
    *,
    initial_indices: np.ndarray,
    scale: float,
    confidence_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Compute a finite-probe UCB/LCB gap, not a certified global regret bound.

    Args:
        means: Posterior means in original objective units at all probes.
        variances: Marginal variances in squared original objective units.
        initial_indices: Indices of every completed D0 point among probes.
        scale: Positive scale computed only from D0 outcomes.
        confidence_multiplier: Multiplier on posterior standard deviation.
    """
    mean = np.asarray(means, dtype=np.float64).reshape(-1)
    variance = np.asarray(variances, dtype=np.float64).reshape(-1)
    indices = np.asarray(initial_indices)
    if (
        mean.shape != variance.shape
        or not np.isfinite(mean).all()
        or not np.isfinite(variance).all()
        or np.any(variance < 0)
        or indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= len(mean))
        or not np.isfinite(scale)
        or scale <= 0
        or not np.isfinite(confidence_multiplier)
        or confidence_multiplier < 0
    ):
        raise ValueError("Invalid finite-probe posterior diagnostic inputs.")
    std = np.sqrt(variance)
    proxy = float(
        np.min(mean[indices] + confidence_multiplier * std[indices]) - np.min(mean - confidence_multiplier * std)
    )
    return {
        "raw_proxy": proxy,
        "scaled_proxy": proxy / scale,
        "valid": True,
        "confidence_multiplier": confidence_multiplier,
        "variance_convention": "marginal_variance",
        "posterior_std_quantiles": np.quantile(std, [0.1, 0.5, 0.9]).tolist(),
        "initial_point_count": len(indices),
        "probe_count": len(mean),
        "certified_global_bound": False,
        "gradient_available": False,
    }
