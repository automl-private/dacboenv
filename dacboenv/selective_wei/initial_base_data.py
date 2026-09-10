"""Validated paired full-static terminal labels, distinct from local branches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_context import InitialContext, initial_output_scale
from dacboenv.selective_wei.schemas import ACTION_COUNT
from dacboenv.selective_wei.targets import TargetSemantics

FULL_STATIC_TARGET = TargetSemantics("full_static_terminal_loss", "minimize")
INITIAL_BASE_DATA_VERSION = "dacbo-paired-initial-base-v1"


@dataclass(frozen=True, slots=True)
class PairedStaticLabels:
    """Five complete logical incumbent curves including the common design.

    Each row is one static alpha, in ascending alpha order. Every curve includes
    initialization exactly once even when physical initial evaluations are
    reused. Replicates of this object are not independent initial contexts.
    """

    context: InitialContext
    initial_costs: tuple[float, ...]
    incumbent_curves: tuple[tuple[float, ...], ...]
    arm_context_digests: tuple[str, ...]
    arm_design_digests: tuple[str, ...]
    physical_evaluations: tuple[int, ...]
    shared_initial_physical_evaluations: int
    continuation_replicate: int

    def __post_init__(self) -> None:  # noqa: C901, PLR0912 - independent fail-closed pairing checks
        """Reject partial, unpaired, overbudget, or fabricated curve groups."""
        if not all(
            isinstance(value, tuple)
            for value in (
                self.initial_costs,
                self.incumbent_curves,
                self.arm_context_digests,
                self.arm_design_digests,
                self.physical_evaluations,
                *self.incumbent_curves,
            )
        ):
            raise TypeError("Paired static labels must use immutable tuples.")
        curves = np.asarray(self.incumbent_curves, dtype=np.float64)
        costs = np.asarray(self.initial_costs, dtype=np.float64)
        if curves.shape != (5, self.context.total_budget) or not np.isfinite(curves).all():
            raise ValueError("Five complete finite logical curves through T are required.")
        if costs.shape != (self.context.n_initial,) or not np.isfinite(costs).all():
            raise ValueError("Raw D0 costs must include exactly n0 observations.")
        if canonical_hash(costs.tolist()) != self.context.ordered_cost_digest:
            raise ValueError("Initial cost digest does not match the immutable anchor.")
        if self.arm_context_digests != (self.context.digest,) * 5:
            raise ValueError("Static continuations do not share the same verified z0/model/data.")
        if self.arm_design_digests != (self.context.ordered_design_digest,) * 5:
            raise ValueError("Static continuations do not share the same ordered D0.")
        if not np.array_equal(curves[:, : len(costs)], np.tile(np.minimum.accumulate(costs), (5, 1))):
            raise ValueError("Logical curves must include the identical initial incumbents exactly once.")
        if np.any(np.diff(curves, axis=1) > 0):
            raise ValueError("Minimization incumbent curves cannot increase.")
        if len(self.physical_evaluations) != ACTION_COUNT or any(value < 0 for value in self.physical_evaluations):
            raise ValueError("Physical evaluation accounting must be explicit for all five arms.")
        n0, total = self.context.n_initial, self.context.total_budget
        if self.shared_initial_physical_evaluations == n0:
            expected = (total - n0,) * 5
        elif self.shared_initial_physical_evaluations == 0:
            expected = (total,) * 5
        else:
            raise ValueError("Initial physical reuse must be either one complete D0 or none.")
        if self.physical_evaluations != expected:
            raise ValueError("Physical counts disagree with the declared initialization reuse protocol.")

    def targets(self) -> dict[str, object]:
        """Expose authoritative raw terminal, paired scaled, and separate anytime targets."""
        curves = np.asarray(self.incumbent_curves, dtype=np.float64)
        terminal = curves[:, -1]
        scale, scale_reason = initial_output_scale(np.asarray(self.initial_costs))
        return {
            "schema_version": INITIAL_BASE_DATA_VERSION,
            "target_semantics": FULL_STATIC_TARGET.name,
            "terminal_raw": terminal.tolist(),
            "terminal_paired_scaled": ((terminal - terminal[2]) / scale).tolist(),
            "anytime_mean_incumbent": curves.mean(axis=1).tolist(),
            "scale": scale,
            "scale_reason": scale_reason,
            "reference_arm_index": 2,
            "logical_evaluations_per_arm": self.context.total_budget,
        }
