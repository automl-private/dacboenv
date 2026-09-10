"""Noninterchangeable scientific targets for the two-step controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TargetSemantics:
    """Versioned label semantics, independent of filenames and model heads."""

    name: Literal[
        "full_static_terminal_loss",
        "fixed_action_local_gain",
        "override_then_base_terminal_advantage",
    ]
    direction: Literal["minimize", "maximize"]
    horizon: int | None = None
    version: int = 1

    def __post_init__(self) -> None:
        """Reject ambiguous terminal/local target declarations."""
        if self.version != 1:
            raise ValueError("Unsupported target semantics version.")
        if self.name == "full_static_terminal_loss":
            if self.direction != "minimize" or self.horizon is not None:
                raise ValueError("Full-static targets minimize terminal loss, not a local horizon gain.")
        elif self.name in {"fixed_action_local_gain", "override_then_base_terminal_advantage"}:
            if self.direction != "maximize" or self.horizon not in (5, 10):
                raise ValueError("Delayed advantages maximize gain at an explicit H=5 or H=10.")
        else:
            raise ValueError("Unknown target semantics.")

    def require_compatible(self, other: TargetSemantics) -> None:
        """Fail closed rather than treating local labels as terminal labels."""
        if self != other:
            raise ValueError(f"Incompatible target semantics: {self!r} versus {other!r}.")
