"""Handling piecewise linear parameter function."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from smac.callback import Callback

if TYPE_CHECKING:
    from smac.main.smbo import SMBO


class PiecewiseCallback(Callback):
    """Callback for utilizing learned piecewise linear function parameter policy."""

    def __init__(self, splity: np.ndarray) -> None:
        self.splity = splity
        super().__init__()

    def on_iteration_start(self, smbo: SMBO) -> None:
        """Update optimizer before each iteration."""
        t = len(smbo.runhistory)
        splits = np.linspace(1, 77, 4 + 1, dtype=int)
        val = np.interp(t, splits, self.splity)

        setattr(
            smbo._intensifier._config_selector._acquisition_function,
            "_beta",  # UCB
            val**2,
        )

        return super().on_iteration_start(smbo)
