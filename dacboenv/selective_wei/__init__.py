"""Selective residual dynamic-WEI control.

The package deliberately separates context routing, delayed-value prediction,
uncertainty calibration, override gating, and deployment.  Importing it has no
training or CARP-S side effects.
"""

from dacboenv.selective_wei.base_selector import BaseSelector, FittedBaseSelector
from dacboenv.selective_wei.policy import SelectiveWEIPolicy
from dacboenv.selective_wei.schemas import BaseDecision, OverrideDecision, ValuePrediction

__all__ = [
    "BaseDecision",
    "BaseSelector",
    "FittedBaseSelector",
    "OverrideDecision",
    "SelectiveWEIPolicy",
    "ValuePrediction",
]
