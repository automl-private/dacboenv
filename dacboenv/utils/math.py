"""Math helper functions."""

from __future__ import annotations

import numpy as np


def safe_log10(x: np.ndarray | float, eps: float = 1e-10) -> float:
    """
    Computes a numerically safe logarithm of x.

    Parameters
    ----------
    - x : array-like or scalar
    - eps : float, small value to avoid log10(0)

    Returns
    -------
    - log10(x) safely
    """
    x = np.asarray(x)
    return np.log10(np.maximum(x, eps))


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid function.

    Parameters
    ----------
    z : scalar or array-like
        Input

    Returns
    -------
    scalar or array-like
        Sigmoid of input.
    """
    return 1 / (1 + np.exp(-z))
