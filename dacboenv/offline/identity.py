"""Collision-proof scientific identities for offline policies and descendants."""

from __future__ import annotations

import re
from math import isfinite


def stable_float_slug(value: float) -> str:
    """Render a finite coefficient without punctuation collisions."""
    if not isfinite(value):
        raise ValueError("Scientific identity coefficients must be finite.")
    rendered = format(float(value), ".12g")
    if rendered.endswith(".0"):
        rendered = rendered[:-2]
    return rendered.replace("-", "m").replace(".", "p").replace("+", "")


def _slug(value: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    if not rendered:
        raise ValueError("Scientific identity component cannot be empty.")
    return rendered


def offline_policy_id(
    *,
    experiment_id: str,
    algorithm_mode: str,
    cql_coefficient: float,
    training_seed: int,
    selected_update: int,
    checkpoint_mode: str,
    model_sha256: str,
) -> str:
    """Build one identity that cannot collide across trained model artifacts."""
    if selected_update < 0:
        raise ValueError("selected_update must be non-negative.")
    if not re.fullmatch(r"[0-9a-f]{64}", model_sha256):
        raise ValueError("model_sha256 must be a lowercase SHA-256 digest.")
    checkpoint_slug = {"best_branch_dev": "bestdev", "final": "final", "explicit": "explicit"}.get(checkpoint_mode)
    if checkpoint_slug is None:
        raise ValueError(f"Unsupported offline checkpoint mode {checkpoint_mode!r}.")
    return (
        f"offline-{_slug(experiment_id)}-{_slug(algorithm_mode)}-"
        f"cql{stable_float_slug(cql_coefficient)}-seed{training_seed}-u{selected_update}-"
        f"{checkpoint_slug}-{model_sha256[:12]}"
    )


__all__ = ["offline_policy_id", "stable_float_slug"]
