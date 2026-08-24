"""Train-only normalization for compact DACBO observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

CONSTANT_TOLERANCE = 1e-8


@dataclass(frozen=True, slots=True)
class ArrayNormalizer:
    """Mean/std normalization with explicit constant and identity masks."""

    mean: tuple[float, ...]
    std: tuple[float, ...]
    constant_mask: tuple[bool, ...]
    preserve_mask: tuple[bool, ...]
    clip: float | None = 10.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Normalize the last axis while preserving declared identity columns."""
        array = np.asarray(values, dtype=np.float32)
        mean = np.asarray(self.mean, dtype=np.float32)
        std = np.asarray(self.std, dtype=np.float32)
        transformed = (array - mean) / std
        preserve = np.asarray(self.preserve_mask, dtype=bool)
        transformed[..., preserve] = array[..., preserve]
        if self.clip is not None:
            transformed = np.clip(transformed, -self.clip, self.clip)
        return np.asarray(transformed, dtype=np.float32)  # type: ignore[no-any-return]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ObservationNormalizer:
    """Normalizer for global and per-action feature arrays."""

    global_state: ArrayNormalizer
    action_features: ArrayNormalizer
    train_dataset_sha256: str
    schema_version: str = "dacbo-offline-normalization-v1"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "schema_version": self.schema_version,
            "train_dataset_sha256": self.train_dataset_sha256,
            "global_state": self.global_state.to_dict(),
            "action_features": self.action_features.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ObservationNormalizer:
        """Restore a frozen normalizer from JSON."""
        return cls(
            global_state=ArrayNormalizer(**value["global_state"]),
            action_features=ArrayNormalizer(**value["action_features"]),
            train_dataset_sha256=str(value["train_dataset_sha256"]),
            schema_version=str(value["schema_version"]),
        )


def _fit(values: np.ndarray, preserve_mask: np.ndarray) -> ArrayNormalizer:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, values.shape[-1])
    mean = flat.mean(axis=0)
    raw_std = flat.std(axis=0)
    constant = raw_std < CONSTANT_TOLERANCE
    std = np.where(constant, 1.0, raw_std)
    mean[preserve_mask] = 0.0
    std[preserve_mask] = 1.0
    return ArrayNormalizer(
        mean=tuple(float(item) for item in mean),
        std=tuple(float(item) for item in std),
        constant_mask=tuple(bool(item) for item in constant),
        preserve_mask=tuple(bool(item) for item in preserve_mask),
    )


def fit_observation_normalizer(
    global_state: np.ndarray,
    action_features: np.ndarray,
    *,
    train_dataset_sha256: str,
) -> ObservationNormalizer:
    """Fit only on a caller-provided training split.

    The first action-feature column is the explicit alpha identity in the
    current structured schema and is deliberately preserved.
    """
    return ObservationNormalizer(
        global_state=_fit(global_state, np.zeros(global_state.shape[-1], dtype=bool)),
        action_features=_fit(action_features, np.arange(action_features.shape[-1]) == 0),
        train_dataset_sha256=train_dataset_sha256,
    )
