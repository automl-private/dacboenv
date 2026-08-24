"""In-memory typed loaders for finalized offline DACBO arrays."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.offline.schema import validate_behavior_arrays

if TYPE_CHECKING:
    from dacboenv.offline.normalization import ObservationNormalizer


class HoldoutAccessError(PermissionError):
    """Raised when ordinary code attempts to open the sealed holdout split."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class BehaviorRow:
    """One fixed-frequency behavior transition."""

    global_state: np.ndarray
    action_features: np.ndarray
    action_index: int
    reward: float
    next_global_state: np.ndarray
    next_action_features: np.ndarray
    terminated: bool
    truncated: bool
    behavior_probability: float
    task_id: str
    task_index: int
    domain_id: int
    scenario_id: int
    phase_bin: int
    episode_index: int
    seed: int
    policy_id: str


class BehaviorDataset:
    """Validated behavior dataset with row, episode, and tensor access."""

    def __init__(
        self,
        path: Path,
        *,
        allow_holdout: bool = False,
        unseal_reason: str | None = None,
        repository_revision: str | None = None,
        config_hash: str | None = None,
        normalizer: ObservationNormalizer | None = None,
    ) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: np.asarray(source[key]) for key in source.files}
        self.metadata = validate_behavior_arrays(self.arrays)
        self.split = str(self.metadata["split"])
        if self.split == "holdout":
            if not allow_holdout:
                raise HoldoutAccessError(
                    "Offline holdout is sealed. Set offline_data.allow_holdout=true with an explicit reason."
                )
            if not unseal_reason:
                raise HoldoutAccessError("Unsealing offline holdout requires a non-empty reason.")
            marker = self.path.parent / "HOLDOUT_UNSEALED.json"
            if marker.exists():
                raise HoldoutAccessError(f"Holdout was already unsealed; refusing to overwrite marker {marker}.")
            payload = {
                "schema_version": "dacbo-offline-holdout-unseal-v1",
                "timestamp": datetime.now(UTC).isoformat(),
                "git_revision": repository_revision or "unknown",
                "config_hash": config_hash or "unknown",
                "dataset_sha256": file_sha256(self.path),
                "reason": unseal_reason,
            }
            payload["marker_hash"] = canonical_sha256(payload)
            _atomic_json(marker, payload)
        self.normalizer = normalizer

    def __len__(self) -> int:
        """Return the number of transitions."""
        return int(self.arrays["reward"].shape[0])

    def __getitem__(self, index: int) -> BehaviorRow:
        """Return one copy-safe typed transition."""
        global_state = np.asarray(self.arrays["global_state"][index], dtype=np.float32)
        action_features = np.asarray(self.arrays["action_features"][index], dtype=np.float32)
        next_global = np.asarray(self.arrays["next_global_state"][index], dtype=np.float32)
        next_actions = np.asarray(self.arrays["next_action_features"][index], dtype=np.float32)
        if self.normalizer is not None:
            global_state = self.normalizer.global_state.transform(global_state)
            action_features = self.normalizer.action_features.transform(action_features)
            next_global = self.normalizer.global_state.transform(next_global)
            next_actions = self.normalizer.action_features.transform(next_actions)
        return BehaviorRow(
            global_state=global_state,
            action_features=action_features,
            action_index=int(self.arrays["action_index"][index]),
            reward=float(self.arrays["reward"][index]),
            next_global_state=next_global,
            next_action_features=next_actions,
            terminated=bool(self.arrays["terminated"][index]),
            truncated=bool(self.arrays["truncated"][index]),
            behavior_probability=float(self.arrays["behavior_probability"][index]),
            task_id=str(self.arrays["task_id"][index]),
            task_index=int(self.arrays["task_index"][index]),
            domain_id=int(self.arrays["domain_id"][index]),
            scenario_id=int(self.arrays["scenario_id"][index]),
            phase_bin=int(self.arrays["phase_bin"][index]),
            episode_index=int(self.arrays["episode_index"][index]),
            seed=int(self.arrays["seed"][index]),
            policy_id=str(self.arrays["policy_id"][index]),
        )

    def episode(self, episode_index: int) -> range:
        """Return transition indices for one complete episode."""
        offsets = self.arrays["episode_offsets"]
        if not 0 <= episode_index < len(offsets) - 1:
            raise IndexError(episode_index)
        return range(int(offsets[episode_index]), int(offsets[episode_index + 1]))

    def episodes(self) -> Iterator[range]:
        """Iterate over all complete episodes."""
        for index in range(len(self.arrays["episode_offsets"]) - 1):
            yield self.episode(index)

    def torch_batch(self, indices: np.ndarray, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Convert selected transitions to training tensors."""
        idx = np.asarray(indices, dtype=np.int64)
        global_state = self.arrays["global_state"][idx]
        action_features = self.arrays["action_features"][idx]
        next_global = self.arrays["next_global_state"][idx]
        next_actions = self.arrays["next_action_features"][idx]
        if self.normalizer is not None:
            global_state = self.normalizer.global_state.transform(global_state)
            action_features = self.normalizer.action_features.transform(action_features)
            next_global = self.normalizer.global_state.transform(next_global)
            next_actions = self.normalizer.action_features.transform(next_actions)
        terminal = np.logical_or(self.arrays["terminated"][idx], self.arrays["truncated"][idx])
        return {
            "global_state": torch.as_tensor(global_state, dtype=torch.float32, device=device),
            "action_features": torch.as_tensor(action_features, dtype=torch.float32, device=device),
            "action": torch.as_tensor(self.arrays["action_index"][idx], dtype=torch.long, device=device),
            "reward": torch.as_tensor(self.arrays["reward"][idx], dtype=torch.float32, device=device),
            "next_global_state": torch.as_tensor(next_global, dtype=torch.float32, device=device),
            "next_action_features": torch.as_tensor(next_actions, dtype=torch.float32, device=device),
            "done": torch.as_tensor(terminal, dtype=torch.float32, device=device),
            "timeout": torch.as_tensor(self.arrays["truncated"][idx], dtype=torch.float32, device=device),
            "behavior_probability": torch.as_tensor(
                self.arrays["behavior_probability"][idx], dtype=torch.float32, device=device
            ),
        }
