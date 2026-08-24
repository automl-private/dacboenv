"""Typed same-state all-action branch datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from dacboenv.offline.provenance import reject_training_provenance
from dacboenv.offline.schema import validate_branch_arrays

if TYPE_CHECKING:
    from dacboenv.offline.normalization import ObservationNormalizer


class BranchDataset:
    """Load initial Q5 or mid-run Q5/Q10 branch targets."""

    def __init__(self, path: Path, *, normalizer: ObservationNormalizer | None = None) -> None:
        self.path = path.resolve()
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: np.asarray(source[key]) for key in source.files}
        self.metadata = validate_branch_arrays(self.arrays)
        reject_training_provenance(self.metadata)
        if self.metadata.get("context_split") not in {"train", "dev"}:
            raise ValueError("Offline branch training accepts only train/dev contexts.")
        self.normalizer = normalizer

    def __len__(self) -> int:
        """Return the number of branch states."""
        return int(self.arrays["q5"].shape[0])

    @property
    def has_q10(self) -> bool:
        """Whether H=10 targets are present."""
        return "q10" in self.arrays

    def torch_batch(self, indices: np.ndarray, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Convert selected states and all-action labels to tensors."""
        idx = np.asarray(indices, dtype=np.int64)
        state = self.arrays["global_state"][idx]
        features = self.arrays["action_features"][idx]
        if self.normalizer is not None:
            state = self.normalizer.global_state.transform(state)
            features = self.normalizer.action_features.transform(features)
        result = {
            "global_state": torch.as_tensor(state, dtype=torch.float32, device=device),
            "action_features": torch.as_tensor(features, dtype=torch.float32, device=device),
            "q5": torch.as_tensor(self.arrays["q5"][idx], dtype=torch.float32, device=device),
            "valid_action_mask": torch.as_tensor(
                self.arrays["valid_action_mask"][idx], dtype=torch.bool, device=device
            ),
            "tie_mask_q5": torch.as_tensor(self.arrays["tie_mask_q5"][idx], dtype=torch.bool, device=device),
            "gap_q5": torch.as_tensor(self.arrays["top1_top2_gap_q5"][idx], dtype=torch.float32, device=device),
        }
        if "candidate_duplicate_groups" in self.arrays:
            duplicate_groups = np.asarray(
                [json.loads(str(value)) for value in self.arrays["candidate_duplicate_groups"][idx]],
                dtype=np.int64,
            )
        else:
            duplicate_groups = np.tile(np.arange(5, dtype=np.int64), (len(idx), 1))
        result["duplicate_groups"] = torch.as_tensor(duplicate_groups, dtype=torch.long, device=device)
        if self.has_q10:
            result["q10"] = torch.as_tensor(self.arrays["q10"][idx], dtype=torch.float32, device=device)
            result["tie_mask_q10"] = torch.as_tensor(self.arrays["tie_mask_q10"][idx], dtype=torch.bool, device=device)
            result["gap_q10"] = torch.as_tensor(
                self.arrays["top1_top2_gap_q10"][idx], dtype=torch.float32, device=device
            )
        return result
