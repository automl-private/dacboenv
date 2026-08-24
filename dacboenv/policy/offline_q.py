"""CARP-S deployment bridge for repository offline Q checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from gymnasium.spaces import (
    Dict as DictSpace,
    Discrete,
)

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.offline.models.shared_dueling_q import build_offline_q_model
from dacboenv.offline.normalization import ObservationNormalizer
from dacboenv.policy.abstract_policy import AbstractPolicy

ACTION_COUNT = 5

if TYPE_CHECKING:
    from dacboenv.dacboenv import DACBOEnv
    from dacboenv.env.observations.types import ObsType


class OfflineQPolicy(AbstractPolicy):
    """Load a shared Q model and choose deterministic argmax actions."""

    def __init__(
        self,
        env: DACBOEnv,
        checkpoint: str,
        normalizer: str,
        checkpoint_sha256: str,
        normalizer_sha256: str,
        deployment_head: str = "long_q",
        interaction_frequency: int = 5,
    ) -> None:
        super().__init__(
            env,
            checkpoint=checkpoint,
            normalizer=normalizer,
            checkpoint_sha256=checkpoint_sha256,
            normalizer_sha256=normalizer_sha256,
            deployment_head=deployment_head,
            interaction_frequency=interaction_frequency,
        )
        if not isinstance(env.action_space, Discrete) or int(env.action_space.n) != ACTION_COUNT:
            raise TypeError("Offline dynamic-WEI policy requires Discrete(5).")
        if int(env.interaction_frequency) != int(interaction_frequency):
            raise ValueError("Offline checkpoint requires fixed interaction frequency f=5.")
        if not isinstance(env.observation_space, DictSpace):
            raise TypeError("Offline Q policy requires a Dict observation space.")
        required = {"global_state", "action_features"}
        if not required <= set(env.observation_space.spaces):
            missing = sorted(required - set(env.observation_space.spaces))
            raise ValueError(f"Offline Q policy observation is missing {missing}.")
        checkpoint_path, normalizer_path = Path(checkpoint).resolve(), Path(normalizer).resolve()
        if file_sha256(checkpoint_path) != checkpoint_sha256 or file_sha256(normalizer_path) != normalizer_sha256:
            raise ValueError("Offline policy artifact hash mismatch.")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") not in {
            "dacbo-offline-q-checkpoint-v1",
            "dacbo-offline-q-checkpoint-v2",
        }:
            raise ValueError("Unsupported offline Q checkpoint schema.")
        selection = payload.get("deployment_selection")
        if selection is not None and selection.get("deployment_head") != deployment_head:
            raise ValueError("Requested deployment head differs from checkpoint-selection metadata.")
        self._model = build_offline_q_model(payload["model_config"])
        self._model.load_state_dict(payload["model_state"])
        self._model.eval()
        self._normalizer = ObservationNormalizer.from_dict(json.loads(normalizer_path.read_text(encoding="utf-8")))
        if self._normalizer.train_dataset_sha256 != payload["provenance"]["behavior_train_sha256"]:
            raise ValueError("Checkpoint and normalizer refer to different training datasets.")
        if deployment_head not in {"long_q", "branch_q5"}:
            raise ValueError("deployment_head must be long_q or branch_q5.")
        self._deployment_head = deployment_head

    def __call__(self, obs: ObsType) -> int:
        """Return the greedy action without modifying environment RNG state."""
        state = self._normalizer.global_state.transform(np.asarray(obs["global_state"], dtype=np.float32))
        action_features = self._normalizer.action_features.transform(
            np.asarray(obs["action_features"], dtype=np.float32)
        )
        with torch.no_grad():
            q = self._model(
                torch.as_tensor(state).unsqueeze(0),
                torch.as_tensor(action_features).unsqueeze(0),
                head=self._deployment_head,  # type: ignore[arg-type]
            )
        return int(q.argmax(dim=1).item())

    def set_seed(self, seed: int | None) -> None:
        """Do nothing because deterministic inference has no policy RNG."""


__all__ = ["OfflineQPolicy"]
