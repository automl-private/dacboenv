"""SB3 DQN policy backed by the shared offline action scorer."""

from __future__ import annotations

from typing import Any, cast

import torch
from stable_baselines3.dqn.policies import DQNPolicy, QNetwork

from dacboenv.offline.models.shared_dueling_q import build_offline_q_model
from dacboenv.offline.normalization import ObservationNormalizer


class OfflineSharedQNetwork(QNetwork):
    """Present :class:`OfflineQNetwork` through SB3's QNetwork interface."""

    def __init__(
        self, *args: Any, offline_model_config: dict[str, object], normalizer: dict[str, Any], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        del self.q_net
        self.shared_q = build_offline_q_model(offline_model_config)
        restored = ObservationNormalizer.from_dict(normalizer)
        self.register_buffer("global_mean", torch.as_tensor(restored.global_state.mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.as_tensor(restored.global_state.std, dtype=torch.float32))
        self.register_buffer("action_mean", torch.as_tensor(restored.action_features.mean, dtype=torch.float32))
        self.register_buffer("action_std", torch.as_tensor(restored.action_features.std, dtype=torch.float32))
        self.register_buffer(
            "action_preserve", torch.as_tensor(restored.action_features.preserve_mask, dtype=torch.bool)
        )

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize raw online observations and return long-horizon Q values."""
        if not isinstance(obs, dict):
            raise TypeError("Offline shared Q requires a Dict observation.")
        global_mean = cast("torch.Tensor", self.global_mean)
        global_std = cast("torch.Tensor", self.global_std)
        action_mean = cast("torch.Tensor", self.action_mean)
        action_std = cast("torch.Tensor", self.action_std)
        action_preserve = cast("torch.Tensor", self.action_preserve)
        state = torch.clamp((obs["global_state"] - global_mean) / global_std, -10.0, 10.0)
        action = (obs["action_features"] - action_mean) / action_std
        action = torch.where(action_preserve, obs["action_features"], action)
        action = torch.clamp(action, -10.0, 10.0)
        return cast("torch.Tensor", self.shared_q(state, action, head="long_q"))


class OfflineSharedDQNPolicy(DQNPolicy):
    """Construct online and target SB3 networks with the offline architecture."""

    def __init__(
        self,
        *args: Any,
        offline_model_config: dict[str, object],
        offline_normalizer: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        self.offline_model_config = offline_model_config
        self.offline_normalizer = offline_normalizer
        super().__init__(*args, **kwargs)

    def make_q_net(self) -> QNetwork:
        """Build one independent shared Q network as required by SB3."""
        net_args = self._update_features_extractor(self.net_args, features_extractor=None)
        return OfflineSharedQNetwork(
            **net_args,
            offline_model_config=self.offline_model_config,
            normalizer=self.offline_normalizer,
        ).to(self.device)


__all__ = ["OfflineSharedDQNPolicy", "OfflineSharedQNetwork"]
