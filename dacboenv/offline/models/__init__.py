"""Action-value architectures for offline DACBO."""

from dacboenv.offline.models.shared_dueling_q import OfflineQNetwork, build_offline_q_model

__all__ = ["OfflineQNetwork", "build_offline_q_model"]
