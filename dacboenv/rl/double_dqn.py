"""Minimal Double-DQN extension for the pinned Stable-Baselines3 release."""

from __future__ import annotations

import inspect
from importlib.metadata import version
from typing import TYPE_CHECKING

import numpy as np
import torch as th
from stable_baselines3 import DQN
from torch.nn import functional

if TYPE_CHECKING:
    from stable_baselines3.common.buffers import DictReplayBuffer

SUPPORTED_SB3_VERSION = "2.9.0"
EXPECTED_TRAIN_SIGNATURE = "(self, gradient_steps: int, batch_size: int = 100) -> None"


def require_supported_sb3() -> None:
    """Fail rather than silently copying target logic from another SB3 API."""
    installed = version("stable-baselines3")
    if installed != SUPPORTED_SB3_VERSION:
        raise RuntimeError(
            f"DoubleDQN is pinned to stable-baselines3 {SUPPORTED_SB3_VERSION}; installed version is {installed}."
        )
    signature = str(inspect.signature(DQN.train))
    if signature != EXPECTED_TRAIN_SIGNATURE:
        raise RuntimeError(f"SB3 DQN.train signature changed: expected {EXPECTED_TRAIN_SIGNATURE}, got {signature}.")


def vanilla_dqn_bootstrap(q_target_next: th.Tensor) -> th.Tensor:
    """Return the ordinary DQN maximum target-network value."""
    return q_target_next.max(dim=1, keepdim=True).values


def double_dqn_bootstrap(q_online_next: th.Tensor, q_target_next: th.Tensor) -> th.Tensor:
    """Evaluate the target network at the online network's greedy action."""
    next_actions = q_online_next.argmax(dim=1, keepdim=True)
    return q_target_next.gather(dim=1, index=next_actions)


class DoubleDQN(DQN):
    """Stable-Baselines3 DQN with only Double-DQN target-action selection.

    ``train`` is copied from SB3 2.9.0, release commit ``8908708``.  The sole optimization change is that
    the online Q-network selects the next action and the target Q-network
    evaluates it.  Additional calculations below that line are diagnostics
    detached from optimization.
    """

    algorithm_id = "double_dqn"

    def __init__(self, *args: object, **kwargs: object) -> None:
        require_supported_sb3()
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._offline_replay_buffer: DictReplayBuffer | None = None
        self._offline_mix_schedule: object | None = None
        self._offline_mix_seed = 0
        self._offline_updates = 0

    def configure_offline_replay(
        self,
        replay_buffer: DictReplayBuffer,
        schedule: object,
        *,
        seed: int,
    ) -> None:
        """Attach an immutable prefilled buffer for audited alternating updates."""
        if replay_buffer.size() == 0:
            raise ValueError("Offline replay buffer must contain transitions.")
        if not callable(getattr(schedule, "use_offline", None)):
            raise TypeError("Offline mixture schedule must define use_offline(step, seed).")
        self._offline_replay_buffer = replay_buffer
        self._offline_mix_schedule = schedule
        self._offline_mix_seed = int(seed)

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:  # noqa: PLR0915
        """Perform pinned SB3 DQN updates with Double-DQN bootstrap targets."""
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses: list[float] = []
        q_values: list[float] = []
        target_values: list[float] = []
        td_errors: list[np.ndarray] = []
        disagreements: list[float] = []
        vanilla_minus_double: list[float] = []
        for local_step in range(gradient_steps):
            update_index = self._n_updates + local_step
            use_offline = bool(
                self._offline_replay_buffer is not None
                and self._offline_mix_schedule is not None
                and self._offline_mix_schedule.use_offline(update_index, self._offline_mix_seed)  # type: ignore[attr-defined]
            )
            source_buffer = self._offline_replay_buffer if use_offline else self.replay_buffer
            normalization_env = None if use_offline else self._vec_normalize_env
            replay_data = source_buffer.sample(batch_size, env=normalization_env)  # type: ignore[union-attr]
            self._offline_updates += int(use_offline)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                q_online_next = self.q_net(replay_data.next_observations)
                q_target_next = self.q_net_target(replay_data.next_observations)
                next_q_values = double_dqn_bootstrap(q_online_next, q_target_next)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                vanilla_next = vanilla_dqn_bootstrap(q_target_next)
                disagreements.append(
                    float((q_online_next.argmax(dim=1) != q_target_next.argmax(dim=1)).float().mean().item())
                )
                vanilla_minus_double.append(float((vanilla_next - next_q_values).mean().item()))

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())
            error = (target_q_values - current_q_values).detach().abs().cpu().numpy().reshape(-1)
            loss = functional.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            q_values.extend(current_q_values.detach().cpu().numpy().reshape(-1).tolist())
            target_values.extend(target_q_values.detach().cpu().numpy().reshape(-1).tolist())
            td_errors.append(error)

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        all_td_errors = np.concatenate(td_errors) if td_errors else np.zeros(1)
        transitions = max(int(self.replay_buffer.size()) * int(self.n_envs), 1)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))
        self.logger.record("train/replay_buffer_size", transitions)
        self.logger.record("train/update_to_data_ratio", self._n_updates / transitions)
        self.logger.record("train/q_mean", np.mean(q_values))
        self.logger.record("train/q_std", np.std(q_values))
        self.logger.record("train/target_q_mean", np.mean(target_values))
        self.logger.record("train/td_error_mean", np.mean(all_td_errors))
        self.logger.record("train/td_error_p90", np.quantile(all_td_errors, 0.9))
        self.logger.record("train/online_target_argmax_disagreement", np.mean(disagreements))
        self.logger.record("train/vanilla_minus_double_target_mean", np.mean(vanilla_minus_double))
        self.logger.record("train/realized_offline_update_fraction", self._offline_updates / max(self._n_updates, 1))


__all__ = [
    "SUPPORTED_SB3_VERSION",
    "DoubleDQN",
    "double_dqn_bootstrap",
    "require_supported_sb3",
    "vanilla_dqn_bootstrap",
]
