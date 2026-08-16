"""Algorithm-neutral command for exporting trained SB3 policy bundles."""

from __future__ import annotations

from fire import Fire

from dacboenv.experiment.collect_ppo import create_ppo_eval_configs


def collect_trained_policies(*args: object, **kwargs: object) -> None:
    """Export PPO, DQN, or Double-DQN checkpoints through the shared bridge."""
    create_ppo_eval_configs(*args, **kwargs)  # type: ignore[arg-type]


if __name__ == "__main__":
    Fire(collect_trained_policies)
