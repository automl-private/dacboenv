"""Compatibility entry point for normalized AlphaNet PPO training.

All sampling, parallelization, checkpointing, and evaluation behavior now
comes from the shared Stable-Baselines3 runner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from dacboenv.experiment.ppo import run

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run the corrected PPO trainer with ``VecNormalize`` enabled."""
    cfg.experiment.vecnormalize = True
    run(cfg)


if __name__ == "__main__":
    main()
