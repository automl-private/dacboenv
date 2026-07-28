"""Compatibility entry point for PPO with observation normalization.

The historical implementation transformed sampled Box actions without
transforming their PPO log probabilities. That optimizes a different
distribution from the one executed by the environment. Keep the module-level
CLI, but route it through the shared Stable-Baselines3 runner instead.
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
