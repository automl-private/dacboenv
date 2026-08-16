"""Algorithm-neutral Stable-Baselines3 training entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import hydra

from dacboenv.experiment.ppo import run

if TYPE_CHECKING:
    from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run the shared validation/checkpoint pipeline for the configured agent."""
    run(cfg)


if __name__ == "__main__":
    main()
