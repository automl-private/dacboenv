"""Evaluate random and static policies on DACBOEnv."""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from carps.loggers.file_logger import get_run_directory
from carps.utils.loggingutils import get_logger
from carps.utils.running import make_task
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# Register OmegaConf resolvers.
import dacboenv  # noqa: F401
from dacboenv.experiment.protocol import (
    require_runnable_manifest,
    validate_manifest_structure,
    validate_native_bbob_manifest,
)
from dacboenv.task import rollout

if TYPE_CHECKING:
    from dacboenv.policy.abstract_policy import AbstractPolicy

logger = get_logger("Baseline")


def _make_policy(cfg: DictConfig, env: Any) -> AbstractPolicy:
    """Instantiate one configured policy against ``env``."""
    policy_factory = instantiate(cfg.optimizer.policy_class)
    policy_kwargs = OmegaConf.to_container(
        cfg.optimizer.get("policy_kwargs", {}),
        resolve=True,
    )
    if not isinstance(policy_kwargs, dict):
        raise TypeError("optimizer.policy_kwargs must be a mapping.")
    policy = policy_factory(env=env, **policy_kwargs)
    policy.set_seed(int(cfg.seed))
    return policy


@hydra.main(version_base=None, config_path="../configs")  # type: ignore[misc]
def main(cfg: DictConfig) -> None:
    """Run one complete baseline pass over the configured context set."""
    logger.info(OmegaConf.to_yaml(cfg))
    manifest = OmegaConf.to_container(cfg.evaluation_instances, resolve=True)
    if not isinstance(manifest, dict):
        raise TypeError("evaluation_instances must be a manifest mapping.")
    validate_manifest_structure(manifest)
    require_runnable_manifest(manifest)
    if manifest["split"] == "test" and not bool(cfg.experiment.get("allow_sealed_test", False)):
        raise PermissionError(
            f"Manifest {manifest['id']!r} is sealed test data; set "
            "experiment.allow_sealed_test=true only for an authorized final report."
        )
    if manifest["domain"] == "bbob":
        validate_native_bbob_manifest(manifest)
    with open_dict(cfg.dacboenv):
        cfg.dacboenv.context_split = str(manifest["split"])
        cfg.dacboenv.protocol_metadata = {
            "evaluation_manifest/version": int(manifest["schema_version"]),
            "evaluation_manifest_id": str(manifest["id"]),
            "evaluation_manifest_hash": str(manifest["manifest_hash"]),
        }
    task = make_task(cfg)
    env = task.objective_function._env
    policy = _make_policy(cfg, env)

    n_contexts = len(env.instance_selector.instances)
    n_episodes = int(cfg.experiment.get("n_episodes", n_contexts))
    if n_episodes <= 0:
        raise ValueError("experiment.n_episodes must be positive.")

    rows: list[dict[str, Any]] = []
    try:
        for episode in range(n_episodes):
            result = rollout(env=env, policy=policy)
            inner_seed, task_id = result.pop("instance")
            rows.append(
                {
                    "episode": episode,
                    "policy_id": cfg.policy_id,
                    "policy_seed": cfg.seed,
                    "interaction_frequency": env.interaction_frequency,
                    "alpha": cfg.get("alpha", ""),
                    "action_index": cfg.get("static_action_index", ""),
                    "inner_seed": inner_seed,
                    "task_id": task_id,
                    **result,
                }
            )
    finally:
        env.close()

    output_path = Path(get_run_directory()) / "baseline_results.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {len(rows)} baseline episodes to {output_path}.")


if __name__ == "__main__":
    main()
