"""Script for creating policy configs out of PPO training runs."""

from __future__ import annotations

import json
from pathlib import Path

from carps.utils.loggingutils import get_logger
from fire import Fire
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

from dacboenv.experiment.checkpoint_selection import SelectedCheckpoint, select_checkpoint
from dacboenv.experiment.collect_snapshots import configured_structured_action_space
from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256

logger = get_logger("CollectPPO")

_STRUCTURED_OBSERVATION_IDS = {
    "structured",
    "structured-quantile",
    "structured-af-selection",
}
_REFERENCE_FREE_REWARD_IDS = {"reference-free-improvement"}
_REFERENCE_FREE_REWARD_KEYS = {"reference_free_improvement"}
_TRUE_REGRET_REWARD_IDS = {"true-regret-improvement"}
_TRUE_REGRET_REWARD_KEYS = {"true_regret_improvement"}


def _find_run_directory(model: Path) -> Path:
    """Find the Hydra run directory that owns a possibly nested model."""
    for directory in model.parents:
        if (directory / ".hydra" / "config.yaml").is_file():
            return directory
    raise FileNotFoundError(f"Could not find .hydra/config.yaml in any parent of model {model!s}.")


def _uses_structured_training_mdp(cfg: DictConfig) -> bool:
    """Whether evaluation must retain training-time reward and reset timing."""
    observation_space_id = str(cfg.get("observation_space_id", ""))
    reward_id = str(cfg.get("reward_id", ""))
    reward_keys = {str(key) for key in cfg.dacboenv.get("reward_keys", [])}
    uses_true_regret = reward_id in _TRUE_REGRET_REWARD_IDS or bool(reward_keys.intersection(_TRUE_REGRET_REWARD_KEYS))
    uses_structured_reference_free = observation_space_id in _STRUCTURED_OBSERVATION_IDS and (
        reward_id in _REFERENCE_FREE_REWARD_IDS or bool(reward_keys.intersection(_REFERENCE_FREE_REWARD_KEYS))
    )
    return uses_true_regret or uses_structured_reference_free


def _uses_structured_reference_free_mdp(cfg: DictConfig) -> bool:
    """Compatibility alias for the former reference-free-only detector."""
    return _uses_structured_training_mdp(cfg)


def gather_selected_ppo(
    rundir: Path | str,
    checkpoint: str = "final",
    *,
    explicit_model: Path | None = None,
    explicit_normalizer: Path | None = None,
) -> list[SelectedCheckpoint]:
    """Select one canonical checkpoint per discovered Hydra run."""
    root = Path(rundir)
    run_directories = sorted(config.parent.parent for config in root.rglob(".hydra/config.yaml"))
    if checkpoint == "explicit" and len(run_directories) != 1:
        raise ValueError("explicit checkpoint mode requires a root containing exactly one Hydra run.")
    selected = [
        select_checkpoint(
            directory,
            checkpoint,
            explicit_model=explicit_model,
            explicit_normalizer=explicit_normalizer,
        )
        for directory in run_directories
    ]
    logger.info(f"Selected {len(selected)} PPO checkpoints below {root!s}.")
    return selected


def gather_trained_ppo(rundir: Path | str, model_selection: str = "final") -> list[Path]:
    """Gather one selected PPO model per Hydra training run.

    This compatibility return type exposes only model paths. Canonical modes
    are ``final``, ``full_best``, ``frequent_best``, and ``explicit``;
    deprecated aliases are resolved by :func:`select_checkpoint`.
    """
    return [item.model_path for item in gather_selected_ppo(rundir, model_selection)]


def _config_output_path(configs_path: Path, cfg: DictConfig) -> Path:
    """Build a stable optimized-policy config path from run metadata."""
    optimizer_group = "-".join(str(cfg.optimizer_id).split("-")[:3])
    return configs_path / optimizer_group / str(cfg.task_id) / f"seed{cfg.seed}.yaml"


def create_ppo_eval_configs(
    rundir: Path | str,
    configs_path: Path | str | None = None,
    model_selection: str = "final",
    inventory_path: Path | str | None = None,
    explicit_model: Path | str | None = None,
    explicit_normalizer: Path | str | None = None,
) -> None:
    """Creates PPO configs. To be called on the targeted runs directory from the DACBOENV repo root."""
    if isinstance(rundir, str):
        rundir = Path(rundir)
    selections = gather_selected_ppo(
        rundir,
        model_selection,
        explicit_model=None if explicit_model is None else Path(explicit_model),
        explicit_normalizer=None if explicit_normalizer is None else Path(explicit_normalizer),
    )
    if configs_path is None:
        configs_path = Path(__file__).parent.parent / "configs/policy/optimized/"
    elif isinstance(configs_path, str):
        configs_path = Path(configs_path)

    eval_conf = DictConfig({})
    eval_conf.optimizer = {}
    eval_conf.optimizer.policy_class = {"_target_": "dacboenv.policy.sb3_model.ModelPolicy", "_partial_": True}  # type: ignore[attr-defined]

    inventory: list[dict[str, object]] = []
    for selected in track(selections, description="Creating model config...", total=len(selections)):
        model = selected.model_path
        run_directory = selected.run_root
        cfg_fn = run_directory / ".hydra" / "config.yaml"
        loaded_cfg = OmegaConf.load(cfg_fn)
        if not isinstance(loaded_cfg, DictConfig):
            raise TypeError(f"Expected mapping at {cfg_fn!s}, got {type(loaded_cfg).__name__}.")
        cfg = loaded_cfg
        eval_conf.optimizer.policy_kwargs = {  # type: ignore[attr-defined]
            # Keep the exact absolute artifact path. SB3 also accepts a path
            # without ``.zip``, but an explicit existing file is safer when
            # CARPS evaluates from a different Hydra working directory.
            "model": str(model),
            "model_class": "stable_baselines3.PPO",
        }
        eval_conf.policy_id = f"{cfg.optimizer_id}--{cfg.task_id}--seed{cfg.seed}"
        eval_conf.optimizer_id = eval_conf.policy_id
        normalization_wrapper = selected.normalization_path
        if normalization_wrapper is not None:
            eval_conf.optimizer.policy_kwargs["normalization_wrapper"] = str(normalization_wrapper)  # type: ignore[attr-defined]
        elif bool(cfg.experiment.get("vecnormalize", False)):
            raise ValueError(f"No normalization wrapper found for model {model!s}.")
        action_family = configured_structured_action_space(cfg)
        frequency = int(OmegaConf.select(cfg, "dacboenv.interaction_frequency", default=1))
        protocol_path = run_directory / "protocol_metadata.json"
        protocol_metadata = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path.is_file() else {}
        training_source_revision = str(
            protocol_metadata.get("scientific_source_revision")
            or protocol_metadata.get("source_revision")
            or protocol_metadata.get("code_commit")
            or OmegaConf.select(cfg, "protocol_metadata.scientific_source_revision", default="unavailable")
        )
        eval_conf.policy_bundle = {
            "schema_version": "domain-neutral-policy-v1",
            "checkpoint_mode": selected.mode,
            "training_step": selected.training_step,
            "action_family": action_family,
            "action_cardinality": 5,
            "action_order": [0, 1, 2, 3, 4],
            "interaction_frequency": frequency,
            "outer_ppo_seed": int(cfg.seed),
            "observation_schema": str(cfg.get("observation_space_id", "structured")),
            "observation_schema_hash": canonical_sha256(
                {
                    "observation_space_id": str(cfg.get("observation_space_id", "structured")),
                    "action_family": action_family,
                }
            ),
            "training_source_revision": training_source_revision,
            "training_protocol_metadata_sha256": file_sha256(protocol_path) if protocol_path.is_file() else None,
            "model_sha256": selected.model_sha256,
            "normalization_sha256": selected.normalization_sha256,
            "training_config_sha256": selected.config_sha256,
        }
        yaml_str = OmegaConf.to_yaml(eval_conf)
        yaml_str = f"# @package _global_\n\n{yaml_str}"
        eval_cfg_fn = _config_output_path(configs_path, cfg)
        eval_cfg_fn.parent.mkdir(parents=True, exist_ok=True)
        with eval_cfg_fn.open("w", encoding="utf-8") as file:
            file.write(yaml_str)
        inventory.append(
            {
                "config_path": str(eval_cfg_fn.resolve()),
                "frequency": frequency,
                "action_family": action_family,
                "checkpoint_mode": selected.mode,
                "training_step": selected.training_step,
                "model_path": str(model),
                "model_sha256": selected.model_sha256,
                "normalization_path": None if normalization_wrapper is None else str(normalization_wrapper),
                "normalization_sha256": selected.normalization_sha256,
                "policy_id": str(eval_conf.policy_id),
                "seed": int(cfg.seed),
                "task_id": str(cfg.task_id),
            }
        )
    if inventory_path is not None:
        destination = Path(inventory_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"policies": inventory}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    Fire(create_ppo_eval_configs)
