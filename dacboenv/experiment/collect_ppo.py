"""Script for creating policy configs out of PPO training runs."""

from __future__ import annotations

import re
from pathlib import Path

from carps.utils.loggingutils import get_logger
from fire import Fire
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

logger = get_logger("CollectPPO")

_CHECKPOINT_PATTERN = re.compile(r"rl_model_(\d+)_steps\.zip")
_STRUCTURED_OBSERVATION_IDS = {
    "structured",
    "structured-quantile",
    "structured-af-selection",
}
_REFERENCE_FREE_REWARD_IDS = {"reference-free-improvement"}
_REFERENCE_FREE_REWARD_KEYS = {"reference_free_improvement"}
_TRUE_REGRET_REWARD_IDS = {"true-regret-improvement"}
_TRUE_REGRET_REWARD_KEYS = {"true_regret_improvement"}


def _select_run_model(run_directory: Path) -> Path | None:
    """Select the model artifact that should represent one training run.

    Validation-selected models are preferred over the final training state.
    Runs created before validation was added retain the historical final-model
    and latest-checkpoint fallbacks.
    """
    best_model = run_directory / "validation" / "best_model.zip"
    if best_model.is_file():
        return best_model

    final_model = run_directory / "model.zip"
    if final_model.is_file():
        return final_model

    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in run_directory.glob("rl_model_*_steps.zip"):
        match = _CHECKPOINT_PATTERN.fullmatch(checkpoint.name)
        if match is not None:
            checkpoints.append((int(match.group(1)), checkpoint))
    if checkpoints:
        return max(checkpoints, key=lambda candidate: candidate[0])[1]
    return None


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


def _normalization_wrapper(model: Path, run_directory: Path) -> Path | None:
    """Locate normalization statistics corresponding to a selected model."""
    checkpoint_match = _CHECKPOINT_PATTERN.fullmatch(model.name)
    if checkpoint_match is not None:
        checkpoint_wrapper = run_directory / f"rl_model_vecnormalize_{checkpoint_match.group(1)}_steps.pkl"
        if checkpoint_wrapper.is_file():
            return checkpoint_wrapper

    candidates = [model.parent / "vecnormalize.pkl"]
    if model.parent != run_directory:
        candidates.append(run_directory / "vecnormalize.pkl")
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def gather_trained_ppo(rundir: Path | str) -> list[Path]:
    """Gather one selected PPO model per Hydra training run.

    Selection prefers ``validation/best_model.zip``, then ``model.zip``, then
    the highest-numbered ``rl_model_*_steps.zip`` compatibility checkpoint.
    """
    if isinstance(rundir, str):
        rundir = Path(rundir)

    model_paths: list[Path] = []
    run_directories = sorted(config.parent.parent for config in rundir.rglob(".hydra/config.yaml"))
    logger.info(f"Found {len(run_directories)} Hydra run dirs.")

    for directory in track(
        run_directories,
        total=len(run_directories),
        description="Finding models...",
    ):
        model = _select_run_model(directory)
        if model is not None:
            logger.info(f"Found {model}")
            model_paths.append(model.resolve())

    logger.info(f"Found {len(model_paths)} trained models in {rundir!s}.")
    return model_paths


def _config_output_path(configs_path: Path, cfg: DictConfig) -> Path:
    """Build a stable optimized-policy config path from run metadata."""
    optimizer_group = "-".join(str(cfg.optimizer_id).split("-")[:3])
    return configs_path / optimizer_group / str(cfg.task_id) / f"seed{cfg.seed}.yaml"


def create_ppo_eval_configs(
    rundir: Path | str,
    configs_path: Path | str | None = None,
) -> None:
    """Creates PPO configs. To be called on the targeted runs directory from the DACBOENV repo root."""
    if isinstance(rundir, str):
        rundir = Path(rundir)
    models = gather_trained_ppo(rundir)
    if configs_path is None:
        configs_path = Path(__file__).parent.parent / "configs/policy/optimized/"
    elif isinstance(configs_path, str):
        configs_path = Path(configs_path)

    eval_conf = DictConfig({})
    eval_conf.optimizer = {}
    eval_conf.optimizer.policy_class = {"_target_": "dacboenv.policy.sb3_model.ModelPolicy", "_partial_": True}  # type: ignore[attr-defined]

    for model in track(models, description="Creating model config...", total=len(models)):
        run_directory = _find_run_directory(model)
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
        normalization_wrapper = _normalization_wrapper(model, run_directory)
        if normalization_wrapper is not None:
            eval_conf.optimizer.policy_kwargs["normalization_wrapper"] = str(normalization_wrapper)  # type: ignore[attr-defined]
        elif bool(cfg.experiment.get("vecnormalize", False)):
            raise ValueError(f"No normalization wrapper found for model {model!s}.")
        eval_conf.dacboenv = cfg.dacboenv
        eval_conf.dacboenv.task_ids = ["${task.name}"]
        eval_conf.dacboenv.inner_seeds = ["${seed}"]
        # Structured potential-reward policies must see exactly the MDP used
        # in training: consume the initial design before the first decision
        # and keep the per-step potential difference active. Legacy
        # reference-based evaluation retains its zero-reward compatibility
        # mode.
        eval_conf.dacboenv.evaluation_mode = not _uses_structured_training_mdp(cfg)
        eval_conf.dacboenv.terminate_after_reference_performance_reached = False
        yaml_str = OmegaConf.to_yaml(eval_conf)
        yaml_str = f"# @package _global_\n\n{yaml_str}"
        eval_cfg_fn = _config_output_path(configs_path, cfg)
        eval_cfg_fn.parent.mkdir(parents=True, exist_ok=True)
        with eval_cfg_fn.open("w", encoding="utf-8") as file:
            file.write(yaml_str)


if __name__ == "__main__":
    Fire(create_ppo_eval_configs)
