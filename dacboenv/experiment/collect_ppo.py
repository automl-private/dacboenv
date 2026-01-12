"""Script for creating policy configs out of PPO training runs."""

from __future__ import annotations

import re
from pathlib import Path

from carps.utils.loggingutils import get_logger
from fire import Fire
from omegaconf import DictConfig, OmegaConf
from rich.progress import track

logger = get_logger("CollectPPO")


# def gather_trained_ppo(rundir: Path | str) -> list[Path]:
#     """Gathers PPO model paths."""
#     if isinstance(rundir, str):
#         rundir = Path(rundir)
#     model_zips = list(rundir.glob("**/model.zip"))
#     logger.info(f"Found {len(model_zips)} in {rundir!s}.")
#     return [p.resolve() for p in model_zips]


def gather_trained_ppo(rundir: Path | str) -> list[Path]:
    """Gathers PPO model paths recursively, preferring model.zip,
    otherwise the highest-numbered rl_model_*.zip per directory.
    """
    if isinstance(rundir, str):
        rundir = Path(rundir)

    model_paths: list[Path] = []
    pattern = re.compile(r"rl_model_(\d+)_steps\.zip")

    run_dirs = []
    for p in rundir.glob("*/*/*/*"):
        if p.is_dir():
            for _ in p.glob("*.zip"):
                run_dirs.append(p)
                break

    logger.info(f"Found {len(run_dirs)} run dirs ")

    for directory in track(run_dirs, total=len(run_dirs), description="Finding models..."):
        # 1. Prefer model.zip if it exists
        model_zip = directory / "model.zip"
        if model_zip.exists():
            model_paths.append(model_zip.resolve())
            logger.info(f"Found {model_zip}")
            continue

        # 2. Otherwise, find highest-numbered rl_model_*.zip
        candidates = []
        for file in directory.glob("rl_model_*.zip"):
            match = pattern.match(file.name)
            if match:
                candidates.append((int(match.group(1)), file))

        if candidates:
            best_model = max(candidates, key=lambda x: x[0])[1]
            logger.info(f"Found {best_model}")
            model_paths.append(best_model.resolve())

    logger.info(f"Found {len(model_paths)} trained models in {rundir!s}.")
    return model_paths


def create_ppo_eval_configs(rundir: Path | str) -> None:
    """Creates PPO configs. To be called on the targeted runs directory from the DACBOENV repo root."""
    if isinstance(rundir, str):
        rundir = Path(rundir)
    models = gather_trained_ppo(rundir)
    configs_path = Path(__file__).parent.parent / "configs/policy/optimized/"

    eval_conf = DictConfig({})
    eval_conf.optimizer = {}
    eval_conf.optimizer.policy_class = {"_target_": "dacboenv.policy.sb3_model.ModelPolicy", "_partial_": True}  # type: ignore[attr-defined]

    for model in track(models, description="Creating model config...", total=len(models)):
        cfg_fn = model.parent / ".hydra/config.yaml"
        cfg = OmegaConf.load(cfg_fn)
        eval_conf.optimizer.policy_kwargs = {  # type: ignore[attr-defined]
            "model": str(model.with_suffix("")),
            "model_class": "stable_baselines3.PPO",
        }
        eval_conf.policy_id = f"{cfg.optimizer_id}--{cfg.task_id}--seed{cfg.seed}"
        eval_conf.optimizer_id = eval_conf.policy_id
        if model.name == "model.zip":
            normalization_wrapper_fn = model.parent / "vecnormalize.pkl"
        else:
            normalization_wrapper_fn = model.parent / model.name.replace("model_", "model_vecnormalize_")
            normalization_wrapper_fn = normalization_wrapper_fn.with_suffix(".pkl")
        if normalization_wrapper_fn.is_file():
            eval_conf.optimizer.policy_kwargs["normalization_wrapper"] = str(normalization_wrapper_fn)  # type: ignore[attr-defined]
        else:
            raise ValueError(
                f"No normalization wrapper found for model {model!s}. Filename: {normalization_wrapper_fn!s}"
            )
        yaml_str = OmegaConf.to_yaml(eval_conf)
        yaml_str = f"# @package _global_\n\n{yaml_str}"
        eval_cfg_fn = configs_path / f"{'-'.join(model.parts[-5].split('-')[:3])}/{model.parts[-3]}/seed{cfg.seed}.yaml"
        eval_cfg_fn.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_cfg_fn, "w") as file:
            file.write(yaml_str)


if __name__ == "__main__":
    Fire(create_ppo_eval_configs)
