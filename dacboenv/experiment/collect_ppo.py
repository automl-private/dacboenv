"""Script for creating policy configs out of PPO training runs."""

from __future__ import annotations

from pathlib import Path

from fire import Fire
from omegaconf import DictConfig, OmegaConf


def gather_trained_ppo(rundir: Path | str) -> list[Path]:
    """Gathers PPO model paths."""
    if isinstance(rundir, str):
        rundir = Path(rundir)
    model_zips = list(rundir.glob("*/*/*/*/model.zip"))
    return [p.resolve() for p in model_zips]


def create_ppo_eval_configs(rundir: Path | str) -> None:
    """Creates PPO configs. To be called on the targeted runs directory from the DACBOENV repo root."""
    if isinstance(rundir, str):
        rundir = Path(rundir)
    models = gather_trained_ppo(rundir)
    configs_path = rundir / "../dacboenv/configs/policy/optimized/"

    eval_conf = DictConfig({})
    eval_conf.optimizer = {}
    eval_conf.optimizer.policy_class = {"_target_": "dacboenv.env.policy.ModelPolicy", "_partial_": True}  # type: ignore[attr-defined]

    for model in models:
        eval_conf.optimizer.policy_kwargs = {  # type: ignore[attr-defined]
            "model": str(model.with_suffix("")),
            "model_class": "stable_baselines3.PPO",
        }
        eval_conf.optimizer_id = f"{model.parts[-5]}-{model.parts[-3]}-{model.parts[-2]}"
        yaml_str = OmegaConf.to_yaml(eval_conf)
        yaml_str = f"# @package _global_\n\n{yaml_str}"
        eval_cfg_fn = (
            configs_path / f"{'-'.join(model.parts[-5].split('-')[:3])}/{model.parts[-3]}/seed_{model.parts[-2]}.yaml"
        )
        eval_cfg_fn.parent.mkdir(parents=True, exist_ok=True)
        with open(eval_cfg_fn, "w") as file:
            file.write(yaml_str)


if __name__ == "__main__":
    Fire(create_ppo_eval_configs)
