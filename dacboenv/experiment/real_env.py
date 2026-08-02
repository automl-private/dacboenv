"""Explicit non-test real DACBO environment factories for offline audits."""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from dacboenv.experiment.protocol import official_yahpo_so_task_ids

_TRAINING_CONFIG_PREFIX_BY_ACTION_SPACE = {
    "wei": "structured_ppo",
    "lcb_quantile": "lcb_quantile_ppo",
    "ucb_quantile": "ucb_quantile_ppo",
    "af_selection": "af_selection_ppo",
}
_SUPPORTED_INTERACTION_FREQUENCIES = (1, 5, 10)


@lru_cache(maxsize=12)
def _template(action_space: str, interaction_frequency: int = 1) -> DictConfig:
    try:
        training_prefix = _TRAINING_CONFIG_PREFIX_BY_ACTION_SPACE[action_space]
    except KeyError as error:
        raise ValueError(
            f"action_space must be one of {sorted(_TRAINING_CONFIG_PREFIX_BY_ACTION_SPACE)}, got {action_space!r}."
        ) from error
    if interaction_frequency not in _SUPPORTED_INTERACTION_FREQUENCIES:
        raise ValueError(
            f"interaction_frequency must be one of {_SUPPORTED_INTERACTION_FREQUENCIES}, got {interaction_frequency!r}."
        )
    training_config = f"{training_prefix}_f{interaction_frequency}"
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        return compose(
            config_name=None,
            overrides=[f"+training={training_config}", "seed=0", "outdir=/tmp/dacboenv-offline-audit"],
        )


def _build_real_structured_bbob_env(
    task_id: str,
    inner_seed: int,
    *,
    action_space: str,
    context_split: str,
    initial_design_n_configs: int | None,
    interaction_frequency: int,
):
    """Instantiate an isolated, fixed-context structured BBOB environment."""
    if not task_id.startswith("bbob/"):
        raise ValueError(f"This offline factory is BBOB-only, got {task_id!r}.")
    cfg = OmegaConf.create(OmegaConf.to_container(_template(action_space, interaction_frequency), resolve=False))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected a mapping training config.")
    audit_directory = Path(tempfile.mkdtemp(prefix=f"dacboenv-{action_space}-", dir="/tmp"))
    with open_dict(cfg):
        cfg.seed = 0
        cfg.outdir = str(audit_directory)
        cfg.dacboenv.task_ids = [task_id]
        cfg.dacboenv.inner_seeds = [int(inner_seed)]
        cfg.dacboenv.context_split = context_split
        # Offline evaluation must use the same privileged exact-reference
        # potential as the reported metrics.  The structured training template
        # intentionally defaults to a reference-free reward, so make this
        # evaluator-only convention explicit instead of inheriting it.
        cfg.dacboenv.reward_keys = ["true_regret_improvement"]
        cfg.dacboenv.optimizer_cfg.smac_cfg.scenario.output_directory = str(audit_directory / "smac3_output")
        if initial_design_n_configs is not None:
            initial_design = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.initial_design
            initial_design.n_configs = int(initial_design_n_configs)
            initial_design.max_ratio = 1.0
    return instantiate(cfg.dacboenv)


def real_structured_bbob_env(
    task_id: str,
    inner_seed: int,
    action_space: str = "wei",
    *,
    context_split: str = "validation",
    interaction_frequency: int = 1,
):
    """Create one Stage-A-equivalent BBOB environment for a paired audit.

    The caller must enforce its manifest/test guard before invoking this
    factory.  Output goes to a unique temporary directory and no trained model
    or Stage-A run root is required.
    """
    return _build_real_structured_bbob_env(
        task_id,
        inner_seed,
        action_space=action_space,
        context_split=context_split,
        initial_design_n_configs=None,
        interaction_frequency=interaction_frequency,
    )


def real_structured_bbob_smoke_env(task_id: str, inner_seed: int, action_space: str = "wei"):
    """Create a reduced-initial-design WEI environment for engineering smokes.

    The native BO budget and objective remain unchanged; only the initial
    design is reduced to two configurations to keep replay-clone smokes small.
    This factory is not the Stage-A-equivalent larger-panel factory above.
    """
    if action_space != "wei":
        raise ValueError(f"The reduced-initial-design smoke factory is WEI-only, got {action_space!r}.")
    return _build_real_structured_bbob_env(
        task_id,
        inner_seed,
        action_space="wei",
        context_split="train",
        initial_design_n_configs=2,
        interaction_frequency=1,
    )


def real_structured_yahpo_env(  # noqa: PLR0913
    task_id: str,
    inner_seed: int,
    action_space: str = "wei",
    *,
    initial_design_n_configs: int | None = None,
    context_split: str = "validation",
    budget_multiplier: float = 1.0,
    random_design_probability: float = 0.0,
    reference_table: str | Path | None = None,
    allow_incomplete_reference: bool = False,
    reference_breach_path: str | Path | None = None,
    interaction_frequency: int = 1,
    allow_sealed_test: bool = False,
):
    """Create an isolated structured environment for a non-test YAHPO smoke.

    Native budgets and maximum fidelity remain active unless a training-only
    budget multiplier is explicitly selected. Reducing only the initial-design
    count keeps vector/reset engineering smokes bounded. Official final-test
    tasks are rejected before objective construction.
    """
    if not task_id.lower().startswith("yahpo/so/"):
        raise ValueError(f"This offline factory is YAHPO-SO-only, got {task_id!r}.")
    if task_id in official_yahpo_so_task_ids() and not allow_sealed_test:
        raise ValueError(f"Offline engineering factory refuses official YAHPO test task {task_id!r}.")
    if initial_design_n_configs is not None and initial_design_n_configs <= 0:
        raise ValueError("initial_design_n_configs must be positive.")
    cfg = OmegaConf.create(OmegaConf.to_container(_template(action_space, interaction_frequency), resolve=False))
    if not isinstance(cfg, DictConfig):
        raise TypeError("Expected a mapping training config.")
    audit_directory = Path(tempfile.mkdtemp(prefix=f"dacboenv-yahpo-{action_space}-", dir="/tmp"))
    with open_dict(cfg):
        cfg.seed = 0
        cfg.outdir = str(audit_directory)
        cfg.dacboenv.task_ids = [task_id]
        cfg.dacboenv.inner_seeds = [int(inner_seed)]
        cfg.dacboenv.context_split = context_split
        cfg.dacboenv.yahpo_training_budget_multiplier = float(budget_multiplier)
        cfg.dacboenv.optimizer_cfg.smac_cfg.scenario.output_directory = str(audit_directory / "smac3_output")
        if initial_design_n_configs is not None:
            initial_design = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.initial_design
            initial_design.n_configs = int(initial_design_n_configs)
            initial_design.max_ratio = 1.0
        cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.random_design.probability = float(random_design_probability)
        if reference_table is not None:
            cfg.dacboenv.reward_keys = ["reference_regret_improvement"]
            cfg.dacboenv.reference_provider = {
                "_target_": "dacboenv.reference.ManifestReferenceProvider",
                "source": str(reference_table),
                "expected_runtime_objective_transform": "negative_accuracy",
                "expected_reporting_objective_transform": "one_minus_accuracy",
                "expected_fidelity": "fixed_maximum",
                "allow_incomplete_best_known": bool(allow_incomplete_reference),
            }
            if reference_breach_path is not None:
                cfg.dacboenv.reference_breach_path = str(reference_breach_path)
    return instantiate(cfg.dacboenv)


def real_structured_mixed_env(
    task_id: str,
    inner_seed: int,
    action_space: str = "wei",
    *,
    context_split: str = "validation",
    reference_table: str | Path | None = None,
    interaction_frequency: int = 1,
    allow_sealed_test: bool = False,
):
    """Dispatch a paired mixed context without flattening domain semantics."""
    if task_id.startswith("bbob/"):
        return real_structured_bbob_env(
            task_id,
            inner_seed,
            action_space,
            context_split=context_split,
            interaction_frequency=interaction_frequency,
        )
    if task_id.lower().startswith("yahpo/so/"):
        if reference_table is None:
            raise ValueError("Mixed YAHPO evaluation requires an explicit provenance-complete reference table.")
        return real_structured_yahpo_env(
            task_id,
            inner_seed,
            action_space,
            context_split=context_split,
            reference_table=reference_table,
            interaction_frequency=interaction_frequency,
            allow_sealed_test=allow_sealed_test,
        )
    raise ValueError(f"Unsupported mixed-evaluation task namespace: {task_id!r}.")


def real_sawei_env(
    task_id: str,
    inner_seed: int,
    action_space: str = "wei",
    *,
    context_split: str = "validation",
    output_directory: str | Path | None = None,
    initial_design_n_configs: int | None = None,
    interaction_frequency: int = 1,
    reference_table: str | Path | None = None,
    allow_incomplete_reference: bool = False,
    allow_sealed_test: bool = False,
):
    """Create a native-observation SAWEI env with evaluator-aligned reward.

    SAWEI requires UBR and WEI-term observations and a continuous WEI action,
    so it cannot reuse the frozen structured policy observation. The objective,
    full native budget, seed, initial design, and exact/best-known telescoping
    reward remain paired with the other evaluator methods. A reduced initial
    design is exposed only for bounded real engineering smokes.
    """
    is_bbob = task_id.startswith("bbob/")
    is_yahpo = task_id.lower().startswith("yahpo/so/")
    if not (is_bbob or is_yahpo):
        raise ValueError(f"Unsupported SAWEI task namespace {task_id!r}.")
    if is_yahpo and task_id in official_yahpo_so_task_ids() and not allow_sealed_test:
        raise ValueError(f"SAWEI factory refuses sealed YAHPO test task {task_id!r} without authorization.")
    if is_yahpo and reference_table is None:
        raise ValueError("YAHPO SAWEI evaluation requires a provenance-complete reference table.")
    if action_space != "wei":
        raise ValueError(f"The native SAWEI factory requires the WEI family, got {action_space!r}.")
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        cfg = compose(
            config_name=None,
            overrides=[
                "+task=dacboenv_sawei_symlog",
                "++instance_set_id=unified-sawei-evaluation",
                f"++seed={int(inner_seed)}",
                "++outdir=/tmp/dacboenv-sawei-evaluation",
            ],
        )
    if output_directory is None:
        output = Path(tempfile.mkdtemp(prefix="dacboenv-sawei-", dir="/tmp"))
    else:
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg):
        cfg.outdir = str(output)
        cfg.dacboenv.task_ids = [task_id]
        cfg.dacboenv.inner_seeds = [int(inner_seed)]
        cfg.dacboenv.context_split = context_split
        cfg.dacboenv.interaction_frequency = int(interaction_frequency)
        # Evaluation mode in DACBOEnv intentionally zeroes rewards. Keep it
        # disabled here so the shared evaluator can verify the same
        # reference-regret telescoping return used by every other method.
        cfg.dacboenv.evaluation_mode = False
        if is_bbob:
            cfg.dacboenv.reward_keys = ["true_regret_improvement"]
        else:
            cfg.dacboenv.reward_keys = ["reference_regret_improvement"]
            cfg.dacboenv.reference_provider = {
                "_target_": "dacboenv.reference.ManifestReferenceProvider",
                "source": str(reference_table),
                "expected_runtime_objective_transform": "negative_accuracy",
                "expected_reporting_objective_transform": "one_minus_accuracy",
                "expected_fidelity": "fixed_maximum",
                "allow_incomplete_best_known": bool(allow_incomplete_reference),
            }
        cfg.dacboenv.optimizer_cfg.smac_cfg.scenario.output_directory = str(output / "smac3_output")
        if initial_design_n_configs is not None:
            initial_design = cfg.dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.initial_design
            initial_design.n_configs = int(initial_design_n_configs)
            initial_design.max_ratio = 1.0
    return instantiate(cfg.dacboenv)


def real_sawei_bbob_env(
    task_id: str,
    inner_seed: int,
    action_space: str = "wei",
    *,
    context_split: str = "validation",
    output_directory: str | Path | None = None,
    initial_design_n_configs: int | None = None,
    interaction_frequency: int = 1,
):
    """Backward-compatible BBOB-only alias for the generic SAWEI factory."""
    if not task_id.startswith("bbob/"):
        raise ValueError(f"The BBOB SAWEI alias received {task_id!r}.")
    return real_sawei_env(
        task_id,
        inner_seed,
        action_space,
        context_split=context_split,
        output_directory=output_directory,
        initial_design_n_configs=initial_design_n_configs,
        interaction_frequency=interaction_frequency,
    )


__all__ = [
    "real_sawei_bbob_env",
    "real_sawei_env",
    "real_structured_bbob_env",
    "real_structured_bbob_smoke_env",
    "real_structured_mixed_env",
    "real_structured_yahpo_env",
]
