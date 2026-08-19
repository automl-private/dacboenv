"""Resume the post-training full-validation stage for completed SB3 runs.

This module intentionally does not resume learning.  It loads the exact final
model and saved frequent-validation checkpoints, reruns the nominated full
panel, and writes the authoritative full-panel selection artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import dacboenv  # noqa: F401  # Register OmegaConf resolvers.
from omegaconf import DictConfig, OmegaConf, open_dict
from stable_baselines3 import DQN, PPO

from dacboenv.experiment.ppo import (
    nominate_full_validation_candidates,
    run_full_panel_validation,
)
from dacboenv.experiment.sb3_algorithms import resolve_rl_algorithm_id
from dacboenv.rl.double_dqn import DoubleDQN

_ALGORITHM_CLASSES = {
    "ppo": PPO,
    "dqn": DQN,
    "double_dqn": DoubleDQN,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Resume only the full-panel validation/selection stage of one "
            "already completed PPO/DQN/Double-DQN run."
        )
    )
    parser.add_argument("run_root", type=Path, help="Completed Hydra run directory.")
    parser.add_argument(
        "--partial-policy",
        choices=("delete", "archive", "keep"),
        default="delete",
        help=(
            "How to handle incomplete validation/full and "
            "smac3_output/validation_full directories before restarting. "
            "Default: delete, which reclaims quota."
        ),
    )
    parser.add_argument(
        "--keep-smac-output",
        action="store_true",
        help="Keep raw SMAC directories after each successfully evaluated candidate.",
    )
    parser.add_argument("--force", action="store_true", help="Rerun even when selection.json exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the candidate plan only.")
    parser.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=True)
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}.")
    return payload


def _assert_zip(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required model archive is missing or empty: {path}")
    if not zipfile.is_zipfile(path):
        raise ValueError(f"Required model archive is not a valid zip file: {path}")


def _resolve_saved_path(raw_path: str | None, *, run_root: Path, basename_fallback: Path) -> str | None:
    if raw_path is None:
        return None
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, run_root / basename_fallback / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(f"Saved checkpoint path cannot be resolved: {raw_path}")


def _load_frequent_history(run_root: Path) -> list[dict[str, Any]]:
    history_path = run_root / "validation" / "frequent" / "history.json"
    payload = _load_json(history_path)
    history = payload.get("checkpoints")
    if not isinstance(history, list) or not history:
        raise ValueError(f"Frequent validation history is empty or malformed: {history_path}")
    normalized: list[dict[str, Any]] = []
    for original in history:
        if not isinstance(original, dict):
            raise TypeError(f"Malformed checkpoint entry in {history_path}.")
        entry = dict(original)
        entry["model_path"] = _resolve_saved_path(
            str(entry["model_path"]),
            run_root=run_root,
            basename_fallback=Path("validation/frequent/checkpoints"),
        )
        entry["normalization_path"] = _resolve_saved_path(
            None if entry.get("normalization_path") is None else str(entry["normalization_path"]),
            run_root=run_root,
            basename_fallback=Path("validation/frequent/checkpoints"),
        )
        normalized.append(entry)
    return normalized


def _handle_partial(run_root: Path, policy: str) -> dict[str, Any]:
    targets = [
        run_root / "validation" / "full",
        run_root / "smac3_output" / "validation_full",
    ]
    existing = [path for path in targets if path.exists()]
    record: dict[str, Any] = {
        "policy": policy,
        "existing": [str(path) for path in existing],
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    if not existing or policy == "keep":
        return record
    if policy == "delete":
        for path in existing:
            shutil.rmtree(path)
        record["deleted"] = [str(path) for path in existing]
        return record
    archive_root = run_root / "resume_backups" / datetime.now(UTC).strftime("full_validation_%Y%m%dT%H%M%SZ")
    archive_root.mkdir(parents=True, exist_ok=False)
    moved: list[str] = []
    for path in existing:
        destination = archive_root / path.relative_to(run_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        moved.append(f"{path} -> {destination}")
    record["moved"] = moved
    return record


def _configure_quiet_logging(cfg: DictConfig) -> None:
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("dacboenv").setLevel(logging.WARNING)
    logging.getLogger("smac").setLevel(logging.WARNING)
    with open_dict(cfg):
        smac_kwargs = OmegaConf.select(cfg, "dacboenv.optimizer_cfg.smac_cfg.smac_kwargs")
        if isinstance(smac_kwargs, DictConfig):
            smac_kwargs.logging_level = False


def _load_model(cfg: DictConfig, model_path: Path):
    algorithm_id = resolve_rl_algorithm_id(cfg)
    algorithm_class = _ALGORITHM_CLASSES[algorithm_id]
    model = algorithm_class.load(str(model_path), device="cpu")
    return algorithm_id, model


def resume_full_validation(
    run_root: Path,
    *,
    partial_policy: str,
    keep_smac_output: bool,
    force: bool,
    dry_run: bool,
    quiet: bool,
) -> dict[str, Any]:
    run_root = run_root.expanduser().resolve()
    selection_path = run_root / "validation" / "full" / "selection.json"
    if selection_path.is_file() and not force:
        return {"status": "already_complete", "run_root": str(run_root), "selection_path": str(selection_path)}

    config_path = run_root / ".hydra" / "config.yaml"
    completion_path = run_root / "training_complete.json"
    model_path = run_root / "model.zip"
    protocol_path = run_root / "protocol_metadata.json"
    for required in (config_path, completion_path, model_path, protocol_path):
        if not required.is_file():
            raise FileNotFoundError(f"Required run artifact is missing: {required}")
    _assert_zip(model_path)

    completion = _load_json(completion_path)
    if completion.get("complete") is not True:
        raise ValueError(f"Run is not marked training-complete: {completion_path}")
    expected = int(completion["expected_final_timesteps"])
    reached = int(completion["num_timesteps"])
    if reached != expected:
        raise ValueError(f"Training did not reach its exact configured end: {reached} != {expected}")

    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Hydra config did not load as DictConfig: {config_path}")
    # Do not resolve the complete saved Hydra tree here. The training config
    # contains object-valued resolvers (for example, a ConfigSpace object for
    # the outer CARP-S task input space), which OmegaConf cannot materialize in
    # a normal primitive-only node tree. The original training pipeline also
    # keeps this tree unresolved and resolves only the fields it accesses.
    validation_cfg = cfg.experiment.validation
    if not bool(validation_cfg.get("enabled", True)) or not bool(validation_cfg.get("full_enabled", True)):
        raise ValueError("This run does not enable full-panel validation.")
    if bool(cfg.experiment.get("vecnormalize", False)):
        raise NotImplementedError(
            "Standalone validation resume currently supports the supplied Stage-C runs with vecnormalize=false."
        )
    if quiet:
        _configure_quiet_logging(cfg)
    with open_dict(validation_cfg):
        validation_cfg.full_keep_smac_output = bool(keep_smac_output)

    frequent_history = _load_frequent_history(run_root)
    protocol_metadata = _load_json(protocol_path)
    algorithm_id, model = _load_model(cfg, model_path)
    if int(model.num_timesteps) != expected:
        raise ValueError(
            f"Saved model timestep differs from training_complete.json: {model.num_timesteps} != {expected}"
        )

    candidates = nominate_full_validation_candidates(
        frequent_history,
        final_model_path=model_path,
        final_normalization_path=None,
        final_training_step=expected,
        top_k=int(validation_cfg.get("full_top_k", 3)),
        include_halfway=bool(validation_cfg.get("full_include_halfway", True)),
        include_final=bool(validation_cfg.get("full_include_final", True)),
        manual_steps=[int(step) for step in validation_cfg.get("full_manual_steps", [])],
    )
    plan = {
        "status": "dry_run" if dry_run else "running",
        "run_root": str(run_root),
        "algorithm_id": algorithm_id,
        "final_timestep": expected,
        "full_panel_episode_count": int(validation_cfg.full_n_eval_episodes),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "training_step": candidate.training_step,
                "nomination_reasons": list(candidate.nomination_reasons),
                "model_path": str(candidate.model_path),
            }
            for candidate in candidates
        ],
    }
    if dry_run:
        return plan

    preflight_path = run_root / "resume_full_validation_preflight.json"
    partial_record = _handle_partial(run_root, partial_policy)
    plan["partial_artifacts"] = partial_record
    preflight_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    payload = run_full_panel_validation(
        model,
        cfg,
        validation_cfg=validation_cfg,
        frequent_history=frequent_history,
        rundir=run_root,
        protocol_metadata=protocol_metadata,
        start_method=cfg.experiment.get("start_method", None),
    )
    completed = {
        **plan,
        "status": "complete",
        "selection_path": str(run_root / "validation" / "full" / "selection.json"),
        "selections": payload["selections"],
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    (run_root / "resume_full_validation_complete.json").write_text(
        json.dumps(completed, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return completed


def main() -> None:
    args = _parser().parse_args()
    try:
        payload = resume_full_validation(
            args.run_root,
            partial_policy=args.partial_policy,
            keep_smac_output=args.keep_smac_output,
            force=args.force,
            dry_run=args.dry_run,
            quiet=args.quiet,
        )
    except Exception as error:
        print(f"Full-validation resume failed: {error}", file=sys.stderr)
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
