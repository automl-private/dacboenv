"""Canonical, provenance-checked PPO checkpoint selection."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf

from dacboenv.experiment.evaluation_determinism import file_sha256

CheckpointMode = Literal["final", "full_best", "frequent_best", "explicit"]
CANONICAL_CHECKPOINT_MODES = frozenset({"final", "full_best", "frequent_best", "explicit"})


@dataclass(frozen=True)
class SelectedCheckpoint:
    """One model and its checkpoint-matched normalization/provenance."""

    mode: CheckpointMode
    run_root: Path
    model_path: Path
    normalization_path: Path | None
    training_step: int
    expected_final_step: int
    selection_metric: str
    selection_score: float | None
    panel_id: str | None
    panel_hash: str | None
    model_sha256: str
    normalization_sha256: str | None
    config_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible immutable selection record."""
        payload = asdict(self)
        for key in ("run_root", "model_path", "normalization_path"):
            value = payload[key]
            payload[key] = None if value is None else str(value)
        return payload


def canonical_checkpoint_mode(mode: str) -> CheckpointMode:
    """Resolve deprecated aliases without preserving ambiguous semantics."""
    if mode == "best":
        return "full_best"
    if mode == "last":
        warnings.warn("checkpoint mode 'last' is deprecated; use 'final'", DeprecationWarning, stacklevel=2)
        return "final"
    if mode not in CANONICAL_CHECKPOINT_MODES:
        raise ValueError(
            f"Unknown checkpoint mode {mode!r}; expected final, full_best, frequent_best, explicit, best, or last."
        )
    return mode  # type: ignore[return-value]


def _resolve_artifact(run_root: Path, recorded: str, *, directory: Path) -> Path:
    candidate = Path(recorded)
    if not candidate.is_absolute():
        candidate = run_root / candidate
    if candidate.is_file():
        return candidate.resolve()
    rebased = directory / Path(recorded).name
    if rebased.is_file():
        return rebased.resolve()
    raise FileNotFoundError(f"Checkpoint provenance references a missing artifact: {recorded}")


def _load_config(run_root: Path) -> tuple[DictConfig, Path, int, bool]:
    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"PPO Hydra config is missing: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(f"Expected a mapping Hydra config at {config_path}")
    raw_final = OmegaConf.select(cfg, "experiment.total_timesteps")
    if raw_final is None:
        raise ValueError(f"Hydra config does not record experiment.total_timesteps: {config_path}")
    expected_final = int(raw_final)
    if expected_final <= 0:
        raise ValueError("Configured final training timestep must be positive.")
    return cfg, config_path, expected_final, bool(OmegaConf.select(cfg, "experiment.vecnormalize", default=False))


def _frequent_entries(run_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    history_path = run_root / "validation" / "frequent" / "history.json"
    if not history_path.is_file():
        raise FileNotFoundError(f"Frequent validation history is missing: {history_path}")
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    entries = payload.get("checkpoints")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Frequent validation history has no checkpoints: {history_path}")
    trained = [entry for entry in entries if int(entry.get("training_step", 0)) > 0]
    if not trained:
        raise ValueError("Step-zero entries are never eligible checkpoint candidates.")
    return payload, trained


def _entry_artifacts(run_root: Path, entry: dict[str, Any], *, vecnormalize: bool) -> tuple[Path, Path | None]:
    directory = run_root / "validation" / "frequent" / "checkpoints"
    model = _resolve_artifact(run_root, str(entry["model_path"]), directory=directory)
    raw_normalization = entry.get("normalization_path")
    normalization = (
        None if raw_normalization is None else _resolve_artifact(run_root, str(raw_normalization), directory=directory)
    )
    if vecnormalize and normalization is None:
        raise FileNotFoundError(f"Checkpoint {model} has no matching VecNormalize state.")
    return model, normalization


def _select_full_best(  # noqa: C901, PLR0912
    run_root: Path, cfg: DictConfig, *, vecnormalize: bool
) -> tuple[Path, Path | None, int, str, float, str, str]:
    selection_path = run_root / "validation" / "full" / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(
            "full_best requires validation/full/selection.json; frequent history is not a fallback."
        )
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("trained_checkpoints_only") is not True:
        raise ValueError("Full-validation selection did not declare trained_checkpoints_only=true.")
    expected_hash = str(OmegaConf.select(cfg, "experiment.validation.full_manifest_hash", default=""))
    panel_hash = str(payload.get("manifest_hash", ""))
    if not expected_hash or panel_hash != expected_hash:
        raise ValueError("Full-validation selection manifest hash differs from the resolved training config.")
    selected = payload.get("selections", {}).get("balanced")
    if not isinstance(selected, dict):
        raise ValueError("Full validation has no authoritative balanced selection.")
    step = int(selected.get("training_step", 0))
    if step <= 0:
        raise ValueError("Full-best selection cannot be step zero.")
    candidate_id = str(selected.get("candidate_id", ""))
    results = [
        result
        for result in payload.get("results", [])
        if str(result.get("candidate_id")) == candidate_id and int(result.get("training_step", -1)) == step
    ]
    if len(results) != 1:
        raise ValueError("Full-best selection does not identify exactly one full-panel result.")
    result = results[0]
    try:
        model = _resolve_artifact(run_root, str(result["model_path"]), directory=run_root / "validation")
    except FileNotFoundError:
        authoritative_copy = run_root / "validation" / "best_balanced_model.zip"
        legacy_authoritative_copy = run_root / "validation" / "best_model.zip"
        model = next(
            (path.resolve() for path in (authoritative_copy, legacy_authoritative_copy) if path.is_file()), None
        )
        if model is None:
            raise
    raw_normalization = result.get("normalization_path")
    if raw_normalization is None:
        normalization = None
    else:
        try:
            normalization = _resolve_artifact(run_root, str(raw_normalization), directory=run_root / "validation")
        except FileNotFoundError:
            authoritative_normalization = run_root / "validation" / "best_balanced_vecnormalize.pkl"
            if not authoritative_normalization.is_file():
                raise
            normalization = authoritative_normalization.resolve()
    if vecnormalize and normalization is None:
        raise FileNotFoundError("Full-best model has no checkpoint-matched VecNormalize state.")
    return (
        model,
        normalization,
        step,
        "full_balanced_score",
        float(selected["score"]),
        str(payload.get("manifest_id", "")),
        panel_hash,
    )


def select_checkpoint(
    run_root: Path,
    mode: str,
    *,
    explicit_model: Path | None = None,
    explicit_normalizer: Path | None = None,
) -> SelectedCheckpoint:
    """Select and hash one canonical checkpoint, failing closed on ambiguity."""
    run_root = run_root.resolve()
    canonical = canonical_checkpoint_mode(mode)
    cfg, config_path, expected_final, vecnormalize = _load_config(run_root)
    panel_id: str | None = None
    panel_hash: str | None = None
    score: float | None = None
    if canonical == "explicit":
        if explicit_model is None:
            raise ValueError("explicit checkpoint mode requires --explicit-model.")
        model = explicit_model.resolve()
        normalization = None if explicit_normalizer is None else explicit_normalizer.resolve()
        if not model.is_file():
            raise FileNotFoundError(f"Explicit model is missing: {model}")
        if vecnormalize and (normalization is None or not normalization.is_file()):
            raise FileNotFoundError("An explicit checkpoint for a normalized run requires its explicit normalizer.")
        step = -1
        metric = "explicit"
    elif canonical == "full_best":
        model, normalization, step, metric, score, panel_id, panel_hash = _select_full_best(
            run_root, cfg, vecnormalize=vecnormalize
        )
    else:
        payload, entries = _frequent_entries(run_root)
        if canonical == "final":
            matches = [entry for entry in entries if int(entry["training_step"]) == expected_final]
            if len(matches) != 1:
                raise FileNotFoundError(
                    "Expected exactly one frequent checkpoint at configured final step "
                    f"{expected_final}, found {len(matches)}."
                )
            selected = matches[0]
            metric = "configured_final_training_step"
        else:
            selected = max(
                entries,
                key=lambda entry: (float(entry["scores"]["balanced"]), -int(entry["training_step"])),
            )
            metric = "frequent_balanced_score_diagnostic_only"
            score = float(selected["scores"]["balanced"])
        model, normalization = _entry_artifacts(run_root, selected, vecnormalize=vecnormalize)
        step = int(selected["training_step"])
        panel_id = str(selected.get("panel_id", payload.get("panel_id", ""))) or None
        panel_hash = str(selected.get("panel_hash", payload.get("panel_hash", ""))) or None
    return SelectedCheckpoint(
        mode=canonical,
        run_root=run_root,
        model_path=model,
        normalization_path=normalization,
        training_step=step,
        expected_final_step=expected_final,
        selection_metric=metric,
        selection_score=score,
        panel_id=panel_id,
        panel_hash=panel_hash,
        model_sha256=file_sha256(model),
        normalization_sha256=None if normalization is None else file_sha256(normalization),
        config_sha256=file_sha256(config_path),
    )


__all__ = [
    "CANONICAL_CHECKPOINT_MODES",
    "CheckpointMode",
    "SelectedCheckpoint",
    "canonical_checkpoint_mode",
    "select_checkpoint",
]
