"""Preflight and orchestrate learned-policy headroom augmentation.

No run roots is a supported reporting mode.  Scientific collection is delegated
to the portable snapshot/branch CLIs after this module has frozen and hashed all
checkpoint inputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

from dacboenv.experiment.checkpoint_selection import select_checkpoint
from dacboenv.experiment.collect_snapshots import configured_structured_action_space
from dacboenv.experiment.evaluation_determinism import file_sha256

GLOBAL_STATIC_FRACTION = 0.99
NEGLIGIBLE_STATE_INCREMENT = 1e-3


def classify_behavior(
    *,
    constant_fraction: float,
    within_episode_variation: bool,
    contextual_dependence: bool,
    phase_explains_actions: bool,
    evolving_state_increment: float,
    beats_modal: bool,
    beats_marginal: bool,
    beats_nonfeedback: bool,
    captures_positive_residual: bool,
    paired_ci_excludes_zero: bool = False,
) -> str:
    """Apply the prespecified conservative behavioral classification rules."""
    if constant_fraction >= GLOBAL_STATIC_FRACTION:
        return "global_static"
    if not within_episode_variation and contextual_dependence:
        return "contextual_static"
    if phase_explains_actions and evolving_state_increment <= NEGLIGIBLE_STATE_INCREMENT:
        return "open_loop_phase_schedule"
    if all(
        (
            within_episode_variation,
            beats_modal,
            beats_marginal,
            beats_nonfeedback,
            captures_positive_residual,
            paired_ci_excludes_zero,
        )
    ):
        return "feedback_dynamic"
    return "unclassified"


def preflight_run(run_root: Path, checkpoint: str, action_family: str) -> dict[str, Any]:
    """Validate a complete checkpoint without opening any environment."""
    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing Hydra config: {config_path}")
    cfg = OmegaConf.load(config_path)
    configured_family = configured_structured_action_space(cfg)
    if configured_family != action_family:
        raise ValueError(f"action family mismatch: requested {action_family!r}, configured {configured_family!r}")
    gamma = OmegaConf.select(cfg, "optimizer.gamma")
    if gamma is None or float(gamma) != 1.0:
        raise ValueError(f"headroom protocol requires gamma=1.0, got {gamma!r}")
    reward_keys = {str(value) for value in OmegaConf.select(cfg, "dacboenv.reward_keys", default=[])}
    if not reward_keys.intersection({"reference_regret_improvement", "true_regret_improvement"}):
        raise ValueError(f"headroom protocol requires reference-regret potential reward, got {reward_keys!r}")
    selected = select_checkpoint(run_root, checkpoint)
    frequent_history = run_root / "validation" / "frequent" / "history.json"
    if not frequent_history.is_file():
        raise FileNotFoundError(f"missing completed validation history: {frequent_history}")
    training_manifest_hash = str(OmegaConf.select(cfg, "training_instances.manifest_hash", default=""))
    validation_manifest_hash = str(OmegaConf.select(cfg, "experiment.validation.manifest_hash", default=""))
    if not training_manifest_hash or not validation_manifest_hash:
        raise ValueError("training and validation manifest hashes must be recorded in the resolved Hydra config")
    protocol_path = run_root / "protocol_metadata.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"training source provenance is missing: {protocol_path}")
    protocol_metadata = json.loads(protocol_path.read_text(encoding="utf-8"))
    training_source_revision = str(
        protocol_metadata.get("scientific_source_revision")
        or protocol_metadata.get("source_revision")
        or protocol_metadata.get("code_commit")
        or ""
    )
    source_revision_status = "recorded" if training_source_revision else "unavailable_in_stageb_export"
    return {
        "run_root": str(run_root.resolve()),
        "checkpoint": selected.mode,
        "training_step": selected.training_step,
        "expected_final_step": selected.expected_final_step,
        "model_path": str(selected.model_path),
        "model_sha256": selected.model_sha256,
        "normalization_path": None if selected.normalization_path is None else str(selected.normalization_path),
        "normalization_sha256": selected.normalization_sha256,
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "outer_ppo_seed": OmegaConf.select(cfg, "seed"),
        "action_family": configured_family,
        "interaction_frequency": int(OmegaConf.select(cfg, "dacboenv.interaction_frequency")),
        "observation_schema": str(OmegaConf.select(cfg, "observation_space_id")),
        "gamma": float(gamma),
        "reward_keys": sorted(reward_keys),
        "training_manifest_hash": training_manifest_hash,
        "validation_manifest_hash": validation_manifest_hash,
        "training_source_revision": training_source_revision or "unavailable",
        "training_source_revision_status": source_revision_status,
        "protocol_metadata_path": str(protocol_path.resolve()),
        "protocol_metadata_sha256": file_sha256(protocol_path),
        "validation_history": str(frequent_history.resolve()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Preflight supplied roots and persist a reproducible augmentation plan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", action="append", type=Path, default=[])
    parser.add_argument("--action-family", choices=("wei", "af_selection"), required=True)
    parser.add_argument("--domain", choices=("yahpo", "mixed"), required=True)
    parser.add_argument(
        "--checkpoint",
        choices=("final", "full_best", "frequent_best", "best", "last"),
        default="final",
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/headroom_learned_policy_v1"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    complete, skipped = [], []
    for root in args.run_root:
        try:
            complete.append(preflight_run(root, args.checkpoint, args.action_family))
        except (FileNotFoundError, ValueError) as error:
            if not args.allow_incomplete:
                raise
            skipped.append({"run_root": str(root), "reason": str(error)})
    status = "ready" if complete else "not_executed_no_run_roots"
    payload = {
        "status": status,
        "domain": args.domain,
        "action_family": args.action_family,
        "checkpoint": args.checkpoint,
        "complete_runs": complete,
        "skipped_runs": skipped,
        "next_stage": "portable snapshot collection and identical-fingerprint action branching",
    }
    (args.output_root / "augmentation_preflight.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not complete:
        pd.DataFrame(columns=["run_root", "outer_ppo_seed", "checkpoint", "snapshot_id"]).to_parquet(
            args.output_root / "snapshots.parquet", index=False
        )
        pd.DataFrame(columns=["snapshot_id", "action", "horizon", "q_value"]).to_parquet(
            args.output_root / "branches.parquet", index=False
        )
        for name in ("policy_comparisons", "behavior_classification", "captured_headroom"):
            pd.DataFrame(columns=["status", "run_root"]).to_csv(args.output_root / f"{name}.csv", index=False)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
