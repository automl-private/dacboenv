"""Checkpoint selection and domain-neutral PPO bundle regression tests."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from dacboenv.experiment.checkpoint_selection import canonical_checkpoint_mode, select_checkpoint
from dacboenv.experiment.collect_ppo import create_ppo_eval_configs, gather_trained_ppo
from dacboenv.experiment.evaluation_determinism import file_sha256
from omegaconf import OmegaConf


def _run(tmp_path: Path, *, final: int = 60, vecnormalize: bool = False) -> Path:
    root = tmp_path / "run"
    cfg = OmegaConf.create(
        {
            "optimizer_id": "PPO-Structured-MLP",
            "task_id": "YAHPO-only-WEI-f1",
            "seed": 2,
            "action_space_id": "wei-discrete-5",
            "observation_space_id": "structured",
            "experiment": {
                "total_timesteps": final,
                "vecnormalize": vecnormalize,
                "validation": {"full_manifest_hash": "full-panel-hash"},
            },
            "dacboenv": {
                "interaction_frequency": 1,
                "task_ids": ["yahpo/so/lcbench/126025/None"],
                "instance_selector_class": {"_target_": "dacboenv.env.instance.RandomInstanceSelector"},
                "reference_provider": {"_target_": "dacboenv.reference.ManifestReferenceProvider"},
                "reward_keys": ["reference_regret_improvement"],
            },
        }
    )
    config = root / ".hydra" / "config.yaml"
    config.parent.mkdir(parents=True)
    OmegaConf.save(cfg, config)
    directory = root / "validation" / "frequent" / "checkpoints"
    directory.mkdir(parents=True)
    entries = []
    for step, score in ((20, 0.8), (60, 0.4)):
        model = directory / f"step_{step}_model.zip"
        model.write_bytes(f"model-{step}".encode())
        normalizer = None
        if vecnormalize:
            normalizer = directory / f"step_{step}_vecnormalize.pkl"
            normalizer.write_bytes(f"normalizer-{step}".encode())
        entries.append(
            {
                "training_step": step,
                "model_path": str(model),
                "normalization_path": None if normalizer is None else str(normalizer),
                "scores": {"balanced": score},
            }
        )
    (directory.parent / "history.json").write_text(json.dumps({"checkpoints": entries}), encoding="utf-8")
    step_zero = root / "validation" / "step_zero"
    step_zero.mkdir()
    (step_zero / "history.json").write_text(
        json.dumps({"checkpoints": [{"training_step": 0, "scores": {"balanced": 99.0}}]}),
        encoding="utf-8",
    )
    return root


def test_final_is_exact_configured_step_not_filename_order(tmp_path: Path) -> None:
    root = _run(tmp_path)
    selected = select_checkpoint(root, "final")
    assert selected.training_step == 60
    assert selected.model_path.name == "step_60_model.zip"
    assert selected.selection_metric == "configured_final_training_step"


def test_frequent_best_is_explicit_and_best_never_means_frequent(tmp_path: Path) -> None:
    root = _run(tmp_path)
    selected = select_checkpoint(root, "frequent_best")
    assert selected.training_step == 20
    with pytest.raises(FileNotFoundError, match="full_best requires"):
        select_checkpoint(root, "best")


def test_last_is_deprecated_final_alias(tmp_path: Path) -> None:
    root = _run(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selected = select_checkpoint(root, "last")
    assert selected.mode == "final"
    assert any("deprecated" in str(item.message) for item in caught)
    assert canonical_checkpoint_mode("best") == "full_best"


def test_full_best_requires_authoritative_full_panel_selection(tmp_path: Path) -> None:
    root = _run(tmp_path)
    frequent = root / "validation" / "frequent" / "checkpoints"
    model = frequent / "step_20_model.zip"
    full = root / "validation" / "full"
    full.mkdir()
    (full / "selection.json").write_text(
        json.dumps(
            {
                "trained_checkpoints_only": True,
                "manifest_id": "mixed-validation-full-v1",
                "manifest_hash": "full-panel-hash",
                "selections": {"balanced": {"candidate_id": "step20", "training_step": 20, "score": 0.9}},
                "results": [
                    {
                        "candidate_id": "step20",
                        "training_step": 20,
                        "model_path": str(model),
                        "normalization_path": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    selected = select_checkpoint(root, "full_best")
    assert selected.training_step == 20
    assert selected.panel_hash == "full-panel-hash"


def test_checkpoint_specific_normalizer_and_hashes(tmp_path: Path) -> None:
    root = _run(tmp_path, vecnormalize=True)
    selected = select_checkpoint(root, "final")
    assert selected.normalization_path is not None
    assert selected.normalization_path.name == "step_60_vecnormalize.pkl"
    assert selected.normalization_sha256 == file_sha256(selected.normalization_path)


def test_final_fails_when_exact_terminal_checkpoint_is_missing(tmp_path: Path) -> None:
    root = _run(tmp_path, final=80)
    with pytest.raises(FileNotFoundError, match="configured final step 80"):
        select_checkpoint(root, "final")


def test_exported_policy_bundle_is_domain_neutral(tmp_path: Path) -> None:
    root = _run(tmp_path)
    destination = tmp_path / "policies"
    inventory = tmp_path / "inventory.json"
    create_ppo_eval_configs(root.parent, destination, "final", inventory)
    generated = OmegaConf.load(destination / "PPO-Structured-MLP" / "YAHPO-only-WEI-f1" / "seed2.yaml")
    assert "dacboenv" not in generated
    assert "reference_provider" not in OmegaConf.to_yaml(generated)
    assert "instance_selector" not in OmegaConf.to_yaml(generated)
    assert generated.policy_bundle.checkpoint_mode == "final"
    assert generated.policy_bundle.action_family == "wei"
    assert generated.policy_bundle.model_sha256 == file_sha256(
        root / "validation/frequent/checkpoints/step_60_model.zip"
    )


def test_gather_defaults_to_final(tmp_path: Path) -> None:
    root = _run(tmp_path)
    assert gather_trained_ppo(root.parent) == [(root / "validation/frequent/checkpoints/step_60_model.zip").resolve()]
