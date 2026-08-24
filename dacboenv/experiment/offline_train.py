"""Hydra entry point for supervised branch-Q and conservative offline Q learning."""

from __future__ import annotations

import copy
import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.offline.branch_dataset import BranchDataset
from dacboenv.offline.dataset import BehaviorDataset
from dacboenv.offline.deployment import (
    DeploymentHead,
    DeploymentSelectionState,
    deployment_head_for_mode,
    deployment_selection_eligible,
)
from dacboenv.offline.losses import (
    behavior_cloning_loss,
    centered_huber_pairwise_loss,
    double_dqn_targets,
    offline_td_cql_loss,
)
from dacboenv.offline.models.shared_dueling_q import OfflineQModelConfig, OfflineQNetwork
from dacboenv.offline.normalization import ObservationNormalizer
from dacboenv.offline.sampler import (
    BranchBatchSampler,
    BranchSamplerConfig,
    HierarchicalBatchSampler,
    SamplerConfig,
)

BRANCH_TIE_TOLERANCE = 1e-3
MIN_BBOB_TASK_PARTS = 2


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write replay-safe line-delimited diagnostics atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    value = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _seed_hierarchy(seed: int) -> dict[str, int]:
    names = ("model", "behavior_batches", "branch_batches", "development", "target_updates")
    children = np.random.SeedSequence(seed).spawn(len(names))
    return {name: int(child.generate_state(1, dtype=np.uint32)[0]) for name, child in zip(names, children, strict=True)}


def _normalizer(root: Path) -> ObservationNormalizer:
    value = json.loads((root / "normalization_schema.json").read_text(encoding="utf-8"))
    return ObservationNormalizer.from_dict(value)


def _scientific_config(config: DictConfig) -> dict[str, Any]:
    """Return the resolved run identity excluding the checkpoint location used to resume it."""
    payload = OmegaConf.to_container(config, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved offline training config must be a mapping.")
    training = payload.get("offline_training")
    if isinstance(training, dict):
        training["resume_from"] = None
    return cast("dict[str, Any]", payload)


def _branch_indices(dataset: BranchDataset, domain: str) -> np.ndarray:
    if domain == "mixed":
        return np.arange(len(dataset), dtype=np.int64)
    domain_id = 1 if domain == "yahpo" else 0
    return np.flatnonzero(dataset.arrays["domain_id"] == domain_id).astype(np.int64)


def _validate_task_boundaries(
    behavior: BehaviorDataset,
    branch_train: BranchDataset,
    branch_dev: BranchDataset,
    manifest: dict[str, Any],
) -> None:
    """Bind every training input to the frozen task-disjoint split manifest."""
    expected_train = set(map(str, manifest["task_splits"]["train"]))
    expected_dev = set(map(str, manifest["task_splits"]["dev"]))
    expected_holdout = set(map(str, manifest["task_splits"]["holdout"]))
    behavior_tasks = set(map(str, np.unique(behavior.arrays["task_id"])))
    branch_train_tasks = set(map(str, np.unique(branch_train.arrays["task_id"])))
    branch_dev_tasks = set(map(str, np.unique(branch_dev.arrays["task_id"])))
    if behavior_tasks != expected_train:
        raise ValueError("Behavior training tasks do not exactly match the frozen train split.")
    if not branch_train_tasks or not branch_train_tasks <= expected_train:
        raise ValueError("Branch training states are not a non-empty subset of the frozen train tasks.")
    if not branch_dev_tasks or not branch_dev_tasks <= expected_dev:
        raise ValueError("Branch development states are not a non-empty subset of the frozen dev tasks.")
    if branch_train_tasks & branch_dev_tasks or (branch_train_tasks | branch_dev_tasks) & expected_holdout:
        raise ValueError("Offline branch inputs overlap each other or the sealed holdout.")


def _branch_dev_metrics(
    model: OfflineQNetwork,
    dataset: BranchDataset,
    indices: np.ndarray,
    device: torch.device,
    nonfeedback_actions: np.ndarray,
    global_static_action: int,
    deployment_head: DeploymentHead,
) -> dict[str, float]:
    """Score deployed and auxiliary branch heads against true fixed-action Q5."""
    if indices.size == 0:
        raise ValueError("Development branch dataset has no selected-domain states.")
    batch = dataset.torch_batch(indices, device)
    model.eval()
    with torch.no_grad():
        deployment_q = model(batch["global_state"], batch["action_features"], head=deployment_head)
        branch_q = model(batch["global_state"], batch["action_features"], head="branch_q5")
    target = batch["q5"]
    comparator = torch.as_tensor(nonfeedback_actions, dtype=torch.long, device=device)
    static_value = target.gather(1, comparator.unsqueeze(1)).squeeze(1)
    global_value = target[:, global_static_action]
    gaps = batch["gap_q5"]
    oracle = target.max(dim=1).values

    def score_head(q: torch.Tensor, prefix: str) -> tuple[dict[str, float], dict[str, np.ndarray]]:
        selected = q.argmax(dim=1)
        selected_value = target.gather(1, selected.unsqueeze(1)).squeeze(1)
        tie_aware = batch["tie_mask_q5"].gather(1, selected.unsqueeze(1)).float().squeeze(1)
        pair_correct: list[torch.Tensor] = []
        for left in range(target.shape[1]):
            for right in range(left + 1, target.shape[1]):
                difference = target[:, left] - target[:, right]
                valid = difference.abs() > BRANCH_TIE_TOLERANCE
                if valid.any():
                    predicted = q[:, left] - q[:, right]
                    pair_correct.append((predicted[valid].sign() == difference[valid].sign()).float())
        q_rank = q.argsort(dim=1).argsort(dim=1).float()
        target_rank = target.argsort(dim=1).argsort(dim=1).float()
        centered_q_rank = q_rank - q_rank.mean(dim=1, keepdim=True)
        centered_target_rank = target_rank - target_rank.mean(dim=1, keepdim=True)
        spearman = (centered_q_rank * centered_target_rank).sum(dim=1) / (
            centered_q_rank.square().sum(dim=1).sqrt() * centered_target_rank.square().sum(dim=1).sqrt()
        ).clamp_min(1e-12)
        values = {
            f"{prefix}_selected_value": float(selected_value.mean().cpu()),
            f"{prefix}_selected_action_regret": float((oracle - selected_value).mean().cpu()),
            f"{prefix}_tie_aware_top1": float(tie_aware.mean().cpu()),
            f"{prefix}_gap_weighted_top1": float((tie_aware * gaps).sum().cpu() / gaps.sum().clamp_min(1e-12).cpu()),
            f"{prefix}_pairwise_accuracy": (
                float(torch.cat(pair_correct).mean().cpu()) if pair_correct else float("nan")
            ),
            f"{prefix}_spearman": float(spearman.mean().cpu()),
            f"{prefix}_learned_minus_nonfeedback": float((selected_value - static_value).mean().cpu()),
        }
        if prefix == "deployment":
            values[f"{prefix}_learned_minus_global_static"] = float((selected_value - global_value).mean().cpu())
        grouped = {
            f"{prefix}_selected_value": selected_value.detach().cpu().numpy(),
            f"{prefix}_selected_action_regret": (oracle - selected_value).detach().cpu().numpy(),
            f"{prefix}_learned_minus_nonfeedback": (selected_value - static_value).detach().cpu().numpy(),
        }
        return values, grouped

    deployment_metrics, deployment_grouped = score_head(deployment_q, "deployment")
    branch_metrics, branch_grouped = score_head(branch_q, "branch_q5")
    result = {**deployment_metrics, **branch_metrics, "states": float(indices.size)}
    # Backward-compatible aliases explicitly mean deployment behavior.
    for name in (
        "selected_value",
        "selected_action_regret",
        "tie_aware_top1",
        "gap_weighted_top1",
        "pairwise_accuracy",
        "spearman",
        "learned_minus_nonfeedback",
        "learned_minus_global_static",
    ):
        result[name] = result[f"deployment_{name}"]
    task_ids = np.asarray(dataset.arrays["task_id"])[indices]
    grouped_values = {**deployment_grouped, **branch_grouped}
    unique_tasks = np.unique(task_ids)
    if unique_tasks.size >= 3:  # noqa: PLR2004 - minimum meaningful grouped resample
        generator = np.random.default_rng(20260824)
        for name, values in grouped_values.items():
            task_means = np.asarray([values[task_ids == task].mean() for task in unique_tasks], dtype=np.float64)
            draws = generator.choice(task_means, size=(500, len(task_means)), replace=True).mean(axis=1)
            result[f"{name}_task_bootstrap_lower"] = float(np.quantile(draws, 0.025))
            result[f"{name}_task_bootstrap_upper"] = float(np.quantile(draws, 0.975))
    return result


def _context_key(task_id: str, domain_id: int, scenario_id: int) -> str:
    """Return one deployable static-context key without exact task identity."""
    if domain_id == 1:
        return f"yahpo:scenario:{scenario_id}"
    parts = task_id.split("/")
    if len(parts) < MIN_BBOB_TASK_PARTS:
        raise ValueError(f"Malformed BBOB task ID {task_id!r}.")
    return f"bbob:dimension:{parts[1]}"


def _fit_contextual_nonfeedback(train: BranchDataset, dev: BranchDataset) -> tuple[dict[str, int], np.ndarray]:
    """Fit context-static actions on training branches and apply them to dev."""
    grouped: dict[str, list[int]] = {}
    for index in range(len(train)):
        key = _context_key(
            str(train.arrays["task_id"][index]),
            int(train.arrays["domain_id"][index]),
            int(train.arrays["scenario_id"][index]),
        )
        grouped.setdefault(key, []).append(index)
    global_action = int(np.asarray(train.arrays["q5"]).mean(axis=0).argmax())
    registry = {
        key: int(np.asarray(train.arrays["q5"])[np.asarray(indices, dtype=np.int64)].mean(axis=0).argmax())
        for key, indices in grouped.items()
    }
    dev_actions = np.asarray(
        [
            registry.get(
                _context_key(
                    str(dev.arrays["task_id"][index]),
                    int(dev.arrays["domain_id"][index]),
                    int(dev.arrays["scenario_id"][index]),
                ),
                global_action,
            )
            for index in range(len(dev))
        ],
        dtype=np.int64,
    )
    return {"global": global_action, **registry}, dev_actions


def _apply_negative_control(
    batch: dict[str, torch.Tensor],
    control: str,
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Apply a prespecified diagnostic corruption to one training batch."""
    if control == "none":
        return batch
    result = dict(batch)
    if control == "shuffled_global_states":
        order = torch.randperm(len(batch["global_state"]), generator=generator).to(batch["global_state"].device)
        result["global_state"] = batch["global_state"][order]
    elif control == "action_row_mean":
        mean = batch["action_features"].mean(dim=1, keepdim=True)
        result["action_features"] = mean.expand_as(batch["action_features"])
    elif control == "mismatched_action_rows":
        permutation = torch.randperm(batch["action_features"].shape[1], generator=generator).to(
            batch["action_features"].device
        )
        result["action_features"] = batch["action_features"][:, permutation]
    elif control == "shuffled_branch_labels":
        order = torch.randperm(len(batch["q5"]), generator=generator).to(batch["q5"].device)
        result["q5"] = batch["q5"][order]
    else:
        raise ValueError(f"Unknown offline negative control {control!r}.")
    return result


def _checkpoint_payload(  # noqa: PLR0913
    *,
    model: OfflineQNetwork,
    target: OfflineQNetwork,
    optimizer: torch.optim.Optimizer,
    update: int,
    config: DictConfig,
    seeds: dict[str, int],
    sampler: HierarchicalBatchSampler,
    branch_sampler: BranchBatchSampler,
    diagnostic_generator: torch.Generator,
    selection: DeploymentSelectionState,
    history: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "dacbo-offline-q-checkpoint-v2",
        "update": update,
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "target_state": target.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "resolved_config": OmegaConf.to_container(config, resolve=True),
        "seed_hierarchy": seeds,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "sampler_rng_state": sampler.rng.bit_generator.state,
        "branch_sampler_rng_state": branch_sampler.rng.bit_generator.state,
        "diagnostic_generator_state": diagnostic_generator.get_state(),
        "best_branch_dev_value": selection.best_value,
        "deployment_selection": selection.to_dict(),
        "history": history,
        "provenance": provenance,
    }


def train(config: DictConfig) -> dict[str, Any]:  # noqa: C901, PLR0912, PLR0915
    """Train one bounded or substantive offline configuration."""
    if bool(config.offline_training.get("replay_prefill_only", False)):
        raise ValueError(
            "O1 replay_prefill_yahpo is an online DoubleDQN experiment. "
            "Use scripts/otus/otus_finetune_offline_ddqn.sh instead of offline_train."
        )
    root = Path(str(config.offline_dataset.root)).resolve()
    output = Path(str(config.output_root)).resolve()
    if not (root / "final_offline_dataset_manifest.json").is_file():
        raise FileNotFoundError(f"Finalized offline dataset is missing under {root}.")
    manifest_path = root / "final_offline_dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalizer = _normalizer(root)
    behavior = BehaviorDataset(root / "behavior_train.npz", normalizer=normalizer)
    branch_train_path = (
        Path(str(config.offline_dataset.branch_train)).resolve()
        if config.offline_dataset.branch_train
        else root / "initial_counterfactual_train.npz"
    )
    branch_dev_path = (
        Path(str(config.offline_dataset.branch_dev)).resolve()
        if config.offline_dataset.branch_dev
        else root / "initial_counterfactual_dev.npz"
    )
    branch_train = BranchDataset(branch_train_path, normalizer=normalizer)
    branch_dev = BranchDataset(branch_dev_path, normalizer=normalizer)
    _validate_task_boundaries(behavior, branch_train, branch_dev, manifest)
    domain = str(config.offline_training.domain)
    domain_id = None if domain == "mixed" else 1 if domain == "yahpo" else 0
    eligible_behavior_indices = (
        np.arange(len(behavior), dtype=np.int64)
        if domain_id is None
        else np.flatnonzero(behavior.arrays["domain_id"] == domain_id)
    )
    branch_train_indices = _branch_indices(branch_train, domain)
    branch_dev_indices = _branch_indices(branch_dev, domain)
    seeds = _seed_hierarchy(int(config.seed))
    random.seed(seeds["model"])
    torch.manual_seed(seeds["model"])
    diagnostic_generator = torch.Generator(device="cpu").manual_seed(seeds["development"])
    device = torch.device(str(config.device))
    model_values = OmegaConf.to_container(config.offline_model, resolve=True)
    if not isinstance(model_values, dict):
        raise TypeError("offline_model must resolve to a mapping.")
    model_cfg = OfflineQModelConfig(**cast("dict[str, Any]", model_values))
    model = OfflineQNetwork(model_cfg).to(device)
    target = copy.deepcopy(model).to(device)
    target.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.offline_training.learning_rate))
    sampler_values = OmegaConf.to_container(config.offline_training.sampler, resolve=True)
    if not isinstance(sampler_values, dict):
        raise TypeError("offline_training.sampler must resolve to a mapping.")
    if domain != "mixed":
        sampler_values["bbob_fraction"] = 0.0 if domain == "yahpo" else 1.0
    sampler_cfg = SamplerConfig(**cast("dict[str, Any]", sampler_values))
    sampler = HierarchicalBatchSampler(
        behavior,
        sampler_cfg,
        seeds["behavior_batches"],
        eligible_indices=eligible_behavior_indices,
    )
    branch_sampler_cfg = BranchSamplerConfig(
        batch_size=int(config.offline_training.branch_batch_size),
        bbob_fraction=0.5 if domain == "mixed" else 0.0 if domain == "yahpo" else 1.0,
        high_gap_fraction=float(config.offline_training.get("high_gap_branch_fraction", 0.25)),
        high_gap_threshold=float(config.offline_algorithm.tie_tolerance),
    )
    branch_sampler = BranchBatchSampler(
        branch_train,
        branch_sampler_cfg,
        seeds["branch_batches"],
        eligible_indices=branch_train_indices,
    )
    nonfeedback_registry, nonfeedback_dev_actions = _fit_contextual_nonfeedback(branch_train, branch_dev)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "resolved_config.yaml"
    OmegaConf.save(config, resolved_path)
    output_normalizer = output / "normalization_schema.json"
    shutil.copyfile(root / "normalization_schema.json", output_normalizer)
    provenance = {
        "dataset_manifest_hash": manifest["manifest_hash"],
        "task_split_hash": manifest["task_split_hash"],
        "behavior_train_sha256": file_sha256(root / "behavior_train.npz"),
        "branch_train_sha256": file_sha256(branch_train_path),
        "branch_dev_sha256": file_sha256(branch_dev_path),
        "normalizer_sha256": file_sha256(root / "normalization_schema.json"),
        "resolved_config_hash": canonical_sha256(_scientific_config(config)),
        "repository_revision": current_source_revision(Path(__file__).resolve().parents[2]),
        "holdout_opened": False,
        "model_schema_hash": canonical_sha256(asdict(model.config)),
    }
    provenance["nonfeedback_registry_hash"] = canonical_sha256(nonfeedback_registry)
    _atomic_json(output / "training_fitted_nonfeedback_registry.json", nonfeedback_registry)
    history: list[dict[str, Any]] = []
    start = 0
    mode = str(config.offline_training.get("algorithm_mode", config.offline_algorithm.mode))
    deployment_head = deployment_head_for_mode(mode)
    selection = DeploymentSelectionState(deployment_head=deployment_head)
    resume = str(config.offline_training.resume_from or "")
    if resume:
        saved = torch.load(Path(resume), map_location=device, weights_only=False)
        if saved["provenance"] != provenance or saved["model_config"] != asdict(model.config):
            raise ValueError("Resume checkpoint dataset/model provenance does not match this run.")
        model.load_state_dict(saved["model_state"])
        target.load_state_dict(saved["target_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        random.setstate(saved["python_rng_state"])
        torch.set_rng_state(saved["torch_rng_state"])
        sampler.rng.bit_generator.state = saved["sampler_rng_state"]
        branch_sampler.rng.bit_generator.state = saved["branch_sampler_rng_state"]
        diagnostic_generator.set_state(saved["diagnostic_generator_state"])
        history = list(saved["history"])
        start = int(saved["update"])
        saved_selection = saved.get("deployment_selection")
        if saved_selection is None:
            if mode != "branch_q5_only":
                raise ValueError(
                    "Legacy offline checkpoint used branch_q5 selection for a long_q deployment mode and cannot resume."
                )
            selection.best_value = float(saved["best_branch_dev_value"])
            selection.selected_update = start if np.isfinite(selection.best_value) else None
            selection.eligible_checkpoint_seen = selection.selected_update is not None
        else:
            if saved_selection.get("deployment_head") != deployment_head:
                raise ValueError("Resume checkpoint deployment head disagrees with the resolved algorithm mode.")
            if saved_selection.get("checkpoint_selection_metric") != selection.metric:
                raise ValueError("Resume checkpoint uses an incompatible development-selection metric.")
            selection.best_value = float(saved_selection["checkpoint_selection_value"])
            selection.selected_update = (
                None if saved_selection.get("selected_update") is None else int(saved_selection["selected_update"])
            )
            selection.eligible_checkpoint_seen = bool(saved_selection["deployment_selection_eligible"])
            selection.patience_counter = int(saved_selection["patience_counter"])
    negative_control = str(config.offline_training.get("negative_control", "none"))
    maximum = int(config.offline_training.maximum_updates)
    pretrain_updates = int(config.offline_training.branch_pretrain_updates)
    patience = int(config.offline_training.get("patience", 0))
    writer = SummaryWriter(output / "tensorboard")
    best_path = output / "best_branch_dev.pt"
    last_update = start
    try:
        for update in range(start + 1, maximum + 1):
            last_update = update
            branch_phase = mode == "branch_q5_only" or (
                mode in {"branch_pretrain_then_fqi", "branch_pretrain_then_cql"} and update <= pretrain_updates
            )
            optimizer.zero_grad()
            if branch_phase or mode == "joint_branch_cql":
                branch_indices = branch_sampler.sample()
                branch_batch = branch_train.torch_batch(branch_indices, device)
                branch_batch = _apply_negative_control(branch_batch, negative_control, diagnostic_generator)
                branch_prediction = model(
                    branch_batch["global_state"], branch_batch["action_features"], head="branch_q5"
                )
                branch_loss = centered_huber_pairwise_loss(
                    branch_prediction,
                    branch_batch["q5"],
                    branch_batch["valid_action_mask"],
                    tie_tolerance=float(config.offline_algorithm.tie_tolerance),
                    regression_weight=float(config.offline_algorithm.branch_regression_weight),
                    ranking_weight=float(config.offline_algorithm.branch_ranking_weight),
                    gap_weighted=bool(config.offline_algorithm.gap_weighted),
                    duplicate_groups=branch_batch["duplicate_groups"],
                )
                q10_loss = None
                q10_weight = float(config.offline_algorithm.get("q10_weight", 0.0))
                if branch_train.has_q10 and q10_weight > 0:
                    q10_prediction = model(
                        branch_batch["global_state"], branch_batch["action_features"], head="branch_q10"
                    )
                    q10_loss = centered_huber_pairwise_loss(
                        q10_prediction,
                        branch_batch["q10"],
                        branch_batch["valid_action_mask"],
                        tie_tolerance=float(config.offline_algorithm.tie_tolerance),
                        regression_weight=q10_weight,
                        ranking_weight=q10_weight * float(config.offline_algorithm.branch_ranking_weight),
                        gap_weighted=bool(config.offline_algorithm.gap_weighted),
                        duplicate_groups=branch_batch["duplicate_groups"],
                    )
            else:
                branch_loss = None
                q10_loss = None
            if not branch_phase:
                indices = sampler.sample()
                batch = behavior.torch_batch(indices, device)
                batch = _apply_negative_control(batch, negative_control, diagnostic_generator)
                q = model(batch["global_state"], batch["action_features"], head="long_q")
                with torch.no_grad():
                    online_next = model(batch["next_global_state"], batch["next_action_features"], head="long_q")
                    target_next = target(batch["next_global_state"], batch["next_action_features"], head="long_q")
                    td_target = double_dqn_targets(batch["reward"], batch["done"], online_next, target_next, gamma=1.0)
                if mode == "behavior_cloning":
                    cloning = behavior_cloning_loss(q, batch["action"])
                    td_loss = None
                else:
                    cloning = None
                    coefficient = (
                        float(config.offline_training.get("cql_coefficient", config.offline_algorithm.cql_coefficient))
                        if "cql" in mode
                        else 0.0
                    )
                    td_loss = offline_td_cql_loss(q, batch["action"], td_target, cql_coefficient=coefficient)
            else:
                td_loss = None
                cloning = None
            total = torch.zeros((), device=device)
            if branch_loss is not None:
                total = total + branch_loss.total
            if q10_loss is not None:
                total = total + q10_loss.total
            if td_loss is not None:
                total = total + td_loss.total
            if cloning is not None:
                total = total + cloning
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.offline_training.gradient_clip))
            optimizer.step()
            if update % int(config.offline_training.target_update_interval) == 0:
                target.load_state_dict(model.state_dict())
            record: dict[str, Any] = {"update": update, "train/total_loss": float(total.detach().cpu())}
            if branch_loss is not None:
                record.update(
                    {
                        "train/branch_regression_loss": float(branch_loss.regression.detach().cpu()),
                        "train/branch_ranking_loss": float(branch_loss.ranking.detach().cpu()),
                    }
                )
                record.update({f"batch/branch_{key}": value for key, value in branch_sampler.last_composition.items()})
            if q10_loss is not None:
                record.update(
                    {
                        "train/q10_regression_loss": float(q10_loss.regression.detach().cpu()),
                        "train/q10_ranking_loss": float(q10_loss.ranking.detach().cpu()),
                    }
                )
            if td_loss is not None:
                td_error = (td_loss.target - td_loss.data_q).abs().detach().cpu().numpy()
                record.update(
                    {
                        "train/td_loss": float(td_loss.td.detach().cpu()),
                        "train/cql_loss": float(td_loss.cql.detach().cpu()),
                        "train/q_mean": float(q.mean().detach().cpu()),
                        "train/q_std": float(q.std().detach().cpu()),
                        "train/q_margin": float(
                            (q.topk(2, dim=1).values[:, 0] - q.topk(2, dim=1).values[:, 1]).mean().detach().cpu()
                        ),
                        "train/target_q_mean": float(td_loss.target.mean().detach().cpu()),
                        "train/td_error_mean": float(td_error.mean()),
                        "train/td_error_p90": float(np.quantile(td_error, 0.9)),
                    }
                )
                record.update({f"batch/{key}": value for key, value in sampler.last_composition.items()})
            if cloning is not None:
                record["train/behavior_cloning_loss"] = float(cloning.detach().cpu())
            if update % int(config.offline_training.dev_interval) == 0 or update == maximum:
                metrics = _branch_dev_metrics(
                    model,
                    branch_dev,
                    branch_dev_indices,
                    device,
                    nonfeedback_dev_actions[branch_dev_indices],
                    int(nonfeedback_registry["global"]),
                    deployment_head,
                )
                record.update({f"dev/{key}": value for key, value in metrics.items()})
                selection_eligible = deployment_selection_eligible(mode, update, pretrain_updates)
                record["dev/deployment_selection_eligible"] = float(selection_eligible)
                record["dev/deployment_head"] = deployment_head
                selected = selection.consider(
                    value=metrics["deployment_selected_value"],
                    update=update,
                    eligible=selection_eligible,
                )
                if selected:
                    _atomic_torch(
                        best_path,
                        _checkpoint_payload(
                            model=model,
                            target=target,
                            optimizer=optimizer,
                            update=update,
                            config=config,
                            seeds=seeds,
                            sampler=sampler,
                            branch_sampler=branch_sampler,
                            diagnostic_generator=diagnostic_generator,
                            selection=selection,
                            history=[*history, record],
                            provenance=provenance,
                        ),
                    )
            history.append(record)
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(key, value, update)
            if update % int(config.offline_training.checkpoint_interval) == 0:
                _atomic_torch(
                    output / f"step_{update:05d}.pt",
                    _checkpoint_payload(
                        model=model,
                        target=target,
                        optimizer=optimizer,
                        update=update,
                        config=config,
                        seeds=seeds,
                        sampler=sampler,
                        branch_sampler=branch_sampler,
                        diagnostic_generator=diagnostic_generator,
                        selection=selection,
                        history=history,
                        provenance=provenance,
                    ),
                )
            if patience > 0 and selection.patience_counter >= patience:
                break
    finally:
        writer.close()
    if not selection.eligible_checkpoint_seen or selection.selected_update is None or not best_path.is_file():
        raise RuntimeError(
            "Training ended without an eligible deployment-head development checkpoint; "
            "increase maximum_updates beyond branch pretraining and include a dev evaluation."
        )
    final_payload = _checkpoint_payload(
        model=model,
        target=target,
        optimizer=optimizer,
        update=last_update,
        config=config,
        seeds=seeds,
        sampler=sampler,
        branch_sampler=branch_sampler,
        diagnostic_generator=diagnostic_generator,
        selection=selection,
        history=history,
        provenance=provenance,
    )
    _atomic_torch(output / "final.pt", final_payload)
    _atomic_json(output / "training_history.json", history)
    _atomic_jsonl(output / "training_history.jsonl", history)
    final_checkpoint = output / "final.pt"
    summary = {
        "schema_version": "dacbo-offline-training-run-v2",
        "status": "complete",
        "update": last_update,
        "stopped_early": last_update < maximum,
        "mode": mode,
        "domain": domain,
        "model_parameter_count": model.parameter_count,
        "model_schema_hash": canonical_sha256(asdict(model.config)),
        "normalizer_path": str(output_normalizer),
        "normalizer_sha256": file_sha256(output_normalizer),
        "seed_hierarchy": seeds,
        "development_metric_scope": (
            "initial-state engineering metric"
            if branch_dev.metadata.get("component") == "initial_same_state_q5"
            else "mid-run same-state branch development"
        ),
        "best_branch_dev_value": selection.best_value,
        "deployment_head": deployment_head,
        "checkpoint_selection_head": deployment_head,
        "checkpoint_selection_metric": selection.metric,
        "checkpoint_selection_value": selection.best_value,
        "selected_update": selection.selected_update,
        "deployment_selection_eligible": selection.eligible_checkpoint_seen,
        "patience_counter": selection.patience_counter,
        "best_branch_dev_path": str(best_path),
        "final_path": str(final_checkpoint),
        "final_sha256": file_sha256(final_checkpoint),
        "best_branch_dev_sha256": file_sha256(best_path),
        "provenance": provenance,
    }
    _atomic_json(output / "training_complete.json", summary)
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="offline_train")  # type: ignore[untyped-decorator]
def main(config: DictConfig) -> None:
    """Train from finalized explicit dataset paths without opening holdout."""
    print(json.dumps(train(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
