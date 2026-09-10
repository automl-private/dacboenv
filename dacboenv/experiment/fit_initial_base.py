"""Hydra supervised fitting of initial-design-conditioned terminal-loss bases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import hydra
import numpy as np

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.experiment.source_provenance import current_source_revision
from dacboenv.selective_wei.context import canonical_hash, selective_train_partition
from dacboenv.selective_wei.initial_base_model import fit_initial_base, load_initial_base, save_initial_base
from dacboenv.selective_wei.initial_collection import read_verified, write_verified
from dacboenv.selective_wei.initial_context import InitialContext
from dacboenv.selective_wei.provenance import validate_selective_branch_provenance

if TYPE_CHECKING:
    from omegaconf import DictConfig

FEATURE_GROUPS = {
    "base_metadata": ("M",),
    "base_initial_data": ("M", "D"),
    "base_initial_gp": ("M", "D", "G"),
    "base_initial_gp_ubr": ("M", "D", "G", "U"),
}


def fit(config: DictConfig) -> dict[str, Any]:
    """Claim a scientific output directory independently of Hydra bookkeeping.

    Resume supports verified completed runs only. Interrupted fitting is cheap
    but cannot silently restart into a partially written scientific artifact.
    """
    root = Path(str(config.dataset_root)).resolve()
    output = Path(str(config.output_root)).resolve()
    source_revision = current_source_revision()
    if config.get("expected_source_revision") and str(config.expected_source_revision) != source_revision:
        raise ValueError("Queued supervised-fit source changed after launch preparation.")
    recipe = {
        "train_sha256": file_sha256(root / "initial_base_train.json"),
        "dev_sha256": file_sha256(root / "initial_base_dev.json"),
        "features": str(config.features),
        "model": str(config.model),
        "seed": int(config.seed),
        "task_partition": str(config.get("task_partition", "all_train")),
        "partition_seed": int(config.get("partition_seed", 20260830)),
        "source_revision": source_revision,
    }
    claim = {"schema_version": "initial-base-fit-claim-v1", "recipe": recipe}
    if output.exists() and any(output.iterdir()):
        if not bool(config.get("resume", False)):
            raise FileExistsError("Initial-base output is nonempty; explicit compatible resume is required.")
        if read_verified(output / "run_claim.json") != claim:
            raise ValueError("Incompatible initial-base resume: data, source or fit settings changed.")
        complete = read_verified(output / "training_complete.json")
        model = load_initial_base(output / "base_bundle.json")
        if complete["semantic_hash"] != model.semantic_hash:
            raise ValueError("Initial-base completion/model identity mismatch.")
        metrics = read_verified(output / "dev_metrics.json")
        if metrics != complete["metrics"]:
            raise ValueError("Initial-base completion/metrics mismatch.")
        return metrics
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".fit.lock").open("x", encoding="utf-8"):
        try:
            write_verified(output / "run_claim.json", claim)
            return _fit_new(config)
        finally:
            (output / ".fit.lock").unlink()


def _fit_new(config: DictConfig) -> dict[str, Any]:
    """Fit using train only and score true terminal selection loss on dev."""
    root = Path(str(config.dataset_root))
    train = read_verified(root / "initial_base_train.json")
    dev = read_verified(root / "initial_base_dev.json")
    for split, dataset in (("train", train), ("dev", dev)):
        if dataset["split"] != split or dataset["target_semantics"] != "full_static_terminal_loss":
            raise ValueError("Initial-base fitter rejects local/headroom/holdout target data.")
        if any(row["recipe"]["split"] != split for row in dataset["rows"]):
            raise ValueError("Initial-base row split provenance mismatch.")
        if not dataset["rows"]:
            raise ValueError("Initial-base train and dev datasets must be nonempty.")
        metadata = dict(dataset)
        metadata["task_ids"] = [row["recipe"]["task_id"] for row in dataset["rows"]]
        validate_selective_branch_provenance(metadata, role="model_fit" if split == "train" else "dev")
    train_tasks = [row["recipe"]["task_id"] for row in train["rows"]]
    dev_tasks = [row["recipe"]["task_id"] for row in dev["rows"]]
    if set(train_tasks) & set(dev_tasks) or train["partition_manifest_hash"] != dev["partition_manifest_hash"]:
        raise ValueError("Initial-base task partitions overlap or disagree.")
    selection = str(config.get("task_partition", "all_train"))
    if selection == "model_fit":
        partition = selective_train_partition(sorted(set(train_tasks)), seed=int(config.partition_seed))
        allowed = set(cast("dict[str, list[str]]", partition["partitions"])["model_fit"])
        train["rows"] = [row for row in train["rows"] if row["recipe"]["task_id"] in allowed]
        train_tasks = [row["recipe"]["task_id"] for row in train["rows"]]
    elif selection != "all_train":
        raise ValueError("Initial base supports all_train or honest model_fit task partition.")
    contexts = [InitialContext.from_dict(row["context"]) for row in train["rows"]]
    losses = np.asarray([row["targets"]["terminal_paired_scaled"] for row in train["rows"]])
    model = fit_initial_base(
        contexts,
        losses,
        train_tasks,
        groups=FEATURE_GROUPS[str(config.features)],
        kind=str(config.model),
        seed=int(config.seed),
    )
    model.metadata.update(
        {
            "feature_settings": train["features"],
            "source_revision": current_source_revision(),
            "train_dataset_hash": canonical_hash(train),
            "partition_manifest_hash": train["partition_manifest_hash"],
            "task_partition": selection,
        }
    )
    output = Path(str(config.output_root)).resolve()
    bundle = save_initial_base(model, output)
    dev_contexts = [InitialContext.from_dict(row["context"]) for row in dev["rows"]]
    truth = np.asarray([row["targets"]["terminal_paired_scaled"] for row in dev["rows"]])
    actions = model.predict(dev_contexts).argmin(axis=1)
    selected = truth[np.arange(len(truth)), actions]
    tasks = np.asarray(dev_tasks)
    metrics = {
        "task_balanced_selected_terminal_loss": float(
            np.mean([selected[tasks == task].mean() for task in sorted(set(tasks))])
        ),
        "task_balanced_selection_regret": float(
            np.mean([(selected - truth.min(axis=1))[tasks == task].mean() for task in sorted(set(tasks))])
        ),
        "target_semantics": "full_static_terminal_loss",
        "evaluation_split": "dev",
        "model_semantic_hash": bundle["semantic_hash"],
        "holdout_opened": False,
    }
    write_verified(output / "dev_metrics.json", metrics)
    write_verified(
        output / "training_complete.json",
        {
            "status": "complete",
            "base_bundle": "base_bundle.json",
            "semantic_hash": bundle["semantic_hash"],
            "metrics": metrics,
            "dev_task_ids": sorted(set(dev_tasks)),
        },
    )
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="initial_base_fit")
def main(config: DictConfig) -> None:
    """Train one supervised base; never train an offline long-Q controller."""
    print(json.dumps(fit(config), indent=2))


if __name__ == "__main__":
    main()
