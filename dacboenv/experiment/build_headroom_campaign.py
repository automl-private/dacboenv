"""Freeze deterministic non-test snapshot inventories for headroom analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf

from dacboenv.experiment.headroom_predictability import campaign_manifest_hash
from dacboenv.experiment.protocol import sealed_final_test_task_ids

PHASES = ("early", "middle", "late")
PHASE_FRACTIONS = {"early": 0.25, "middle": 0.5, "late": 0.75}
HISTORIES = ("default_smac", "sawei", "uniform_random", "family_static")
FAMILIES = ("wei", "af_selection")


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping manifest at {path}.")
    return payload


def _reference_task_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("references", payload) if isinstance(payload, Mapping) else payload
    if isinstance(rows, Mapping):
        return {str(task_id) for task_id in rows}
    return {str(row["task_id"]) for row in rows}


def _stratum(task_id: str) -> str:
    parts = task_id.split("/")
    return f"d{parts[1]}" if task_id.startswith("bbob/") else parts[2]


def _balanced_rows(
    *,
    manifest: Mapping[str, Any],
    domain: str,
    split: str,
    family: str,
    count: int,
    seed: int,
    reference_ids: set[str],
) -> list[dict[str, Any]]:
    forbidden = sealed_final_test_task_ids()
    tasks = [str(task) for task in manifest["task_ids"] if str(task) not in forbidden]
    if domain == "yahpo":
        tasks = [task for task in tasks if task in reference_ids]
    if not tasks:
        raise ValueError(f"No eligible {domain} {split} tasks.")
    expected_strata = {"d2", "d4"} if domain == "bbob" and split == "train" else None
    if domain == "bbob" and split == "validation":
        expected_strata = {"d4", "d8"}
    if domain == "yahpo":
        expected_strata = {"lcbench", "rbv2_glmnet", "rbv2_rpart", "rbv2_ranger", "rbv2_xgboost", "rbv2_super"}
    actual = {_stratum(task) for task in tasks}
    if expected_strata is not None and actual != expected_strata:
        raise ValueError(f"Required strata unavailable for {domain}/{split}: expected {expected_strata}, got {actual}.")

    by_stratum: dict[str, list[str]] = defaultdict(list)
    for task in sorted(tasks):
        by_stratum[_stratum(task)].append(task)
    rng = np.random.default_rng(seed)
    generated_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=count, dtype=np.uint32)
    strata = sorted(by_stratum)
    rows: list[dict[str, Any]] = []
    counters: dict[tuple[str, str, str], int] = defaultdict(int)
    for index in range(count):
        stratum = strata[index % len(strata)]
        history = HISTORIES[(index // len(strata)) % len(HISTORIES)]
        phase = PHASES[(index // (len(strata) * len(HISTORIES))) % len(PHASES)]
        cell = (stratum, history, phase)
        task_pool = by_stratum[stratum]
        task = task_pool[counters[cell] % len(task_pool)]
        counters[cell] += 1
        static_action = 0
        if family == "wei":
            static_action = (3, 4)[index % 2]
        rows.append(
            {
                "snapshot_id": f"{family}-{split}-{domain}-{index:04d}",
                "split": split,
                "domain": domain,
                "task_id": task,
                "inner_seed": int(generated_seeds[index]),
                "action_family": family,
                "interaction_frequency": 1,
                "history_generator": history,
                "history_seed": int(rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32)),
                "static_action": static_action,
                "budget_phase": phase,
                "budget_fraction": PHASE_FRACTIONS[phase],
                "source_manifest": str(manifest["id"]),
                "source_manifest_hash": str(manifest["manifest_hash"]),
            }
        )
    return rows


def build_campaign_inventory(
    *,
    bbob_train: Mapping[str, Any],
    bbob_validation: Mapping[str, Any],
    yahpo_train: Mapping[str, Any],
    yahpo_validation: Mapping[str, Any],
    reference_ids: set[str],
    seed: int = 904211,
    smoke: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build exact disjoint training and validation snapshot rows."""
    counts = {"train": 4 if smoke else 120, "validation": 4 if smoke else 60}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for family_index, family in enumerate(FAMILIES):
        for domain_index, (domain, train_manifest, validation_manifest) in enumerate(
            (("bbob", bbob_train, bbob_validation), ("yahpo", yahpo_train, yahpo_validation))
        ):
            offset = family_index * 100 + domain_index * 10
            train.extend(
                _balanced_rows(
                    manifest=train_manifest,
                    domain=domain,
                    split="train",
                    family=family,
                    count=counts["train"],
                    seed=seed + offset,
                    reference_ids=reference_ids,
                )
            )
            validation.extend(
                _balanced_rows(
                    manifest=validation_manifest,
                    domain=domain,
                    split="validation",
                    family=family,
                    count=counts["validation"],
                    seed=seed + 1000 + offset,
                    reference_ids=reference_ids,
                )
            )
    if {row["task_id"] for row in train} & {row["task_id"] for row in validation}:
        raise RuntimeError("Frozen training and validation inventories overlap by task ID.")
    return train, validation


def _write(path: Path, rows: Sequence[Mapping[str, Any]], protocol: str) -> None:
    payload = {
        "protocol_version": protocol,
        "snapshot_count": len(rows),
        "manifest_hash": campaign_manifest_hash(rows),
        "snapshots": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Build and write the frozen full or smoke snapshot inventories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=Path("artifacts"))
    parser.add_argument("--reference-table", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Write the frozen non-SAWEI subset for independently executable baseline-history branching.",
    )
    args = parser.parse_args(argv)
    root = Path("dacboenv/configs/instance_sets")
    train, validation = build_campaign_inventory(
        bbob_train=_load_manifest(root / "bbob_train.yaml"),
        bbob_validation=_load_manifest(root / "bbob_validation.yaml"),
        yahpo_train=_load_manifest(root / "yahpo_train.yaml"),
        yahpo_validation=_load_manifest(root / "yahpo_validation.yaml"),
        reference_ids=_reference_task_ids(args.reference_table),
        smoke=args.smoke,
    )
    suffix = "_smoke" if args.smoke else ""
    if args.baseline_only:
        train = [row for row in train if row["history_generator"] != "sawei"]
        validation = [row for row in validation if row["history_generator"] != "sawei"]
        suffix += "_baseline_histories"
    _write(args.output_directory / f"snapshot_manifest_train{suffix}.json", train, "headroom-predictability-v1")
    _write(
        args.output_directory / f"snapshot_manifest_validation{suffix}.json",
        validation,
        "headroom-predictability-v1",
    )
    print(json.dumps({"train": len(train), "validation": len(validation)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
