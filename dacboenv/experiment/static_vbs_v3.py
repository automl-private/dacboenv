"""Build and execute the bounded paired static-WEI protocol-v3 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dacboenv.experiment.collect_snapshots import completed_evaluations
from dacboenv.experiment.evaluation_determinism import runhistory_fingerprints
from dacboenv.experiment.real_env import real_headroom_train_env, real_headroom_validation_env
from dacboenv.experiment.task_metadata import parse_task_metadata

ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
TARGETS = {"train": 60, "validation": 40}
SELECTION_SEED = 882301


def _context_inventory() -> list[dict[str, object]]:
    records = []
    for split in ("train", "validation"):
        frame = pd.read_parquet(f"artifacts/headroom_{split}_snapshots.parquet")
        frame = frame.drop_duplicates(["task_id", "inner_seed"])
        rows = []
        for row in frame.itertuples(index=False):
            metadata = parse_task_metadata(row.task_id)
            rows.append(
                {
                    "split": split,
                    "task_id": row.task_id,
                    "inner_seed": int(row.inner_seed),
                    "domain": metadata.domain,
                    "scenario": metadata.scenario,
                    "dimension": metadata.dimension,
                    "manifest_hash": row.source_manifest_hash,
                    "reference_kind": row.reference_kind,
                    "reference_value": float(row.reference_value),
                    "reference_source_hash": row.reference_source_hash,
                    "stratum": metadata.scenario if metadata.domain == "yahpo" else f"bbob-d{metadata.dimension}",
                }
            )
        candidates = pd.DataFrame(rows).sort_values(["stratum", "task_id", "inner_seed"])
        rng = np.random.default_rng(SELECTION_SEED + (split == "validation"))
        groups = {key: value.index.to_list() for key, value in candidates.groupby("stratum", sort=True)}
        selected = []
        while len(selected) < TARGETS[split]:
            progressed = False
            for key in sorted(groups):
                if groups[key] and len(selected) < TARGETS[split]:
                    position = int(rng.integers(len(groups[key])))
                    selected.append(groups[key].pop(position))
                    progressed = True
            if not progressed:
                raise RuntimeError(f"Insufficient non-test contexts for {split} static panel.")
        for index in selected:
            context = candidates.loc[index].to_dict()
            for action, alpha in enumerate(ALPHAS):
                records.append({**context, "action": action, "alpha": alpha})
    return records


def _run(record: dict[str, object]) -> dict[str, object]:
    split = str(record["split"])
    factory = real_headroom_train_env if split == "train" else real_headroom_validation_env
    env = factory(str(record["task_id"]), int(record["inner_seed"]), "wei")
    try:
        _observation, _info = env.reset()
        initial_design_size = len(env._smac_instance.intensifier.config_selector._initial_design_configs)
        initial_fingerprints = runhistory_fingerprints(
            env._smac_instance.runhistory, env._smac_instance._scenario.configspace, initial_design_size
        )
        pre_policy_hash = hashlib.sha256(
            json.dumps(initial_fingerprints, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        action = int(record["action"])
        trajectory = []
        while True:
            _observation, reward, terminated, truncated, _info = env.step(action)
            evaluations = completed_evaluations(env)
            trajectory.append(
                {
                    "bo_evaluation": len(evaluations),
                    "cost": evaluations[-1].cost,
                    "incumbent": float(env.get_incumbent_cost()),
                    "reward": float(reward),
                }
            )
            if terminated or truncated:
                break
        return {
            **record,
            "initial_design_hash": initial_fingerprints["initial_design_hash"],
            "initial_design_configuration_hashes_json": json.dumps(
                initial_fingerprints["initial_design_configuration_hashes"]
            ),
            "initial_design_costs_json": json.dumps(initial_fingerprints["initial_design_costs"]),
            "pre_policy_optimizer_fingerprint": pre_policy_hash,
            "trajectory_json": json.dumps(trajectory, separators=(",", ":")),
            "final_incumbent": trajectory[-1]["incumbent"],
            "status": "complete",
        }
    finally:
        env.close()


def main() -> int:
    """Build inventory, run one array cell, or consolidate the complete panel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/headroom_predictability_v3/static_shards"))
    parser.add_argument("--build-inventory", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--consolidate", action="store_true")
    args = parser.parse_args()
    if args.build_inventory:
        records = _context_inventory()
        payload = {"protocol": "static-wei-v3", "count": len(records), "records": records}
        args.build_inventory.parent.mkdir(parents=True, exist_ok=True)
        args.build_inventory.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 0
    if args.inventory is None:
        raise ValueError("--inventory is required")
    payload = json.loads(args.inventory.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.consolidate:
        shards = sorted(args.output_root.glob("[0-9][0-9][0-9][0-9].json"))
        if len(shards) != payload["count"]:
            raise RuntimeError(f"Expected {payload['count']} static shards, found {len(shards)}")
        frame = pd.DataFrame([json.loads(path.read_text()) for path in shards])
        pairing = frame.groupby(["split", "task_id", "inner_seed"])[
            ["initial_design_hash", "pre_policy_optimizer_fingerprint"]
        ].nunique()
        if not (pairing == 1).all().all():
            raise RuntimeError("Static levels do not share exact pre-policy fingerprints.")
        frame.to_parquet("artifacts/headroom_predictability_v3/paired_static_trajectories.parquet", index=False)
        return 0
    if args.index is None or not 1 <= args.index <= payload["count"]:
        raise ValueError("invalid --index")
    result = _run(payload["records"][args.index - 1])
    (args.output_root / f"{args.index:04d}.json").write_text(json.dumps(result, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
