"""Run the bounded real CARP-S/SMAC determinism smoke panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dacboenv.experiment.build_evaluation_contexts import build_evaluation_contexts
from dacboenv.experiment.evaluation_determinism import require_process_determinism
from dacboenv.experiment.evaluation_runner import make_dacbo_method_runner, make_default_smac_method_runner
from dacboenv.experiment.paired_evaluator import (
    DEFAULT_SMAC,
    SAWEI,
    UNIFORM_RANDOM,
    EvaluationMethod,
    MethodRegistry,
    evaluate_registered_methods,
    write_evaluation_records_csv,
)
from dacboenv.experiment.protocol import load_manifest
from dacboenv.experiment.real_env import real_sawei_env, real_structured_bbob_env
from dacboenv.policy.random import RandomPolicy
from dacboenv.policy.sawei import SAWEIPolicy
from dacboenv.policy.static import StaticParameterPolicy

LEARNED_CONSTANT = "learned_constant_action_3"
STATIC_MATCH = "static_action_3"


def main() -> None:
    """Execute four frozen non-test contexts and persist canonical traces."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[LEARNED_CONSTANT, STATIC_MATCH, UNIFORM_RANDOM, DEFAULT_SMAC, SAWEI],
    )
    args = parser.parse_args()
    process_contract = require_process_determinism()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"Refusing populated smoke output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(args.manifest)
    contexts = build_evaluation_contexts(manifest, interaction_frequency=1)
    if len(contexts) != 4:  # noqa: PLR2004
        raise ValueError("The determinism smoke manifest must contain exactly four contexts.")

    registry = MethodRegistry(n_static_actions=5)
    registry.register_method(EvaluationMethod(LEARNED_CONSTANT, requires_trained_model=True))
    common = {
        "env_factory": real_structured_bbob_env,
        "action_family": "wei",
        "checkpoint_type": "smoke",
        "trace_directory": args.output_dir / "traces",
        "policy_seed": 20260810,
    }
    registry.register_runner(
        LEARNED_CONSTANT,
        make_dacbo_method_runner(
            **common,
            policy_factory=lambda env, _context, _method: StaticParameterPolicy(env, 3),
            outer_ppo_seed=999,
            policy_metadata={"test_policy": "constant_action_3", "trained_model": False},
        ),
    )
    registry.register_runner(
        STATIC_MATCH,
        make_dacbo_method_runner(
            **common,
            policy_factory=lambda env, _context, _method: StaticParameterPolicy(env, 3),
            outer_ppo_seed=None,
        ),
    )
    registry.register_runner(
        UNIFORM_RANDOM,
        make_dacbo_method_runner(
            **common,
            policy_factory=lambda env, _context, _method: RandomPolicy(env),
            outer_ppo_seed=None,
        ),
    )
    registry.register_runner(
        SAWEI,
        make_dacbo_method_runner(
            env_factory=real_sawei_env,
            policy_factory=lambda env, _context, _method: SAWEIPolicy(env),
            action_family="wei_continuous",
            checkpoint_type="none",
            outer_ppo_seed=None,
            trace_directory=args.output_dir / "traces",
            policy_seed=20260810,
        ),
    )
    registry.register_runner(
        DEFAULT_SMAC,
        make_default_smac_method_runner(
            output_directory=args.output_dir / "native_default_smac",
            trace_directory=args.output_dir / "traces",
            context_split="validation",
        ),
    )
    methods = args.methods
    records = evaluate_registered_methods(manifest, contexts, methods, registry)
    write_evaluation_records_csv(records, args.output_dir / "evaluation_determinism_smoke.csv")
    fingerprints = []
    for trace_path in sorted((args.output_dir / "traces").glob("*.json")):
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        fingerprints.append(
            {
                "method": trace["record"]["method"],
                "task_id": trace["record"]["task_id"],
                "inner_seed": trace["record"]["inner_seed"],
                **trace["fingerprints"],
            }
        )
    payload = {"process_contract": process_contract, "fingerprints": fingerprints}
    (args.output_dir / "evaluation_determinism_fingerprints.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
