"""Optional override-then-frozen-base terminal labels through CARP-S replay."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, cast

import numpy as np

from dacboenv.env.reward import normalized_reference_regret_potential
from dacboenv.experiment.collect_snapshots import completed_evaluations, observation_digest
from dacboenv.experiment.snapshot_branch import BOSnapshot, replay_snapshot, snapshot_record_digest
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_context import InitialAnchorCache


def terminal_interventions(  # noqa: C901, PLR0912 - explicit replay/accounting guards
    snapshot: BOSnapshot,
    factory: Callable[[str, int], Any],
    *,
    horizon: int,
) -> dict[str, Any]:
    """Hold each action for H BO evaluations, then a0 to the same total T.

    This target is deliberately distinct from local fixed-action potential
    gain. Shortened interventions are marked explicitly; no branch overruns T.
    All labels require a portable initial anchor and complete source history.
    """
    if horizon not in {5, 10} or snapshot.interaction_frequency != 5:  # noqa: PLR2004
        raise ValueError("Terminal interventions require H5/H10 and f5.")
    if not snapshot.initial_anchor_json or not snapshot.completed_evaluations:
        raise ValueError("Terminal intervention requires the original a0 and complete source records.")
    if snapshot.reference_value is None or snapshot.initial_design_incumbent is None:
        raise ValueError("Terminal intervention labels require reference/initial-incumbent provenance.")
    anchor = InitialAnchorCache.from_dict(json.loads(snapshot.initial_anchor_json)["state"])
    if anchor.context is None or anchor.decision is None:
        raise ValueError("Terminal intervention cannot infer a base at the intervention boundary.")
    base = anchor.decision.action
    total = anchor.context.total_budget
    source_count = len(snapshot.completed_evaluations)
    if source_count >= total:
        raise ValueError("No remaining intervention budget.")
    effective_horizon = min(horizon, total - source_count)
    outcomes: list[dict[str, Any]] = []
    for action in range(5):
        env = cast("Any", replay_snapshot(snapshot, factory))
        try:
            records = completed_evaluations(env)
            if records != snapshot.completed_evaluations:
                raise ValueError("Terminal branch source replay mismatch.")
            if snapshot.observation_hash and observation_digest(env.get_observation()) != snapshot.observation_hash:
                raise ValueError("Terminal branch source observation mismatch.")
            initial_potential = normalized_reference_regret_potential(
                env.get_incumbent_cost(),
                snapshot.reference_value,
                snapshot.initial_design_incumbent,
            )
            reward_sum = 0.0
            actions = []
            while env.get_n_finished_trials() < total:
                before = env.get_n_finished_trials()
                chosen = action if before - source_count < effective_horizon else base
                _, reward, terminated, truncated, _ = env.step(chosen)
                after = env.get_n_finished_trials()
                if after <= before or after > total:
                    raise ValueError("Terminal intervention has invalid BO accounting.")
                if before - source_count < effective_horizon < after - source_count:
                    raise ValueError("An action block overshot the requested intervention boundary.")
                reward_sum += float(reward)
                actions.append(chosen)
                if (terminated or truncated) and after != total:
                    raise ValueError("Terminal branch stopped before T.")
            potential = normalized_reference_regret_potential(
                env.get_incumbent_cost(),
                snapshot.reference_value,
                snapshot.initial_design_incumbent,
            )
            if not np.isfinite(reward_sum) or not np.isclose(
                reward_sum, potential - initial_potential, atol=1e-10, rtol=0
            ):
                raise ValueError("Terminal branch rewards do not telescope.")
            outcomes.append(
                {
                    "action": action,
                    "base_action": base,
                    "action_suffix": actions,
                    "source_digest": snapshot_record_digest(snapshot),
                    "final_incumbent": env.get_incumbent_cost(),
                    "final_potential": potential,
                    "reward_sum": reward_sum,
                    "records": [asdict(record) for record in completed_evaluations(env)],
                }
            )
        finally:
            env.close()
    advantages = np.asarray([row["final_potential"] - outcomes[base]["final_potential"] for row in outcomes])
    advantages[base] = 0.0
    protocol = {
        "version": "override-then-base-terminal-v1",
        "horizon": horizon,
        "base_semantic_hash": anchor.decision.base_semantic_hash,
        "interaction_frequency": 5,
    }
    return {
        "target_semantics": "override_then_base_terminal_advantage",
        "protocol": protocol,
        "protocol_hash": canonical_hash(protocol),
        "requested_horizon": horizon,
        "effective_horizon": effective_horizon,
        "shortened": effective_horizon != horizon,
        "source_count": source_count,
        "total_budget": total,
        "base_action": base,
        "terminal_advantages": advantages.tolist(),
        "branches": outcomes,
    }
