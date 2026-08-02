"""Build the deterministic larger non-test action-feature fidelity panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dacboenv.utils.carps_optimizer import get_bbob_n_trials

_CONTEXTS = (
    ("bbob/2/3/0", 2, 113),
    ("bbob/4/2/1", 4, 127),
    ("bbob/8/2/1", 8, 139),
)
_ACTION_SPACES = ("wei", "lcb_quantile", "ucb_quantile", "af_selection")
_TARGET_PHASES = (0.25, 0.5, 0.75)
_STATIC_ACTIONS_BY_PHASE = {
    0.25: (0, 1),
    0.5: (2, 3),
    0.75: (4, 0),
}


def build_panel() -> list[dict[str, object]]:
    """Return 162 replay entries spanning actions, dimensions, and phases.

    WEI alpha 0.5 and posterior-mode expected improvement are the two
    controller states that reproduce native SMAC's default EI semantics, so
    those families also receive a ``default_smac_equivalent`` history. Native
    EI has no meaningful LCB/UCB action index and is not mislabeled there.
    """
    entries: list[dict[str, object]] = []
    for action_space in _ACTION_SPACES:
        for task_id, dimension, inner_seed in _CONTEXTS:
            budget = get_bbob_n_trials(dimension)
            initial_design = min(8 * dimension, int(np.floor(0.2 * budget)))
            for phase in _TARGET_PHASES:
                history_length = max(int(np.ceil(phase * budget)) - initial_design, 0)
                static_actions = _STATIC_ACTIONS_BY_PHASE[phase]
                policies: list[tuple[str, list[int]]] = [
                    (f"static_{action}", [action] * history_length) for action in static_actions
                ]
                if action_space in {"wei", "af_selection"}:
                    policies.append(("default_smac_equivalent", [2] * history_length))
                for policy_seed in (17, 29):
                    rng = np.random.default_rng([inner_seed, policy_seed])
                    policies.append(
                        (f"uniform_random_seed_{policy_seed}", rng.integers(0, 5, size=history_length).tolist())
                    )
                for history_policy, action_history in policies:
                    entries.append(
                        {
                            "task_id": task_id,
                            "inner_seed": inner_seed,
                            "action_space": action_space,
                            "history_policy": history_policy,
                            "action_history": action_history,
                        }
                    )
    return entries


def main() -> None:
    """Write the deterministic large fidelity-panel specification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = build_panel()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(panel)} snapshots to {args.output}")


if __name__ == "__main__":
    main()
