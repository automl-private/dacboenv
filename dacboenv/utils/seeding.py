"""Named, reproducible random streams for the experimental protocol."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

SEED_STREAM_VERSION = "numpy-seedsequence-v1"

# Numeric spawn keys are deliberately stable protocol data.  Adding a new
# stream must not renumber an existing entry, otherwise old experiments could
# no longer be replayed from their logged root seed.
SEED_STREAM_IDS: Mapping[str, int] = {
    "policy_model": 1,
    "vector_worker": 2,
    "task_selector": 3,
    "fallback_inner_pool": 4,
    "episode_inner": 5,
    "configspace": 6,
    "initial_design": 7,
    "random_design": 8,
    "acquisition_maximizer": 9,
    "policy_action_space": 10,
}


def _validate_seed_component(value: int, *, name: str) -> int:
    """Validate one non-negative integer used in a SeedSequence hierarchy."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def derive_named_seed(root_seed: int, stream: str, *, index: int = 0) -> int:
    """Derive one uint32 seed without consuming any sibling stream.

    ``spawn_key`` makes streams addressable by name and index.  Their values
    therefore do not depend on call order, action-space configuration, or how
    many values another component draws.
    """
    root_seed = _validate_seed_component(root_seed, name="root_seed")
    index = _validate_seed_component(index, name="index")
    try:
        stream_id = SEED_STREAM_IDS[stream]
    except KeyError as error:
        raise ValueError(f"Unknown seed stream {stream!r}; expected one of {sorted(SEED_STREAM_IDS)}.") from error

    sequence = np.random.SeedSequence(entropy=root_seed, spawn_key=(stream_id, index))
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def episode_component_seeds(inner_seed: int) -> dict[str, int]:
    """Return independent SMAC component seeds for one selected BO seed."""
    return {
        name: derive_named_seed(inner_seed, name)
        for name in ("configspace", "initial_design", "random_design", "acquisition_maximizer")
    }


def run_seed_metadata(root_seed: int, n_workers: int) -> dict[str, object]:
    """Create serializable metadata sufficient to replay a PPO seed tree."""
    root_seed = _validate_seed_component(root_seed, name="root_seed")
    n_workers = _validate_seed_component(n_workers, name="n_workers")
    if n_workers == 0:
        raise ValueError("n_workers must be positive.")
    return {
        "version": SEED_STREAM_VERSION,
        "root_outer_seed": root_seed,
        "policy_model_seed": derive_named_seed(root_seed, "policy_model"),
        "vector_worker_seeds": [
            derive_named_seed(root_seed, "vector_worker", index=index) for index in range(n_workers)
        ],
        "worker_child_streams": ["task_selector", "fallback_inner_pool", "episode_inner"],
        "episode_component_streams": [
            "configspace",
            "initial_design",
            "random_design",
            "acquisition_maximizer",
        ],
    }
