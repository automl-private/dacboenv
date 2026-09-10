"""Pure initial-anchor tests; these do not stand in for CARP-S integration."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_base_data import PairedStaticLabels
from dacboenv.selective_wei.initial_context import (
    InitialAnchorCache,
    InitialBaseDecision,
    InitialContext,
    finite_probe_ubr,
    initial_output_scale,
)
from dacboenv.selective_wei.schemas import BaseDecision, CalibrationArtifact
from dacboenv.selective_wei.stochastic_base import SeededBaseMixture
from dacboenv.selective_wei.targets import TargetSemantics


def anchor() -> InitialContext:
    """Construct an explicit engineering-only initial anchor."""
    return InitialContext(
        "episode",
        ("dimension", "spread"),
        (2.0, 1.0),
        (True, True),
        ("M", "D"),
        2,
        4,
        "design",
        canonical_hash([8.0, 4.0]),
        "model",
        "data",
        "protocol",
        123,
    )


def test_initial_anchor_cache_serialization_and_no_reselection() -> None:
    """Later GP states or actions cannot replace the frozen initial choice."""
    cache = InitialAnchorCache()
    context = cache.freeze(anchor())
    choice = InitialBaseDecision(context.digest, 3, "base-model", 0)
    cache.select(choice)
    cache.select(choice)
    restored = InitialAnchorCache.from_dict(json.loads(json.dumps(cache.to_dict(), allow_nan=False)))
    assert restored.context == context
    assert restored.decision == choice
    with pytest.raises(ValueError, match="reset"):
        cache.freeze(replace(context, model_digest="later-model"))
    with pytest.raises(ValueError, match="reselected"):
        cache.select(replace(choice, action=1))
    cache.reset()
    cache.freeze(replace(context, total_budget=2))
    with pytest.raises(ValueError, match="No controlled budget"):
        cache.select(replace(choice, context_digest=cache.context.digest))


def test_anchor_feature_arrays_are_copies_and_masked() -> None:
    """Predictive arrays contain only explicitly selected feature values."""
    context = anchor()
    values, masks = context.predictor_input(("M",))
    values[0] = 999
    assert context.values[0] == 2
    assert masks.tolist() == [True]
    assert InitialContext.from_dict(asdict(context)).digest == context.digest


def test_probe_ubr_includes_all_initial_points_and_uses_std() -> None:
    """The finite-probe calculation neither drops the incumbent nor uses variance as std."""
    result = finite_probe_ubr(
        np.asarray([5.0, 3.0, 1.0]),
        np.asarray([4.0, 1.0, 1.0]),
        initial_indices=np.asarray([0, 1]),
        scale=2.0,
        confidence_multiplier=2.0,
    )
    assert result["raw_proxy"] == 6
    assert result["scaled_proxy"] == 3
    assert result["initial_point_count"] == 2
    assert result["gradient_available"] is False
    assert initial_output_scale(np.asarray([7.0, 7.0])) == (1.0, "constant_unit_fallback")


def test_full_static_curves_pairing_and_target_separation() -> None:
    """Local branch targets cannot masquerade as complete terminal losses."""
    context = anchor()
    curves = tuple((8.0, 4.0, 3.0, float(index) / 2) for index in range(5))
    group = PairedStaticLabels(
        context, (8.0, 4.0), curves, (context.digest,) * 5, (context.ordered_design_digest,) * 5, (2,) * 5, 2, 0
    )
    assert group.targets()["terminal_raw"] == [0.0, 0.5, 1.0, 1.5, 2.0]
    with pytest.raises(ValueError, match="complete"):
        replace(group, incumbent_curves=curves[:4])
    with pytest.raises(ValueError, match="Incompatible target"):
        TargetSemantics("full_static_terminal_loss", "minimize").require_compatible(
            TargetSemantics("fixed_action_local_gain", "maximize", 5)
        )


def test_unbounded_conformal_strict_json_roundtrip() -> None:
    """Unbounded inference is encoded as null plus status, never Infinity."""
    artifact = CalibrationArtifact(
        "id", "task_max_simultaneous", 5, (), "tasks", 0.1, float("inf"), None, "insufficient_tasks", 5
    )
    serialized = json.dumps(artifact.to_dict(), allow_nan=False)
    assert "Infinity" not in serialized
    restored = CalibrationArtifact.from_dict(json.loads(serialized))
    assert np.isinf(restored.residual_quantile)


def test_stochastic_base_draws_once_and_restores_random_stream() -> None:
    """A stochastic base is not its modal projection and repeated reads do not redraw."""
    base = BaseDecision(0, 0.0, "B4", "global", "global", 5, "m", "d", action_probabilities=(0.2,) * 5)
    sampler = SeededBaseMixture("episode_static_mixture", 10)
    other = SeededBaseMixture("episode_static_mixture", 10)
    actions = [sampler.select(base, episode_id=str(i), block_index=0).action for i in range(20)]
    assert len(set(actions)) > 1
    assert actions == [other.select(base, episode_id=str(i), block_index=0).action for i in range(20)]
    fixed = sampler.select(base, episode_id="new", block_index=0)
    assert sampler.select(base, episode_id="new", block_index=9) == fixed
    restored = SeededBaseMixture("episode_static_mixture", 123)
    restored.load_state_dict(json.loads(json.dumps(sampler.state_dict())))
    assert (
        restored.select(base, episode_id="next", block_index=0).action
        == sampler.select(base, episode_id="next", block_index=0).action
    )
