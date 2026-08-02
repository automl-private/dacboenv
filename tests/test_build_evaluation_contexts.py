"""Tests for manifest-to-evaluator context generation."""

from __future__ import annotations

from typing import Any

from dacboenv.experiment.build_evaluation_contexts import build_evaluation_contexts
from dacboenv.experiment.protocol import manifest_hash


def test_real_bbob_context_generation_uses_native_budget_and_live_exact_reference() -> None:
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": "context-builder-smoke",
        "domain": "bbob",
        "split": "validation",
        "status": "ready",
        "runnable": True,
        "task_ids": ["bbob/2/3/0"],
        "inner_seeds": [17],
    }
    manifest["manifest_hash"] = manifest_hash(manifest)

    contexts = build_evaluation_contexts(manifest, interaction_frequency=5)

    assert len(contexts) == 1
    context = contexts[0]
    assert context.evaluation_budget == 77
    assert context.reference_kind == "exact"
    assert context.reference_value == 20.91
    assert context.objective_transform == "identity"
    assert context.interaction_frequency == 5
    assert context.manifest_hash == manifest["manifest_hash"]
