"""Synthetic transaction tests, separate from real CARP-S pairing pilots."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from dacboenv.experiment.snapshot_branch import CompletedBOEvaluation
from dacboenv.selective_wei import initial_collection
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.initial_context import InitialContext


def test_static_collection_failure_cleanup_replay_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted arm resumes by replay, while corrupt complete output fails."""
    context = InitialContext(
        "synthetic",
        ("dimension",),
        (1.0,),
        (True,),
        ("M",),
        2,
        12,
        "design",
        canonical_hash([8.0, 4.0]),
        "model",
        "data",
        "protocol",
        0,
    )
    fail = [True]

    class Env:
        interaction_frequency = 5

        def __init__(self) -> None:
            self.records: list[CompletedBOEvaluation] = []
            self._initial_feature_provenance: dict[str, Any] = {}
            self._smac_instance = SimpleNamespace(runhistory={})

        def reset(self) -> None:
            self.records = [CompletedBOEvaluation(i, f'{{"x":{i}}}', cost) for i, cost in enumerate([8.0, 4.0])]
            self._smac_instance.runhistory = {i: SimpleNamespace(time=0.0) for i in range(2)}

        def get_initial_context(self, _settings: Any) -> InitialContext:
            return context

        def get_n_finished_trials(self) -> int:
            return len(self.records)

        def step(self, action: int) -> tuple[None, float, bool, bool, dict[str, Any]]:
            if fail[0]:
                fail[0] = False
                raise RuntimeError("Injected ask failure")
            for _ in range(min(5, 12 - len(self.records))):
                index = len(self.records)
                self.records.append(CompletedBOEvaluation(index, f'{{"x":{index}}}', 3.0 - action / 4))
                self._smac_instance.runhistory[index] = SimpleNamespace(time=0.0)
            return None, 0.0, len(self.records) == 12, False, {}

        def close(self) -> None:
            pass

    monkeypatch.setattr(initial_collection, "completed_evaluations", lambda env: tuple(env.records))
    with pytest.raises(RuntimeError, match="Injected"):
        initial_collection.collect_static_context(Env, tmp_path, recipe={"split": "train"})
    assert not (tmp_path / "collection.lock").exists()
    result = initial_collection.collect_static_context(Env, tmp_path, recipe={"split": "train"})
    assert result["status"] == "complete"
    assert initial_collection.read_verified(tmp_path / "arm_0.json")["replayed_previous_prefix"] == 2
    (tmp_path / "arm_0.json").write_text('{"content_hash":"wrong","status":"complete"}')
    with pytest.raises(ValueError, match="Corrupt"):
        initial_collection.collect_static_context(Env, tmp_path, recipe={"split": "train"})
    assert not (tmp_path / "collection.lock").exists()
