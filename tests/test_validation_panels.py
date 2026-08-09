"""Contracts for the frozen two-tier validation-panel manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dacboenv.env.instance import RoundRobinInstanceSelector
from dacboenv.experiment.protocol import (
    manifest_hash,
    require_runnable_manifest,
    validate_manifest_structure,
)
from omegaconf import OmegaConf

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_INSTANCE_SET_DIRECTORY = _REPOSITORY_ROOT / "dacboenv" / "configs" / "instance_sets"
_PANEL_DIRECTORY = _REPOSITORY_ROOT / "dacboenv" / "configs" / "validation_panels"
_FROZEN_BBOB_VALIDATION_HASH = "36ed3fb56ddc141069b1efad21f4f2ee51d98fed5a0ebaf8c1cdc0d3fcfec196"


def _load(path: Path) -> dict[str, Any]:
    manifest = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(manifest, dict)
    return manifest


def _panel(name: str) -> dict[str, Any]:
    return _load(_PANEL_DIRECTORY / f"{name}.yaml")


@pytest.mark.parametrize(
    "name",
    ["bbob_frequent", "bbob_full", "yahpo_frequent", "yahpo_full", "mixed_frequent", "mixed_full"],
)
def test_validation_panel_manifest_hashes_are_frozen_and_valid(name: str) -> None:
    panel = _panel(name)

    validate_manifest_structure(panel)

    assert panel["manifest_hash"] == manifest_hash(panel)
    assert panel["split"] == "validation"


def test_bbob_tiers_are_exact_replays_of_the_frozen_validation_contexts() -> None:
    canonical = _load(_INSTANCE_SET_DIRECTORY / "bbob_validation.yaml")
    frequent = _panel("bbob_frequent")
    full = _panel("bbob_full")

    validate_manifest_structure(canonical)
    assert canonical["manifest_hash"] == _FROZEN_BBOB_VALIDATION_HASH
    assert frequent["source_manifest"]["hash"] == _FROZEN_BBOB_VALIDATION_HASH
    assert full["source_manifest"]["hash"] == _FROZEN_BBOB_VALIDATION_HASH
    assert frequent["task_ids"] == canonical["task_ids"] == full["task_ids"]
    assert frequent["inner_seeds"] == canonical["inner_seeds"][:2]
    assert full["inner_seeds"] == canonical["inner_seeds"]
    assert len(frequent["task_ids"]) * len(frequent["inner_seeds"]) == 20
    assert len(full["task_ids"]) * len(full["inner_seeds"]) == 40

    for panel in (frequent, full):
        first = RoundRobinInstanceSelector(panel["task_ids"], panel["inner_seeds"])
        second = RoundRobinInstanceSelector(panel["task_ids"], panel["inner_seeds"])
        count = panel["panel"]["episode_count"]
        first_replay = first.select_instance(size=count)
        second_replay = second.select_instance(size=count)

        assert first_replay == second_replay
        assert len(first_replay) == len(set(first_replay)) == count


def test_bbob_panels_remain_disjoint_from_the_strict_test_manifest() -> None:
    strict_test = _load(_INSTANCE_SET_DIRECTORY / "bbob_test_strict.yaml")
    frequent = _panel("bbob_frequent")
    full = _panel("bbob_full")

    assert not set(frequent["task_ids"]).intersection(strict_test["task_ids"])
    assert not set(full["task_ids"]).intersection(strict_test["task_ids"])


@pytest.mark.parametrize(
    ("name", "task_count", "seed_count", "episode_count"),
    [("yahpo_frequent", 12, 2, 24), ("yahpo_full", 24, 4, 96)],
)
def test_yahpo_tiers_are_frozen_ready_replays(name: str, task_count: int, seed_count: int, episode_count: int) -> None:
    panel = _panel(name)

    assert panel["status"] == "ready"
    assert panel["runnable"] is True
    assert len(panel["task_ids"]) == task_count
    assert len(panel["inner_seeds"]) == seed_count
    assert panel["panel"]["episode_count"] == episode_count
    assert panel["panel"]["official_test_instances_excluded"] is True
    require_runnable_manifest(panel)
    selector = RoundRobinInstanceSelector(panel["task_ids"], panel["inner_seeds"])
    contexts = selector.select_instance(size=episode_count)
    assert len(contexts) == len(set(contexts)) == episode_count


@pytest.mark.parametrize(
    ("name", "bbob_episodes", "yahpo_episodes", "total_episodes"),
    [("mixed_frequent", 20, 24, 44), ("mixed_full", 40, 96, 136)],
)
def test_mixed_tiers_are_complete_balanced_panels(
    name: str,
    bbob_episodes: int,
    yahpo_episodes: int,
    total_episodes: int,
) -> None:
    panel = _panel(name)
    canonical = _load(_INSTANCE_SET_DIRECTORY / "bbob_validation.yaml")
    bbob = panel["composition"]["bbob"]
    yahpo = panel["composition"]["yahpo"]

    assert panel["runnable"] is True
    assert len(panel["task_ids"]) * len(panel["inner_seeds"]) == total_episodes
    assert panel["panel"]["episode_count"] == total_episodes
    assert bbob["task_ids"] == canonical["task_ids"]
    assert bbob["episode_count"] == bbob_episodes
    assert len(yahpo["task_ids"]) * len(yahpo["inner_seeds"]) == yahpo_episodes
    assert panel["composition"]["balanced_score_weights"] == {"bbob": 0.5, "yahpo": 0.5}
    require_runnable_manifest(panel)
