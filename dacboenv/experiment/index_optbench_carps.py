"""Register installed OptBench task configs in CARP-S's cached index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from carps.utils.index_configs import cache_path, index_configs
from omegaconf import OmegaConf

from dacboenv.utils.carps_optimizer import get_installed_optbench_task_configs


def _expected_index_rows() -> dict[str, Path]:
    """Return CARP-S task IDs and files from the installed OptBench package."""
    expected: dict[str, Path] = {}
    for config_path in get_installed_optbench_task_configs().values():
        cfg = OmegaConf.load(config_path)
        task_id = str(cfg.task_id)
        if task_id in expected:
            raise RuntimeError(f"Duplicate OptBench CARP-S task ID {task_id!r}.")
        expected[task_id] = config_path.resolve()
    return expected


def _validate_index(index: pd.DataFrame, expected: dict[str, Path]) -> list[dict[str, str]]:
    """Validate that each OptBench display ID maps to its installed YAML."""
    if not {"config_fn", "task_id"}.issubset(index.columns):
        raise RuntimeError("CARP-S index is missing its config_fn/task_id columns.")

    rows: list[dict[str, str]] = []
    for task_id, expected_path in sorted(expected.items()):
        matches = index.loc[index["task_id"].astype(str) == task_id, "config_fn"]
        resolved = [Path(str(path)).resolve() for path in matches.dropna().tolist()]
        if resolved != [expected_path]:
            raise RuntimeError(
                f"CARP-S index entry for OptBench task {task_id!r} is not unique/current: "
                f"expected {[str(expected_path)]}, found {[str(path) for path in resolved]}."
            )
        rows.append({"task_id": task_id, "config_fn": str(expected_path)})
    return rows


def ensure_optbench_carps_index() -> dict[str, Any]:
    """Create or reuse a CARP-S index containing installed OptBench configs.

    Returns
    -------
    dict[str, Any]
        Machine-readable index path, refresh status, and validated rows.
    """
    expected = _expected_index_rows()
    refreshed = True
    if cache_path.is_file():
        current = pd.read_csv(cache_path)
        try:
            rows = _validate_index(current, expected)
            refreshed = False
        except RuntimeError:
            rows = []
    else:
        rows = []

    if refreshed:
        task_root = next(iter(expected.values())).parent
        index_configs(extra_task_paths=[str(task_root)])
        rows = _validate_index(pd.read_csv(cache_path), expected)

    return {
        "cache_path": str(cache_path.resolve()),
        "optbench_task_root": str(next(iter(expected.values())).parent),
        "refreshed": refreshed,
        "task_count": len(rows),
        "rows": rows,
    }


def main() -> None:
    """Ensure the cached index is ready and print its validated inventory."""
    print(json.dumps(ensure_optbench_carps_index(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
