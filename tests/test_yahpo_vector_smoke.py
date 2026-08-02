"""Safety guards for the real vectorized YAHPO smoke command."""

from __future__ import annotations

import pytest
from dacboenv.experiment.yahpo_vector_smoke import run_vector_smoke


def test_vector_smoke_rejects_official_test_before_environment_creation() -> None:
    with pytest.raises(ValueError, match="refuses official"):
        run_vector_smoke((("yahpo/so/lcbench/167168/None", 1),))


def test_vector_smoke_requires_distinct_scenarios_before_environment_creation() -> None:
    with pytest.raises(ValueError, match="distinct scenarios"):
        run_vector_smoke(
            (
                ("yahpo/so/lcbench/3945/None", 1),
                ("yahpo/so/lcbench/34539/None", 2),
            )
        )
