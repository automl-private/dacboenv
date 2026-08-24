"""Deployable static-context WEI comparator fitted on offline training states."""

from __future__ import annotations

import json
from pathlib import Path

from dacboenv.experiment.evaluation_determinism import file_sha256
from dacboenv.policy.abstract_policy import AbstractPolicy

SCENARIOS = ("bbob", "lcbench", "rbv2_glmnet", "rbv2_ranger", "rbv2_rpart", "rbv2_super", "rbv2_xgboost")


class ContextStaticWEIPolicy(AbstractPolicy):
    """Choose a training-fitted action from deployable scenario/dimension context."""

    def __init__(self, env: object, registry: str, registry_sha256: str) -> None:
        super().__init__(env, registry=registry, registry_sha256=registry_sha256)  # type: ignore[arg-type]
        path = Path(registry).resolve()
        if file_sha256(path) != registry_sha256:
            raise ValueError("Context-static selector registry hash mismatch.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._global = int(payload["global"])
        self._registry = {str(key): int(value) for key, value in payload.items() if key != "global"}

    def __call__(self, obs: object = None) -> int:  # noqa: ARG002
        """Return the action fixed for the current known task context."""
        task_id = str(getattr(self._env, "current_task_id", ""))
        if task_id.startswith("yahpo/so/"):
            scenario = task_id.split("/")[2]
            key = f"yahpo:scenario:{SCENARIOS.index(scenario)}"
        elif task_id.startswith("bbob/"):
            key = f"bbob:dimension:{task_id.split('/')[1]}"
        else:
            raise ValueError(f"Cannot derive deployable context from task {task_id!r}.")
        return self._registry.get(key, self._global)

    def set_seed(self, seed: int | None) -> None:
        """Do nothing because this comparator is deterministic."""


__all__ = ["ContextStaticWEIPolicy"]
