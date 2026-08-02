"""Short non-test vectorized YAHPO scenario-persistence smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from dacboenv.experiment.protocol import official_yahpo_so_task_ids
from dacboenv.experiment.real_env import real_structured_yahpo_env

DEFAULT_CONTEXTS = (
    ("yahpo/so/lcbench/3945/None", 710_001),
    ("yahpo/so/rbv2_super/28/None", 710_002),
)


def _observation_hash(observation: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        values = np.asarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def run_vector_smoke(
    contexts: tuple[tuple[str, int], ...] = DEFAULT_CONTEXTS,
) -> dict[str, Any]:
    """Reset and step fixed workers from distinct, persistent scenarios."""
    if not contexts:
        raise ValueError("At least one YAHPO context is required.")
    sealed = set(official_yahpo_so_task_ids())
    prohibited = sorted(task_id for task_id, _seed in contexts if task_id in sealed)
    if prohibited:
        raise ValueError(f"Vector smoke refuses official YAHPO test tasks: {prohibited!r}.")
    scenarios = [task_id.split("/")[2] for task_id, _seed in contexts]
    if len(set(scenarios)) != len(scenarios):
        raise ValueError("Vector smoke contexts must use distinct scenarios to prove persistence.")

    def factory(task_id: str, seed: int):
        env = real_structured_yahpo_env(
            task_id,
            seed,
            initial_design_n_configs=2,
            context_split="validation",
            random_design_probability=1.0,
        )
        env.prepare_for_first_reset()
        return env

    factories = [lambda task_id=task_id, seed=seed: factory(task_id, seed) for task_id, seed in contexts]
    vec_env = DummyVecEnv(factories)
    try:
        observation = vec_env.reset()
        for index, env in enumerate(vec_env.envs):
            row = {key: np.asarray(value[index]) for key, value in observation.items()}
            if not env.observation_space.contains(row):
                raise RuntimeError(f"Worker {index} reset observation is outside its declared space.")
            if not all(np.isfinite(value).all() for value in row.values()):
                raise RuntimeError(f"Worker {index} reset observation is non-finite.")
        next_observation, rewards, dones, infos = vec_env.step(np.zeros(len(contexts), dtype=np.int64))
        if not np.isfinite(rewards).all():
            raise RuntimeError("Vector smoke produced non-finite reward.")
        realized = [
            {
                "worker": index,
                "task_id": str(env.current_task_id),
                "inner_seed": int(env.current_seed),
                "scenario": str(env.current_task_id).split("/")[2],
                "bo_budget": int(env._n_trials),
                "bo_evaluations": int(env.get_n_finished_trials()),
                "maximum_fidelity": dict(env._carps_solver.task.objective_function.max_other_fidelities),
                "reward": float(rewards[index]),
                "done": bool(dones[index]),
                "step_task_id": str(infos[index]["task_id"]),
            }
            for index, env in enumerate(vec_env.envs)
        ]
        if [(row["task_id"], row["inner_seed"]) for row in realized] != list(contexts):
            raise RuntimeError("A vector worker escaped its fixed task/seed stratum.")
        return {
            "schema_version": 1,
            "contexts": [{"task_id": task_id, "inner_seed": seed} for task_id, seed in contexts],
            "reset_observation_sha256": _observation_hash(observation),
            "step_observation_sha256": _observation_hash(next_observation),
            "workers": realized,
            "clean_shutdown": True,
        }
    finally:
        vec_env.close()


def main() -> None:
    """Run the guarded two-scenario deterministic engineering smoke."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_vector_smoke()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
