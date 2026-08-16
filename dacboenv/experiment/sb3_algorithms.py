"""Algorithm-neutral construction and diagnostics for pinned SB3 agents."""

from __future__ import annotations

import contextlib
import inspect
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch as th
from gymnasium import spaces
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

from dacboenv.env.observations.gp_hyperparameters import GP_HP_SUMMARY_INDEX
from dacboenv.experiment.evaluation_determinism import canonical_sha256
from dacboenv.rl.double_dqn import DoubleDQN, double_dqn_bootstrap, vanilla_dqn_bootstrap

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.vec_env import VecEnv

SB3_VERSION = "2.9.0"
DQN_DISCRETE_ERROR = (
    "DQN and Double DQN currently require a Discrete action space. "
    "Use a fixed interaction-frequency discrete controller or PPO."
)


@dataclass(frozen=True)
class RLAlgorithmSpec:
    """Stable algorithm identity persisted with every run and bundle."""

    algorithm_id: str
    algorithm_class: str
    policy_class: str
    on_policy: bool
    supports_dict_observation: bool
    supports_multiple_environments: bool
    native_dict_n_steps: bool


ALGORITHM_REGISTRY = {
    "ppo": RLAlgorithmSpec(
        algorithm_id="ppo",
        algorithm_class="stable_baselines3.PPO",
        policy_class="MultiInputPolicy",
        on_policy=True,
        supports_dict_observation=True,
        supports_multiple_environments=True,
        native_dict_n_steps=False,
    ),
    "dqn": RLAlgorithmSpec(
        algorithm_id="dqn",
        algorithm_class="stable_baselines3.DQN",
        policy_class="MultiInputPolicy",
        on_policy=False,
        supports_dict_observation=True,
        supports_multiple_environments=True,
        native_dict_n_steps=False,
    ),
    "double_dqn": RLAlgorithmSpec(
        algorithm_id="double_dqn",
        algorithm_class="dacboenv.rl.double_dqn.DoubleDQN",
        policy_class="MultiInputPolicy",
        on_policy=False,
        supports_dict_observation=True,
        supports_multiple_environments=True,
        native_dict_n_steps=False,
    ),
}


def resolve_rl_algorithm_id(cfg: DictConfig) -> str:
    """Return a canonical ID, defaulting legacy PPO configurations to PPO."""
    algorithm_id = str(OmegaConf.select(cfg, "rl_algorithm_id", default="ppo"))
    if algorithm_id not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown rl_algorithm_id {algorithm_id!r}; expected {sorted(ALGORITHM_REGISTRY)}.")
    return algorithm_id


def validate_algorithm_action_space(algorithm_id: str, action_space: spaces.Space[Any]) -> None:
    """Fail before rollout for every unsupported off-policy action schema."""
    if algorithm_id in {"dqn", "double_dqn"} and not isinstance(action_space, spaces.Discrete):
        raise TypeError(DQN_DISCRETE_ERROR)


def observation_space_schema(space: spaces.Space[Any]) -> dict[str, Any]:
    """Return a canonical JSON-compatible observation/action space schema."""

    def _json_bound(array: np.ndarray) -> list[Any]:
        """Represent unbounded Gym limits without non-finite JSON numbers."""
        result: list[Any] = []
        for value in np.asarray(array).reshape(-1):
            numeric = float(value)
            if np.isneginf(numeric):
                result.append("-inf")
            elif np.isposinf(numeric):
                result.append("+inf")
            elif np.isnan(numeric):
                result.append("nan")
            else:
                result.append(numeric)
        return np.asarray(result, dtype=object).reshape(np.asarray(array).shape).tolist()

    if isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {name: observation_space_schema(subspace) for name, subspace in space.spaces.items()},
        }
    payload: dict[str, Any] = {"type": type(space).__name__, "shape": list(space.shape or ())}
    if isinstance(space, spaces.Box):
        payload["dtype"] = str(space.dtype)
        payload["low"] = _json_bound(space.low)
        payload["high"] = _json_bound(space.high)
    elif isinstance(space, spaces.Discrete):
        payload.update({"n": int(space.n), "start": int(space.start)})
    elif isinstance(space, spaces.MultiDiscrete):
        payload["nvec"] = np.asarray(space.nvec).tolist()
    return payload


def algorithm_run_metadata(cfg: DictConfig, env: VecEnv) -> dict[str, Any]:
    """Build immutable algorithm/space metadata for protocol files."""
    algorithm_id = resolve_rl_algorithm_id(cfg)
    spec = ALGORITHM_REGISTRY[algorithm_id]
    observation_schema = observation_space_schema(env.observation_space)
    action_schema = observation_space_schema(env.action_space)
    hyperparameter_node = (
        cfg.optimizer if algorithm_id == "ppo" else OmegaConf.select(cfg, "rl_algorithm.hyperparameters")
    )
    optimizer = OmegaConf.to_container(hyperparameter_node, resolve=True)
    return {
        **asdict(spec),
        "stable_baselines3_version": version("stable-baselines3"),
        "observation_keys": list(env.observation_space.spaces)
        if isinstance(env.observation_space, spaces.Dict)
        else [],
        "observation_schema": observation_schema,
        "observation_schema_hash": canonical_sha256(observation_schema),
        "action_space_type": type(env.action_space).__name__,
        "action_schema": action_schema,
        "action_schema_hash": canonical_sha256(action_schema),
        "algorithm_hyperparameters": optimizer,
    }


def _off_policy_optimizer_config(
    cfg: DictConfig, algorithm_class: type[BaseAlgorithm]
) -> tuple[dict[str, Any], DictConfig]:
    """Remove inherited PPO-only keys before constructing DQN/DDQN."""
    plain = OmegaConf.to_container(OmegaConf.select(cfg, "rl_algorithm.hyperparameters"), resolve=True)
    if not isinstance(plain, dict):
        raise TypeError("optimizer must be a mapping.")
    policy_kwargs: dict[str, Any] = {}
    algorithm_policy_kwargs = OmegaConf.select(cfg, "rl_algorithm.policy_kwargs", default=None)
    if algorithm_policy_kwargs is not None:
        converted = OmegaConf.to_container(algorithm_policy_kwargs, resolve=True)
        if not isinstance(converted, dict):
            raise TypeError("rl_algorithm.policy_kwargs must be a mapping.")
        policy_kwargs = converted
    signature_class = DQN if issubclass(algorithm_class, DQN) else algorithm_class
    accepted = set(inspect.signature(signature_class.__init__).parameters)
    accepted.discard("self")
    hydra_keys = {"_target_", "_partial_", "_recursive_", "_convert_"}
    filtered = {key: value for key, value in plain.items() if key in accepted or key in hydra_keys}
    filtered["_target_"] = "stable_baselines3.DQN" if algorithm_class is DQN else "dacboenv.rl.double_dqn.DoubleDQN"
    filtered["_partial_"] = True
    if isinstance(filtered.get("train_freq"), list):
        filtered["train_freq"] = tuple(filtered["train_freq"])
    optimizer_cfg = OmegaConf.create(filtered)
    if not isinstance(optimizer_cfg, DictConfig):
        raise TypeError("optimizer must resolve to a mapping.")
    return policy_kwargs, optimizer_cfg


def build_sb3_algorithm(
    cfg: DictConfig,
    env: VecEnv,
    *,
    tensorboard_log: str,
    model_seed: int | None = None,
) -> BaseAlgorithm:
    """Construct PPO, vanilla SB3 DQN, or local DoubleDQN from one config."""
    installed = version("stable-baselines3")
    if installed != SB3_VERSION:
        raise RuntimeError(f"DACBOEnv RL support is pinned to SB3 {SB3_VERSION}, found {installed}.")
    algorithm_id = resolve_rl_algorithm_id(cfg)
    validate_algorithm_action_space(algorithm_id, env.action_space)
    if algorithm_id == "ppo":
        plain = OmegaConf.to_container(cfg.optimizer, resolve=True)
        if not isinstance(plain, dict):
            raise TypeError("optimizer must be a mapping.")
        policy_kwargs = plain.pop("policy_kwargs", {})
        optimizer_cfg = OmegaConf.create(plain)
        assert isinstance(optimizer_cfg, DictConfig)
    else:
        algorithm_class = DQN if algorithm_id == "dqn" else DoubleDQN
        policy_kwargs, optimizer_cfg = _off_policy_optimizer_config(cfg, algorithm_class)
        n_steps = int(optimizer_cfg.get("n_steps", 1))
        if isinstance(env.observation_space, spaces.Dict) and n_steps != 1:
            raise ValueError("SB3 2.9.0 does not support n-step replay with Dict observations; set n_steps=1.")
    if model_seed is not None:
        optimizer_cfg.seed = int(model_seed)
    if algorithm_id == "ppo":
        model = instantiate(optimizer_cfg)(
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
        )
    else:
        algorithm_class = DQN if algorithm_id == "dqn" else DoubleDQN
        kwargs = OmegaConf.to_container(optimizer_cfg, resolve=True)
        assert isinstance(kwargs, dict)
        for key in ("_target_", "_partial_", "_recursive_", "_convert_"):
            kwargs.pop(key, None)
        if isinstance(kwargs.get("train_freq"), list):
            kwargs["train_freq"] = tuple(kwargs["train_freq"])
        model = algorithm_class(
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            **kwargs,
        )
    if not isinstance(model, (PPO, DQN, DoubleDQN)):
        raise TypeError(f"Configured RL target created unsupported model {type(model).__name__}.")
    return model


class DQNDiagnosticsCallback(BaseCallback):
    """Side-effect-neutral diagnostics for vanilla DQN and DoubleDQN."""

    def _on_step(self) -> bool:
        model = self.model
        if not isinstance(model, DQN):
            return True
        replay_buffer = model.replay_buffer
        transitions = int(replay_buffer.size()) * int(model.n_envs)
        model.logger.record("train/replay_buffer_size", transitions)
        model.logger.record("train/update_to_data_ratio", model._n_updates / max(transitions, 1))
        target_period = max(model.target_update_interval // model.n_envs, 1)
        model.logger.record("train/target_updates", model._n_calls // target_period)
        if transitions == 0 or getattr(model, "_last_obs", None) is None:
            return True
        with th.no_grad():
            observation, _ = model.policy.obs_to_tensor(model._last_obs)
            q_online = model.q_net(observation)
            next_observation_value = self.locals.get("new_obs")
            next_observation, _ = model.policy.obs_to_tensor(next_observation_value)
            q_online_next = model.q_net(next_observation)
            q_target = model.q_net_target(next_observation)
            if isinstance(model, DoubleDQN):
                bootstrap = double_dqn_bootstrap(q_online_next, q_target)
            else:
                bootstrap = vanilla_dqn_bootstrap(q_target)
            actions = th.as_tensor(self.locals["actions"], device=model.device).long().reshape(-1, 1)
            current = q_online.gather(dim=1, index=actions)
            rewards = th.as_tensor(self.locals["rewards"], device=model.device).float().reshape(-1, 1)
            dones = np.asarray(self.locals["dones"], dtype=bool)
            infos = self.locals.get("infos", [{} for _ in dones])
            terminal = np.asarray(
                [done and not info.get("TimeLimit.truncated", False) for done, info in zip(dones, infos, strict=True)],
                dtype=np.float32,
            )
            terminal_tensor = th.as_tensor(terminal, device=model.device).reshape(-1, 1)
            target = rewards + (1.0 - terminal_tensor) * float(model.gamma) * bootstrap
            td_error = (target - current).abs()
        model.logger.record("train/q_mean", q_online.mean().item())
        model.logger.record("train/q_std", q_online.std(unbiased=False).item())
        model.logger.record("train/target_q_mean", target.mean().item())
        model.logger.record("train/td_error_mean", td_error.mean().item())
        model.logger.record("train/td_error_p90", th.quantile(td_error.flatten(), 0.9).item())
        return True


class GPHyperparameterDiagnosticsCallback(BaseCallback):
    """Log aggregate GP-feature availability and extraction diagnostics."""

    def __init__(self) -> None:
        super().__init__()
        self._observation_count = 0
        self._available_sum = 0.0
        self._raw_mask_active_sum = 0.0

    def _on_step(self) -> bool:
        raw_observation = getattr(self.model, "_last_obs", None)
        get_original_obs = getattr(self.training_env, "get_original_obs", None)
        if callable(get_original_obs):
            raw_observation = get_original_obs()
        if isinstance(raw_observation, dict):
            batch_size = int(next(iter(raw_observation.values())).shape[0])
            if "gp_hp_summary" in raw_observation:
                summary = np.asarray(raw_observation["gp_hp_summary"])
                self._available_sum += float(summary[:, GP_HP_SUMMARY_INDEX["available"]].sum())
                self._observation_count += batch_size
            elif "gp_hp_raw_mask" in raw_observation:
                mask = np.asarray(raw_observation["gp_hp_raw_mask"])
                self._raw_mask_active_sum += float(np.mean(mask, axis=1).sum())
                self._observation_count += batch_size

        diagnostics: list[dict[str, int | float]] = []
        with contextlib.suppress(AttributeError, NotImplementedError):
            diagnostics = self.training_env.env_method("get_gp_hyperparameter_diagnostics")
        totals: dict[str, float] = {}
        for worker in diagnostics:
            for name, value in worker.items():
                totals[name] = totals.get(name, 0.0) + float(value)
        for name in (
            "gp_hp/extraction_calls",
            "gp_hp/extraction_failures",
            "gp_hp/non_gp_states",
            "gp_hp/unfitted_gp_states",
            "gp_hp/fallback_bound_count",
            "gp_hp/clipping_count",
            "gp_hp/truncation_count",
        ):
            self.logger.record(name, totals.get(name, 0.0))
        for role in ("lengthscale", "signal", "noise", "other"):
            key = f"gp_hp/role_count/{role}"
            self.logger.record(key, totals.get(key, 0.0))
        if self._observation_count:
            self.logger.record("gp_hp/availability_rate", self._available_sum / self._observation_count)
            self.logger.record(
                "gp_hp/raw_mask_active_fraction",
                self._raw_mask_active_sum / self._observation_count,
            )
        else:
            self.logger.record("gp_hp/availability_rate", 0.0)
            self.logger.record("gp_hp/raw_mask_active_fraction", 0.0)
        return True


def write_algorithm_metadata(cfg: DictConfig, env: VecEnv, path: Path) -> dict[str, Any]:
    """Persist the resolved registry record and return it."""
    metadata = algorithm_run_metadata(cfg, env)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "ALGORITHM_REGISTRY",
    "DQN_DISCRETE_ERROR",
    "SB3_VERSION",
    "DQNDiagnosticsCallback",
    "GPHyperparameterDiagnosticsCallback",
    "RLAlgorithmSpec",
    "algorithm_run_metadata",
    "build_sb3_algorithm",
    "observation_space_schema",
    "resolve_rl_algorithm_id",
    "validate_algorithm_action_space",
    "write_algorithm_metadata",
]
