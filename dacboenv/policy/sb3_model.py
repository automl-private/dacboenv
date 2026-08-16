"""Stable Baselines3 Model Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.utils import get_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import ActType, DACBOEnv
    from dacboenv.env.observations.types import ObsType


class SB3DiscretePolicy(AbstractPolicy):
    """Policy bridge for a metadata-selected discrete SB3 algorithm."""

    def __init__(
        self,
        env: DACBOEnv,
        model: BaseAlgorithm | str,
        model_class: type[BaseAlgorithm] | str | None = None,
        normalization_wrapper: str | None = None,
        algorithm_id: str | None = None,
    ) -> None:
        """Initialize the model parameter policy.

        Parameters
        ----------
        env : DACBOEnv
            The environment in which the policy operates.
        model : BaseAlgorithm | str
            The RL model instance or path to a saved model.
        model_class : type[BaseAlgorithm] | str | None, optional
            The class of the RL model, required if loading from a path.
        normalization_wrapper : str | None, optional
            Path to a saved VecNormalize wrapper, if applicable.
        algorithm_id : str | None, optional
            Stable algorithm identifier. New bundles must provide it; legacy
            PPO bundles may omit it.
        """
        super().__init__(env, model=model, model_class=model_class, normalization_wrapper=normalization_wrapper)

        vec_env = DummyVecEnv([lambda: env])

        if normalization_wrapper is not None:
            vec_env = VecNormalize.load(normalization_wrapper, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        self._vec_env = vec_env

        expected_classes = {
            "ppo": "stable_baselines3.PPO",
            "dqn": "stable_baselines3.DQN",
            "double_dqn": "dacboenv.rl.double_dqn.DoubleDQN",
        }
        if algorithm_id is not None:
            if algorithm_id not in expected_classes:
                raise ValueError(f"Unknown SB3 policy algorithm metadata {algorithm_id!r}.")
            expected_class = get_class(expected_classes[algorithm_id])
            if model_class is None and not isinstance(model, str):
                declared_class = type(model)
            else:
                declared_class = model_class if isinstance(model_class, type) else get_class(str(model_class))
            if declared_class is not expected_class:
                raise ValueError(
                    f"Algorithm metadata {algorithm_id!r} requires {expected_classes[algorithm_id]}, "
                    f"not {model_class!r}."
                )

        if isinstance(model, str):
            assert model_class is not None, "If model is loaded from path, model_class must be provided."
            model_class = model_class if isinstance(model_class, type) else get_class(model_class)
            self._model = model_class.load(model, env=self._vec_env)
        else:
            self._model = model
            self._model.set_env(self._vec_env)

    def __call__(self, obs: ObsType | None = None) -> ActType:
        """Call the model for the action to take.

        Parameters
        ----------
        obs : ObsType | None, optional
            The current environment observation.

        Returns
        -------
        ActType
            Action predicted by the model
        """
        if isinstance(self._vec_env, VecNormalize):
            obs = self._vec_env.normalize_obs(obs)
        return self._model.predict(obs, deterministic=True)[0]

    def set_seed(self, seed: int | None) -> None:
        """Set seed for the model.

        Parameters
        ----------
        seed : int | None
            Seed
        """
        self._model.set_random_seed(seed=seed)


# Backward-compatible import used by existing PPO policy YAML files.
ModelPolicy = SB3DiscretePolicy

__all__ = ["ModelPolicy", "SB3DiscretePolicy"]
