"""Stable Baselines3 Model Policy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.utils import get_class
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from dacboenv.policy.abstract_policy import AbstractPolicy

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

    from dacboenv.dacboenv import ActType, DACBOEnv, ObsType


class ModelPolicy(AbstractPolicy):
    """Policy that uses a pre-trained RL model to select actions."""

    def __init__(
        self,
        env: DACBOEnv,
        model: BaseAlgorithm | str,
        model_class: type[BaseAlgorithm] | str | None = None,
        normalization_wrapper: str | None = None,
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
        """
        super().__init__(env, model=model, model_class=model_class, normalization_wrapper=normalization_wrapper)

        vec_env = DummyVecEnv([lambda: env])

        if normalization_wrapper is not None:
            vec_env = VecNormalize.load(normalization_wrapper, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False

        self._vec_env = vec_env

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
