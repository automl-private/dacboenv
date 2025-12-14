"""DACBOenv as played by the policy as a carps objective function."""

from __future__ import annotations

import time
from abc import abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
from carps.objective_functions.objective_function import ObjectiveFunction
from carps.utils.trials import TrialValue
from ConfigSpace import Configuration, ConfigurationSpace, Float
from hydra.utils import get_class

from dacboenv.env.policy import PerceptronPolicy, Policy
from dacboenv.env.reward import get_initial_design_size

if TYPE_CHECKING:
    from carps.loggers.abstract_logger import AbstractLogger
    from carps.utils.trials import TrialInfo

    from dacboenv.dacboenv import DACBOEnv


def get_perceptron_configspace(n_obs: int, weight_bounds: tuple[float, float]) -> ConfigurationSpace:
    """Get configuration space for perceptron policy.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    weight_bounds : tuple[float,float]
        The weight bounds.

    Returns
    -------
    ConfigurationSpace
        The configuration space, contaings n_obs + 1 hyperparameters (weight vector and bias).
    """
    n_hps = n_obs + 1  # theta + bias
    configspace = ConfigurationSpace()
    configspace.add([Float(name=f"w{i}", bounds=weight_bounds) for i in range(n_hps)])
    return configspace


def get_perceptron_configspace_from_env(env: DACBOEnv, weight_bounds: tuple[float, float]) -> ConfigurationSpace:
    """Get perceptron configspace from env.

    Parameters
    ----------
    env : DACBOEnv
        DACBO env.
    weight_bounds : tuple[float,float]
        The bounds for the weights.

    Returns
    -------
    ConfigurationSpace
        Configuration space.
    """
    n_obs = len(env.observation_space)
    return get_perceptron_configspace(n_obs=n_obs, weight_bounds=weight_bounds)


def rollout(env: DACBOEnv, policy: Policy, max_episode_length: int = 100000) -> dict:
    """Rollout policy on environment.

    Parameters
    ----------
    env : DACBOEnv
        DACBO env.
    policy : Policy
        The policy to evaluate.
    max_episode_length : int, optional
        The maximum episode length, by default 100000. Needed for adaptive capping.

    Returns
    -------
    dict
        instance, cost_inc, reward_mean, rewards, episode_length
    """
    obs, info = env.reset()

    terminated = False
    truncated = False
    counter = 0

    rewards = []

    while not (terminated | truncated):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action=action)
        counter += 1
        rewards.append(reward)
        if counter == max_episode_length:
            break

    cost_inc = env.get_incumbent_cost()
    reward_mean = np.mean(rewards)

    return {
        "instance": env.instance,
        "cost_inc": cost_inc,
        "reward_mean": reward_mean,
        "rewards": rewards,
        "episode_length": counter,
    }


class DACBOObjectiveFunction(ObjectiveFunction):
    """DACBO objective function.

    Goal: Optimize a policy for a DACBO env.
    """

    def __init__(
        self,
        env: DACBOEnv,
        loggers: list[AbstractLogger] | None = None,
        policy_class: type[Policy] | str = PerceptronPolicy,
        policy_kwargs: dict[str, Any] | None = None,
        cost: str = "episode_length_scaled",
    ) -> None:
        """Init.

        Parameters
        ----------
        env : DACBOEnv
            DACBO env.
        loggers : list[AbstractLogger] | None, optional
            Carps loggers, by default None
        policy_class : type[Policy] | str, optional
            The policy class, by default PerceptronPolicy
        policy_kwargs : dict[str, Any] | None, optional
            Policy kwargs, by default None
        cost : str, optional
            Which type of cost to return, by default `episode_length_scaled`, which is the (episode length - n_initial
            design) / n_model_based_budget. Can also be `cost_inc`, which is simply the cost of the incumbent.
        """
        super().__init__(loggers)
        self._env = env
        self._policy_class = policy_class if isinstance(policy_class, type | partial) else get_class(policy_class)
        self._policy_kwargs = policy_kwargs if policy_kwargs is not None else {}

        self._internal_seeds = self._env._inner_seeds
        self._seed_map: dict[int, int] = {}

        assert cost in ["episode_length_scaled", "cost_inc"]
        self._cost = cost

        # Create action space and observation space
        _, _ = self._env.reset()

    @abstractmethod
    def make_policy(self, config: Configuration, seed: int | None = None) -> Policy:
        """Make perceptron policy.

        Parameters
        ----------
        config : Configuration
            The configuration containing the weights.
        seed : int | None, optional
            Seed, by default None

        Returns
        -------
        PerceptronPolicy
            Instantiated perceptron policy.
        """

    def _evaluate(self, trial_info: TrialInfo) -> TrialValue:
        starttime = time.time()
        cost = self.target_function(
            config=trial_info.config,
            budget=trial_info.budget,
            instance=trial_info.instance,
            seed=trial_info.seed,
            cutoff=trial_info.cutoff,
        )
        endtime = time.time()
        duration = endtime - starttime

        internal_seed = self._get_internal_seed(trial_info.seed)
        additional_info = {"internal_seed": internal_seed, "cutoff": trial_info.cutoff}
        return TrialValue(
            cost=cost, time=duration, starttime=starttime, endtime=endtime, additional_info=additional_info
        )

    def set_dacbo_env_instance(self, instance: str | None = None, seed: int | None = None) -> None:
        """Set instance in DACBO env.

        Instance here is carps `task_id`.

        Parameters
        ----------
        instance : str | None, optional
            The carps `task_id`, by default None
        seed : int | None, optional
            The seed, by default None
        """
        if instance is not None:
            self._env.task_ids = [instance]
        if seed is not None:
            self._env._inner_seeds = [seed]

    def _get_internal_seed(self, seed: int | None) -> int | None:
        """Get internal seed based on outer seed.

        Relevant, because e.g. SMAC asks for specific seeds to evaluate. If we terminate our env based on a reached
        threshold, that threshold only counts for very specific seeds. Therefore we must internally match the seeds.

        Parameters
        ----------
        seed : int | None
            The external seed, e.g. provided by SMAC.

        Returns
        -------
        int | None
            The internal seed, which is mapped from the external seed.
        """
        if seed is None:
            return seed
        external_seed = seed
        internal_seeds_used = set(self._seed_map.values())
        seeds_left = set(self._internal_seeds) - internal_seeds_used
        if external_seed not in self._seed_map:
            self._seed_map[external_seed] = next(iter(seeds_left))
        return self._seed_map[external_seed]

    def target_function(
        self,
        config: Configuration,
        budget: int | float | None = None,  # noqa: ARG002
        instance: str | None = None,
        seed: int | None = None,
        cutoff: float | None = None,
    ) -> float:
        """The SMAC compatible target function.

        Parameters
        ----------
        config : Configuration
            The configuration.
        budget : int | float | None, optional
            The multi-fidelity budget, not applicable here, by default None
        instance : str | None, optional
            The instance, here carps `task_id`, by default None
        seed : int | None, optional
            The seed to evaluate the config on, by default None
        cutoff : float | None, optional
            The runtime cutoff, here as the maximum fraction of the model-based optimization budget, by default None

        Returns
        -------
        float
            Cost as runtime: episode_length/model_based_budget
        """
        internal_seed = self._get_internal_seed(seed)

        self.set_dacbo_env_instance(instance=instance, seed=internal_seed)
        smbo = self._env._smac_instance
        n_initial_design = get_initial_design_size(smbo)
        n_smbo = smbo._scenario.n_trials
        n_model_based = n_smbo - n_initial_design

        policy = self.make_policy(config=config, seed=internal_seed)

        max_episode_length = 10000
        if cutoff is not None:
            max_episode_length = cutoff * n_model_based + n_initial_design
        result = rollout(env=self._env, policy=policy, max_episode_length=max_episode_length)

        if self._cost == "episode_length_scaled":
            ep_length = result["episode_length"] - n_initial_design
            return ep_length / n_model_based
        if self._cost == "cost_inc":
            return result["cost_inc"]
        raise ValueError(f"Cannot handle request cost: {self._cost}.")


class PerceptronDACBOObjectiveFunction(DACBOObjectiveFunction):
    """Perceptron policy for DACBO env objective function.

    Optimize the weight vector and bias for controlling a DACBO env.
    """

    def __init__(
        self,
        env: DACBOEnv,
        loggers: list[AbstractLogger] | None = None,
        policy_class: type[Policy] | str = PerceptronPolicy,
        policy_kwargs: dict[str, Any] | None = None,
        weight_bounds: tuple[float, float] = (-5, 5),
        weight_in_log: bool = True,  # noqa: FBT001, FBT002
        cost: str = "episode_length_scaled",
    ) -> None:
        """Init.

        Parameters
        ----------
        env : DACBOEnv
            DACBO env.
        loggers : list[AbstractLogger] | None, optional
            Carps loggers, by default None
        policy_class : type[Policy] | str, optional
            The policy class, by default PerceptronPolicy
        policy_kwargs : dict[str, Any] | None, optional
            Policy kwargs, by default None
        weight_bounds : tuple[float,float], optional
            Bounds for the weights of the perceptron, by default (-5, 5)
        weight_in_log : bool, optional
            Bounds are in log space, by default True
        cost : str, optional
            Which type of cost to return, by default `episode_length_scaled`, which is the (episode length - n_initial
            design) / n_model_based_budget. Can also be `cost_inc`, which is simply the cost of the incumbent.
        """
        super().__init__(env=env, loggers=loggers, policy_class=policy_class, policy_kwargs=policy_kwargs, cost=cost)

        self._weight_bounds = weight_bounds
        self._weight_in_log = weight_in_log

    @property
    def configspace(self) -> ConfigurationSpace:
        """Get the configuration space for the perceptron.

        Returns
        -------
        ConfigurationSpace
            Configuration space (continuous, n_obs + 1 HPs)
        """
        return get_perceptron_configspace_from_env(self._env, self._weight_bounds)

    def make_policy(self, config: Configuration, seed: int | None = None) -> PerceptronPolicy:
        """Make perceptron policy.

        Parameters
        ----------
        config : Configuration
            The configuration containing the weights.
        seed : int | None, optional
            Seed, by default None

        Returns
        -------
        PerceptronPolicy
            Instantiated perceptron policy.
        """
        weights = list(config.values())
        if self._weight_in_log:
            weights = list(10 ** np.array(weights))
        policy_kwargs = self._policy_kwargs.copy()
        policy_kwargs.update({"weights": weights})
        policy = self._policy_class(env=None, **policy_kwargs)
        policy.set_seed(seed=seed)
        return policy
