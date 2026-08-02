"""RL Environment for DACBO."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    SupportsFloat,
)

import gymnasium as gym
import numpy as np
from carps.objective_functions.bbob import BBOBObjectiveFunction
from dataclasses_json import dataclass_json
from gymnasium.spaces import Box

from dacboenv.env.action import (
    AbstractActionSpace,
    AcqParameterActionSpace,
    PosteriorModeActionSpace,
    PosteriorQuantileActionSpace,
    WEIDiscreteActionSpace,
    WEITempoRLActionSpace,
)
from dacboenv.env.instance import InstanceSelector, RoundRobinInstanceSelector
from dacboenv.env.observation import WEI_ALPHA_LEVELS, ObservationSpace
from dacboenv.env.reward import REFERENCE_REGRET_REWARD_KEY, TRUE_REGRET_REWARD_KEY, DACBOReward
from dacboenv.reference import (
    BBOBExactReferenceProvider,
    JSONLReferenceBreachRecorder,
    ObjectiveReference,
    ReferenceBreachContext,
    ReferenceProvider,
    reference_regret,
)
from dacboenv.utils.carps_optimizer import build_carps_optimizer, is_bbob_task_id
from dacboenv.utils.loggingutils import get_logger
from dacboenv.utils.math import safe_log10
from dacboenv.utils.reference_performance import ReferencePerformance
from dacboenv.utils.seeding import derive_named_seed

if TYPE_CHECKING:
    from carps.optimizers.optimizer import Optimizer
    from omegaconf import DictConfig
    from smac.facade.abstract_facade import AbstractFacade
    from smac.main.smbo import SMBO

    from dacboenv.env.observations.types import ObsType

ActType = int | float | list[float] | np.ndarray | None

logger = get_logger("dacboenv")

_YAHPO_SCENARIO_INDEX = 2
_YAHPO_INSTANCE_INDEX = 3


@dataclass_json
@dataclass(frozen=True)
class InstanceSet:
    """Instance Set."""

    task_ids: list[str]
    seeds: list[int | None]


class DACBOEnv(gym.Env):
    """Gymnasium environment for Dynamic Algorithm Configuration in Bayesian Optimization (DACBO).

    This environment wraps a SMAC optimizer and offers a reinforcement learning interface for
    dynamically adjusting acquisition functions / parameters during Bayesian optimization.

    Observation Space
    ----------
    incumbent_changes : int
        Number of times the incumbent solution has changed.
    trials_passed : int
        Number of optimization trials completed.
    trials_left : int
        Number of trials remaining.
    ubr : float
        Upper bound regret.
    modelfit_mse : float
        Model fit measured as mean squared error.

    Action Space
    ----------
    acquisition_function : int
        Discrete selection among EI, PI, UCB, WEI.
    ei_pi_xi : float
        Parameter for EI/PI acquisition functions.
    ucb_beta : float
        Parameter for UCB acquisition function (log scale).
    wei_alpha : float
        Parameter for WEI acquisition function.

    Methods
    -------
    step(action)
        Executes one optimization step using the selected acquisition function and parameters.
    reset(seed=None, options=None)
        Resets the environment and optimizer state.
    update_optimizer(action)
        Updates the SMAC optimizer with the given action.
    get_observation()
        Computes the current observation and reward from the optimizer.
    get_reward()
        Computes the current reward from the optimizer.
    """

    def __init__(  # noqa: PLR0913, PLR0915, PLR0917
        self,
        task_ids: list[str],
        optimizer_cfg: DictConfig | None = None,
        observation_keys: list[str] | None = None,
        action_space_class: type[AbstractActionSpace] = AcqParameterActionSpace,
        action_space_kwargs: dict[str, Any] | None = None,
        reward_keys: list[str] | None = None,
        rho: float = 0.05,
        seed: int | None = None,
        reference_performance_fn: str = "reference_performance/reference_performance.parquet",
        reference_performance_optimizer_id: str = "SMAC3-BlackBoxFacade",
        inner_seeds: list[int | None] | None = None,
        terminate_after_reference_performance_reached: bool = False,  # noqa: FBT001, FBT002
        instance_selector_class: type[InstanceSelector] | None = None,
        evaluation_mode: bool = False,  # noqa: FBT001, FBT002
        interaction_frequency: int = 1,
        protocol_metadata: dict[str, Any] | None = None,
        reference_provider: ReferenceProvider | None = None,
        reference_breach_path: str | Path | None = None,
        run_id: str | None = None,
        yahpo_training_budget_multiplier: float = 1.0,
        context_split: str = "train",
        **kwargs: dict,
    ) -> None:
        """Initialize the DACBOEnv environment.

        Parameters
        ----------
        task_ids : list[str], optional
            The carps task ids that BO should run on.
        optimizer_cfg : DictConfig, optional
            The carps (SMAC) optimizer config. Defaults to `SMAC3-BlackBoxFacade` which is the standard blackbox
            facade with a GP.
        observation_keys : list[str], optional
            Which observations to compute at each step.
        action_space_class : type[AbstractActionSpace], optional
            Which action space, either parameter control or acquisition function selection.
        action_space_kwargs : dict[str, Any], optional
            Keyword arguments for the action space class.
        reward_keys : list[str], optional
            Which rewards to compute at each step. If nothing provided, will be `incumbent_cost`. Beware,
            this might not make sense for DAC as the tasks live on different scales.
        rho : float, optional
            ParEGO scalarization parameter.
        seed : int, optional
            Seed for the outer Gym environment and instance selector. This
            does not replace explicitly configured inner BO seeds.
        inner_seeds : list[int | None], optional
            The seeds that the inner BO will run on. A singleton ``[None]``
            generates a fresh deterministic inner seed on every episode from
            a stream derived from the outer seed.
        terminate_after_reference_performance_reached : bool, optional
            Terminate episode after a certain reference performance on a task/seed has been reached. Defaults to False.
        evaluation_mode : bool, optional
            Whether to be in train (default) or evaluation mode. Evaluation mode means that the episode is not
            terminated after a reference performance has been reached, and the reward will be 0.
            This circumvents running a reference optimizer on each evaluation task.
        interaction_frequency : int, optional
            Number of BO evaluations for which a non-Tempo action is held.
            Defaults to one evaluation per policy decision. Tempo actions
            continue to obtain their duration from the action itself.
        """
        legacy_instance_config = kwargs.get("deprecated_legacy_instance_config")
        if legacy_instance_config:
            raise ValueError(
                "Legacy configs/instances composition "
                f"{legacy_instance_config!r} is deprecated and is not a scientific manifest. "
                "Select a versioned instance_sets manifest; see "
                "reports/RUN_COMMANDS_POST_STAGE_A.md."
            )
        if reward_keys is None:
            reward_keys = ["incumbent_cost"]
        if action_space_kwargs is None:
            action_space_kwargs = {
                # SMAC's default acquisition function is EI, thus we adjust xi, thus those are sensible default bounds
                "bounds": (-10, 10)
            }
        super().__init__()

        # ``seed`` controls the outer environment. Independent child streams
        # drive context selection, compatibility fallback seeds, and fresh
        # per-episode BO seeds. The selected inner BO seed is kept separate in
        # ``current_seed`` and is passed to CARPS/SMAC.
        self._initialize_outer_seed(seed)

        self._optimizer_cfg = optimizer_cfg
        self._action_space_class = action_space_class
        self._action_space_kwargs = action_space_kwargs
        self._action_space: AbstractActionSpace
        self._observation_keys = observation_keys
        self._reward_keys = reward_keys
        self._rho = rho
        if not isinstance(interaction_frequency, int) or isinstance(interaction_frequency, bool):
            raise TypeError(f"interaction_frequency must be a positive integer, got {interaction_frequency!r}.")
        if interaction_frequency <= 0:
            raise ValueError(f"interaction_frequency must be > 0, got {interaction_frequency}.")
        self._interaction_frequency = interaction_frequency
        self._protocol_metadata = dict(protocol_metadata or {})
        self._reference_provider = reference_provider
        self._reference_breach_path = None if reference_breach_path is None else Path(reference_breach_path)
        self._run_id = run_id or str(self._protocol_metadata.get("run_id", f"dacboenv-seed-{self._seed}"))
        self._yahpo_training_budget_multiplier = float(yahpo_training_budget_multiplier)
        if context_split not in {"train", "validation", "test"}:
            raise ValueError(f"context_split must be train, validation, or test, got {context_split!r}.")
        self._context_split = context_split

        # Instance Set
        self._prepared_reset_result: tuple[ObsType, dict[str, Any], int | None] | None = None
        self._instance_set: InstanceSet
        self._instance_selector_class = (
            instance_selector_class if instance_selector_class else RoundRobinInstanceSelector
        )
        self.instance_selector: InstanceSelector  # Set whenever task_id or inner_seeds are updated
        self._uses_fallback_seeds = not inner_seeds
        inner_seeds = inner_seeds or self._fallback_seeds
        self.instance_set = (inner_seeds, task_ids)  # type: ignore[assignment]
        self._instance: tuple[int, str] | None = None

        self._evaluation_mode = evaluation_mode
        if self._evaluation_mode:
            logger.info(
                "Env is in evaluation mode! This means that a reward is not calculated, and episodes will be full "
                "length."
            )

        # Reference Performance
        self._terminate_after_reference_performance_reached = terminate_after_reference_performance_reached
        if self._evaluation_mode:
            self._terminate_after_reference_performance_reached = False
        self.reference_performance_fn = reference_performance_fn
        self.reference_performance_optimizer_id = reference_performance_optimizer_id
        self._requires_reference_performance = not self._evaluation_mode and (
            "symlogregret" in self._reward_keys or self._terminate_after_reference_performance_reached
        )
        if self._requires_reference_performance:
            self._reference_performance = ReferencePerformance(
                optimizer_id=self.reference_performance_optimizer_id,
                task_ids=self.instance_set.task_ids,
                seeds=None if self.is_in_random_mode else self.instance_set.seeds,
                reference_performance_fn=self.reference_performance_fn,
            )

        self._carps_solver: Optimizer
        self._smac_facade: AbstractFacade
        self._smac_instance: SMBO
        self._n_trials = None

        self._episode_reward = 0.0
        self._episode_length = 0

        self.current_task_id = ""
        self.current_seed = -1
        self.current_threshold: float | None = None
        self._objective_minimum: float | None = None
        self._objective_reference: ObjectiveReference | None = None
        self._reference_breach_recorder: JSONLReferenceBreachRecorder | None = None
        self.last_action: ActType | None = None

    @property
    def instance_set(self) -> InstanceSet:
        """The instance set."""
        return self._instance_set

    @instance_set.setter
    def instance_set(self, seeds_taskids: tuple[list[int | None], list[str]]) -> None:
        seeds, task_ids = seeds_taskids
        self._validate_inner_seeds(seeds)
        self._validate_true_regret_task_ids(task_ids)
        self._instance_set = InstanceSet(task_ids=task_ids, seeds=seeds)
        self.is_in_random_mode = seeds == [None] or (self._uses_fallback_seeds and seeds == self._fallback_seeds)
        self._build_instance_selector()
        self._instance = None
        # A prepared episode belongs to the old selector/context set.
        self._prepared_reset_result = None

    @staticmethod
    def _validate_inner_seeds(seeds: list[int | None]) -> None:
        """Validate a fixed seed pool or the singleton streaming marker."""
        if any(seed is None for seed in seeds):
            if seeds != [None]:
                raise ValueError("Use exactly [None] for dynamic inner seeds; mixed seed specifications are invalid.")
            return

        max_seed = int(np.iinfo(np.uint32).max)
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int | np.integer) or not 0 <= int(seed) <= max_seed:
                raise ValueError(f"Inner seeds must be integers in [0, {max_seed}], got {seed!r}.")

    def _validate_true_regret_task_ids(self, task_ids: list[str]) -> None:
        """Reject non-BBOB contexts for the known-optimum reward."""
        if not self._uses_true_regret_reward:
            return
        unsupported_task_ids = [task_id for task_id in task_ids if not is_bbob_task_id(task_id)]
        if unsupported_task_ids:
            raise ValueError(
                f"{TRUE_REGRET_REWARD_KEY!r} is restricted to canonical BBOB task IDs "
                f"('bbob/<dimension>/<function>/<instance>'); got {unsupported_task_ids!r}."
            )

    @property
    def _uses_true_regret_reward(self) -> bool:
        """Whether the selected reward requires a known BBOB optimum."""
        return TRUE_REGRET_REWARD_KEY in self._reward_keys

    @property
    def _uses_reference_regret_reward(self) -> bool:
        """Whether any reward consumes an exact or best-known reference."""
        return bool({TRUE_REGRET_REWARD_KEY, REFERENCE_REGRET_REWARD_KEY} & set(self._reward_keys))

    def _get_bbob_objective_minimum(self, task_id: str) -> float:
        """Return the active CARP-S BBOB objective's finite true minimum."""
        objective_function = self._carps_solver.task.objective_function
        if not is_bbob_task_id(task_id) or not isinstance(objective_function, BBOBObjectiveFunction):
            raise ValueError(
                f"{TRUE_REGRET_REWARD_KEY!r} requires a CARP-S BBOB objective for task {task_id!r}; "
                f"got {type(objective_function).__name__}."
            )
        objective_minimum = objective_function.f_min
        if objective_minimum is None or not np.isfinite(objective_minimum):
            raise ValueError(
                f"BBOB task {task_id!r} did not expose a finite true objective minimum: {objective_minimum!r}."
            )
        return float(objective_minimum)

    def _setup_reward(self, task_id: str) -> None:
        """Build the reward manager and its privileged reference context."""
        # The known BBOB optimum is privileged reward-only information. Read
        # it from the live CARP-S objective rather than reconstructing a
        # second IOH problem, and never add it to the policy observation.
        if self._uses_reference_regret_reward:
            provider = self._reference_provider
            if provider is None and is_bbob_task_id(task_id):
                provider = BBOBExactReferenceProvider()
            if provider is None:
                raise ValueError(
                    f"Reward {REFERENCE_REGRET_REWARD_KEY!r} requires an explicit reference_provider "
                    f"for non-BBOB task {task_id!r}."
                )
            objective_function = self._carps_solver.task.objective_function
            if is_bbob_task_id(task_id):
                task_metadata: dict[str, Any] = {
                    "task_id": task_id,
                    "runtime_objective_transform": "identity",
                    "reporting_objective_transform": "identity",
                    "fidelity": "not_applicable",
                }
            else:
                parts = task_id.split("/")
                objective_targets = tuple(str(value) for value in getattr(objective_function, "metrics", ()))
                if len(objective_targets) != 1:
                    raise ValueError(
                        f"Reference-regret reward requires exactly one runtime objective target for {task_id!r}; "
                        f"got {list(objective_targets)!r}."
                    )
                task_metadata = {
                    "task_id": task_id,
                    "runtime_objective_transform": "negative_accuracy",
                    "reporting_objective_transform": "one_minus_accuracy",
                    "fidelity": "fixed_maximum",
                    "scenario": (parts[_YAHPO_SCENARIO_INDEX] if len(parts) > _YAHPO_SCENARIO_INDEX else "unknown"),
                    "instance": (parts[_YAHPO_INSTANCE_INDEX] if len(parts) > _YAHPO_INSTANCE_INDEX else "unknown"),
                    "objective_target": objective_targets[0],
                }
            self._objective_reference = provider.get_reference(task_id, objective_function, task_metadata)
            self._objective_minimum = self._objective_reference.value
            breach_path = self._reference_breach_path
            if breach_path is None:
                output_directory = Path(self._smac_instance._scenario.output_directory)
                breach_path = output_directory / "reference_breaches.jsonl"
            self._reference_breach_recorder = JSONLReferenceBreachRecorder(breach_path)

        self._reward = DACBOReward(
            self._smac_instance,
            self._reward_keys,
            self._rho,
            objective_minimum=self._objective_minimum,
        )

    @staticmethod
    def _task_scenario_and_instance(task_id: str) -> tuple[str, str]:
        parts = task_id.split("/")
        if parts[0].lower() == "yahpo" and len(parts) >= 4:  # noqa: PLR2004
            return parts[2], parts[3]
        if parts[0].lower() == "bbob" and len(parts) == 4:  # noqa: PLR2004
            return "bbob", parts[3]
        return parts[0], parts[-1]

    def _check_reference_breach(self, trial_value: Any) -> None:
        """Persist a successful evaluation that materially beats its reference."""
        objective_reference = getattr(self, "_objective_reference", None)
        breach_recorder = getattr(self, "_reference_breach_recorder", None)
        if objective_reference is None or breach_recorder is None:
            return
        status = getattr(trial_value, "status", None)
        if status is not None and getattr(status, "name", str(status)) != "SUCCESS":
            return
        costs = np.asarray(trial_value.cost, dtype=float).reshape(-1)
        observed = float(costs[0]) if costs.size else np.inf
        scenario, instance = self._task_scenario_and_instance(self.current_task_id)
        reference_regret(
            objective_reference,
            observed,
            recorder=breach_recorder,
            context=ReferenceBreachContext(
                run_id=self._run_id,
                trial=int(self._smac_instance.runhistory.finished),
                outer_seed=self._seed,
                inner_seed=self.current_seed,
                scenario=scenario,
                instance=instance,
            ),
        )

    @property
    def instance(self) -> tuple[int, str]:
        """The intance (seed, task_ids).

        Raise
        -----
        ValueError
            When the env has to been reset after setting a new instance set.
        """
        if self._instance is None:
            raise ValueError("Reset the env first to select an instance!")
        return self._instance

    @instance.setter
    def instance(self, instance: tuple[int, str]) -> None:
        self._instance = instance

    @property
    def interaction_frequency(self) -> int:
        """Number of BO evaluations represented by a non-Tempo transition."""
        return self._interaction_frequency

    def _build_instance_selector(self) -> None:
        self.instance_selector = self._instance_selector_class(  # type: ignore[operator]
            task_ids=self._instance_set.task_ids,
            seeds=self._instance_set.seeds,
            selector_seed=self._selector_seed,
        )

    def restart_fixed_instance_sequence(self) -> None:
        """Restart a frozen evaluation manifest at its first context.

        SB3 vector environments automatically reset a completed episode.  A
        later explicit evaluation reset would otherwise skip that prepared
        context and rotate round-robin labels between checkpoints.
        """
        if any(seed is None for seed in self._instance_set.seeds):
            raise ValueError("Only a frozen integer-seed instance sequence can be restarted.")
        self._build_instance_selector()
        self._instance = None
        self._prepared_reset_result = None

    def _initialize_outer_seed(self, seed: int | None) -> None:
        """Initialize independent deterministic streams from one outer seed."""
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
        self._seed = int(seed)
        self._selector_seed = derive_named_seed(self._seed, "task_selector")
        fallback_generator = np.random.default_rng(derive_named_seed(self._seed, "fallback_inner_pool"))
        self._fallback_seeds = [int(value) for value in fallback_generator.integers(low=344, high=46483, size=3)]
        self._inner_seed_generator = np.random.default_rng(derive_named_seed(self._seed, "episode_inner"))

    def _reseed_episode_selection(self, seed: int) -> None:
        """Restart outer context selection without changing fixed BO seeds."""
        uses_current_fallbacks = self._uses_fallback_seeds and self._instance_set.seeds == self._fallback_seeds
        self._initialize_outer_seed(seed)

        # When no inner seeds were configured, make the generated fallback
        # context set reproducible from the Gym reset seed. Explicit inner
        # seeds remain an independent, unchanged experimental factor.
        if uses_current_fallbacks:
            self._instance_set = InstanceSet(
                task_ids=self._instance_set.task_ids,
                seeds=self._fallback_seeds,
            )

        self._build_instance_selector()
        self._instance = None

    def _apply_reset_seed(self, seed: int | None) -> None:
        """Apply an explicitly requested Gym reset seed to outer state."""
        if seed is not None:
            self._reseed_episode_selection(int(seed))

    def update_optimizer(self, action: ActType) -> None:
        """Update the SMAC optimizer with the given action.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Raises
        ------
        ValueError
            If the action type is invalid.
        """
        if action is not None:
            self._action_space.update_optimizer(action)
            self.last_action = action

    def _validate_action_feature_space(self) -> None:
        """Ensure action-conditioned rows match the environment action indices."""
        observation_keys = set(self._dacbo_observation_space._keys)  # type: ignore[arg-type]
        if "action_features" in observation_keys:
            if isinstance(
                self._action_space,
                WEIDiscreteActionSpace | WEITempoRLActionSpace,
            ):
                parameter_levels = np.asarray(
                    self._action_space._param_levels,
                    dtype=float,
                )
                if parameter_levels.shape != WEI_ALPHA_LEVELS.shape or not np.allclose(
                    parameter_levels,
                    WEI_ALPHA_LEVELS,
                ):
                    raise ValueError(
                        "The action_features rows require WEI alpha levels "
                        f"{WEI_ALPHA_LEVELS.tolist()}, got {parameter_levels.tolist()}."
                    )
            elif isinstance(self._action_space, PosteriorQuantileActionSpace):
                if len(self._action_space.quantile_levels) != len(WEI_ALPHA_LEVELS):
                    raise ValueError(
                        "The action_features table has five rows and therefore "
                        "requires exactly five posterior quantiles, got "
                        f"{list(self._action_space.quantile_levels)}."
                    )
            elif isinstance(self._action_space, PosteriorModeActionSpace):
                self._validate_posterior_mode_feature_space("action_features")
            else:
                raise ValueError(
                    "The action_features observation requires WEIDiscreteActionSpace, "
                    "WEITempoRLActionSpace, PosteriorQuantileActionSpace, or "
                    "PosteriorModeActionSpace."
                )

        if "af_action_features" in observation_keys:
            self._validate_posterior_mode_feature_space("af_action_features")

    def _validate_posterior_mode_feature_space(self, observation_key: str) -> None:
        """Validate the common or legacy posterior-mode feature table."""
        if not isinstance(self._action_space, PosteriorModeActionSpace):
            raise ValueError(f"The {observation_key} observation requires PosteriorModeActionSpace.")
        if self._action_space.space.n != 5:  # noqa: PLR2004
            raise ValueError(f"The {observation_key} table requires exactly five posterior modes.")

    def _previous_parameter_from_last_action(self) -> Any:
        """Map the last public action to the declared parameter observation."""
        if isinstance(self.action_space, Box):
            # Gymnasium's Box.contains is shape-sensitive. Policies may return
            # a scalar for a one-dimensional continuous action, but the
            # observation must match the declared ``(1,)`` float32 Box.
            return np.asarray(
                self.last_action,
                dtype=self.action_space.dtype,
            ).reshape(self.action_space.shape)
        if isinstance(self._action_space, WEITempoRLActionSpace):
            assert isinstance(self.last_action, Sequence | np.ndarray)
            return np.array([self._action_space._param_levels[int(self.last_action[1])]])
        if isinstance(self._action_space, WEIDiscreteActionSpace):
            action_idx = int(np.asarray(self.last_action).item())
            return np.array([self._action_space._param_levels[action_idx]])
        if isinstance(self._action_space, PosteriorQuantileActionSpace):
            action_idx = int(np.asarray(self.last_action).item())
            return np.array([self._action_space.quantile_levels[action_idx]])
        if isinstance(self._action_space, PosteriorModeActionSpace):
            return np.array([self._action_space.normalized_action])
        return self.last_action

    def modify_obs(self, obs: ObsType) -> ObsType:
        """Modify observations.

        Only modify the `previous_param` observation such that it is never None.
        That would not be liked by any neural network.
        `previous_param` will be set to a default, which is the middle of the action space.

        Parameters
        ----------
        obs : ObsType
            The observations.

        Returns
        -------
        ObsType
            The modified observations.
        """
        if "previous_param" in obs:
            if self.last_action is not None:
                previous_param = self._previous_parameter_from_last_action()
            elif isinstance(self.action_space, Box):
                # TODO adjust default/initial action. Right now: middle of action space
                previous_param = (self.action_space.high + self.action_space.low) / 2
            elif isinstance(self._action_space, WEIDiscreteActionSpace | WEITempoRLActionSpace):
                n_levels = len(self._action_space._param_levels)
                previous_param = np.array([self._action_space._param_levels[n_levels // 2]])
            elif isinstance(
                self._action_space,
                PosteriorQuantileActionSpace | PosteriorModeActionSpace,
            ):
                previous_param = np.array(
                    [self._action_space.current_control_value],
                )
            else:
                raise ValueError(f"Cannot handle space {self.action_space} to set last action.")

            obs["previous_param"] = previous_param
        return obs

    def get_observation(self) -> ObsType:
        """Compute the current observation from the optimizer.

        Returns
        -------
        obs : dict[str, Any]
            Dictionary of observation values.
        """
        obs = self._dacbo_observation_space.get_observation()
        return self.modify_obs(obs=obs)

    def get_reward(self) -> float:
        """Compute the current reward from the optimizer.

        Returns
        -------
        reward : float
            The current reward signal.
        """
        if not self._evaluation_mode:
            return self._reward.get_reward(self.current_threshold)
        return 0

    def get_next_instance(self) -> tuple[int | None, str]:
        """Get the next instance.

        Returns
        -------
        tuple[int,str]
            (seed,task_id)
        """
        return self.instance_selector.select_instance()  # type: ignore[return-value]

    def _context_info(self) -> dict[str, Any]:
        """Return auditable episode context without adding it to the policy state."""
        instance = getattr(self, "_instance", None)
        fallback_seed = instance[0] if instance is not None else -1
        fallback_task_id = instance[1] if instance is not None else ""
        task_id = getattr(self, "current_task_id", fallback_task_id)
        inner_seed = getattr(self, "current_seed", fallback_seed)
        info: dict[str, Any] = {
            "task_id": task_id,
            "inner_seed": inner_seed,
        }
        scenario = getattr(getattr(self, "_smac_instance", None), "_scenario", None)
        if scenario is not None and hasattr(scenario, "n_trials"):
            try:
                bo_evaluations = self.get_n_finished_trials()
            except AttributeError:
                # Lightweight environment probes need not implement SMAC's
                # private runhistory attributes merely to receive reset info.
                bo_evaluations = 0
            bo_budget = int(scenario.n_trials)
            info.update(
                {
                    "bo_evaluations": bo_evaluations,
                    "bo_budget": bo_budget,
                    "budget_fraction": bo_evaluations / max(bo_budget, 1),
                    "domain": task_id.split("/", maxsplit=1)[0].lower(),
                }
            )
        info.update(getattr(self, "_protocol_metadata", {}))
        seed_metadata = getattr(getattr(self, "_carps_solver", None), "seed_stream_metadata", None)
        if seed_metadata is not None:
            info["seed_stream_metadata"] = dict(seed_metadata)
        return info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one optimization step using the selected acquisition function and parameters.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Returns
        -------
        obs : dict
            The new observation after taking the action.
        reward : float
            The reward for the action taken.
        terminated : bool
            Whether the episode has terminated (reference performance reached).
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (empty).
        """
        if isinstance(self._action_space, WEITempoRLActionSpace):
            assert isinstance(action, Sequence | np.ndarray)
            step_duration = self._action_space._step_durations[int(action[0])]
            param_level = action[1]
            logger.info(f"Do action {param_level} for {step_duration} steps.")
        else:
            step_duration = self._interaction_frequency

        # Apply a policy decision once per agent transition. In particular,
        # incremental actions must not be accumulated once per Tempo sub-step.
        self.update_optimizer(action)
        total_reward = 0.0
        for _ in range(step_duration):
            obs, reward, terminated, truncated, info = self._step(action=action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        # Gym/SB3 episode statistics count agent transitions, not the BO
        # evaluations hidden inside one held action.
        self._episode_reward += total_reward
        self._episode_length += 1
        info = dict(info)
        info["bo_evaluations"] = self.get_n_finished_trials()
        info["policy_decisions"] = self._episode_length
        if terminated or truncated:
            info["episode"] = {
                "r": self._episode_reward,
                "l": self._episode_length,
            }
            self._episode_reward = 0.0
            self._episode_length = 0

        return obs, total_reward, terminated, truncated, info

    def _step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one optimization step using the selected acquisition function and parameters.

        Parameters
        ----------
        action : ActType
            Action specifying either the acquisition function or its parameter.

        Returns
        -------
        obs : dict
            The new observation after taking the action.
        reward : float
            The reward for the action taken.
        terminated : bool
            Whether the episode has terminated (reference performance reached).
        truncated : bool
            Whether the episode was truncated (always False).
        info : dict
            Additional information (empty).
        """
        # BO step
        trial_info = self._smac_instance.ask()
        _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
        self._smac_instance.tell(trial_info, trial_value)
        self._check_reference_breach(trial_value)

        terminated = False

        curr_incumbent = self.get_incumbent_cost()
        threshold = self.current_threshold
        if self._requires_reference_performance:
            threshold = self._reference_performance.query_cost(  # type: ignore[attr-defined]
                optimizer_id=self.reference_performance_optimizer_id,
                task_id=self.current_task_id,
                seed=None if self.is_in_random_mode else self.current_seed,
            )
            self.current_threshold = threshold

        if self._terminate_after_reference_performance_reached:
            if threshold is None:
                raise RuntimeError("Reference-performance termination requires a threshold.")
            distance = abs(curr_incumbent - threshold)
            log_distance = safe_log10(distance)
            logger.info(f"Current: {curr_incumbent:.4f}, threshold: {threshold:.4f}, log distance: {log_distance:.4f}")
            terminated = curr_incumbent <= threshold  # We minimize

        remaining_trials = self._smac_instance._scenario.n_trials - self._smac_instance.runhistory.finished
        # The BO budget is part of the MDP state and defines its finite
        # horizon. Treat budget exhaustion as a genuine terminal state so
        # value-based RL code does not bootstrap beyond the final BO trial.
        terminated = terminated or remaining_trials <= 0
        truncated = False

        # Compute observation + reward
        obs = self.get_observation()
        reward = self.get_reward() if not self._evaluation_mode else 0

        info = self._context_info()
        observation_space = getattr(self, "_dacbo_observation_space", None)
        diagnostics_fn = getattr(observation_space, "get_action_feature_diagnostics", None)
        if callable(diagnostics_fn):
            info.update(diagnostics_fn())

        logger.info(
            f"BO trial: {self._smac_instance.runhistory.finished}, "
            f"instance: {self.instance}, action: {action}, reward: {reward}, "
            f"terminated: {terminated}, truncated: {truncated}, info: {info}"
        )

        return obs, reward, terminated, truncated, info

    def get_incumbent_cost(self) -> float:
        """Get the current incumbent cost.

        Returns
        -------
        float
            Minimum cost found so far on this target function (not necessarily the reward).
        """
        return self._smac_instance.runhistory.get_min_cost(self._smac_instance.intensifier.get_incumbent())

    def prepare_for_first_reset(self) -> None:
        """Initialize spaces once and preserve that episode for the first caller.

        CARPS constructs a DACBO objective before handing its environment to
        SB3. The objective needs concrete Gym spaces, but an eager reset also
        evaluates the full initial design. Caching that reset prevents SB3
        from immediately discarding and repeating those evaluations.
        """
        if self._prepared_reset_result is not None:
            return
        prepared_seed = self._seed
        observation, info = self.reset(seed=prepared_seed)
        self._prepared_reset_result = (
            {key: value.copy() for key, value in observation.items()},
            info.copy(),
            prepared_seed,
        )

    def _consume_prepared_reset(
        self,
        seed: int | None,
    ) -> tuple[ObsType, dict[str, Any]] | None:
        """Return a compatible prepared episode once, or discard it."""
        if self._prepared_reset_result is None:
            return None

        observation, info, prepared_seed = self._prepared_reset_result
        self._prepared_reset_result = None
        if seed is not None and seed != prepared_seed:
            return None

        return (
            {key: value.copy() for key, value in observation.items()},
            info.copy(),
        )

    def reset(  # noqa: C901, PLR0915
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            Gymnasium seed for the outer environment and context selector.
            Explicitly configured inner BO seeds are not replaced by this
            value.
        options : dict, optional
            Additional reset options.

        Returns
        -------
        obs : tuple
            The initial observation.
        info : dict
            Additional information (empty).
        """
        # Gymnasium owns this RNG and requires it to be initialized from the
        # reset seed before any environment-specific reset logic.
        super().reset(seed=seed)

        # Space discovery may have prepared the first full BO episode. Reuse
        # it for an unseeded reset or the same outer seed. A new explicit seed
        # must restart context selection so the result corresponds to it.
        prepared_result = self._consume_prepared_reset(seed)
        if prepared_result is not None:
            return prepared_result

        self._apply_reset_seed(seed)

        self._episode_reward = 0.0
        self._episode_length = 0
        self.current_threshold = None
        self._objective_minimum = None
        self._objective_reference = None
        self._reference_breach_recorder = None

        # Reset SMAC instance
        if hasattr(self, "_carps_solver"):
            del self._carps_solver
        if hasattr(self, "_smac_instance"):
            del self._smac_instance

        # Select the next task and either a configured BO seed or the marker
        # requesting a fresh seed from the independent per-episode stream.
        selected_seed, task_id = self.get_next_instance()
        inner_seed = selected_seed
        if inner_seed is None:
            inner_seed = self._inner_seed_generator.integers(low=0, high=2**32, dtype=np.uint32)
        inner_seed = int(inner_seed)
        self.instance = (inner_seed, task_id)
        # Reference resolution and initial-design breach persistence happen
        # before the first policy-visible observation. Make the active context
        # authoritative before either operation so journals never contain the
        # constructor sentinels (empty task / seed -1).
        self.current_seed = inner_seed
        self.current_task_id = task_id

        # Build carps optimizer (wrapper around smac) with appropriate objective function
        optimizer_id = "SMAC3-BlackBoxFacade" if self._optimizer_cfg is None else None
        self._carps_solver = build_carps_optimizer(
            optimizer_id=optimizer_id,
            task_id=task_id,
            seed=inner_seed,
            optimizer_cfg=self._optimizer_cfg,
            yahpo_budget_multiplier=self._yahpo_training_budget_multiplier,
            context_split=self._context_split,
        )
        # Get the smac instance
        self._smac_facade = self._carps_solver.solver
        self._smac_instance = self._carps_solver.solver.optimizer
        self._n_trials = int(self._smac_instance._scenario.n_trials)

        if self._smac_instance._scenario.count_objectives() != 1:
            raise NotImplementedError("Multi-objective not supported.")

        # Build the action controller before structured observations so their
        # rows and previous-control state describe the installed acquisition
        # operation. Legacy observations do not depend on this ordering.
        self._action_space = self._action_space_class(smac_instance=self._smac_instance, **self._action_space_kwargs)
        self.action_space = self._action_space.space  # gym action space
        self.action_space.seed(derive_named_seed(inner_seed, "policy_action_space"))
        self.last_action = None

        # Setup observation space
        self._dacbo_observation_space = ObservationSpace(
            self._smac_instance,
            self._observation_keys,
            action_space=self._action_space,
        )
        self._dacbo_observation_space.reset()
        self.observation_space = self._dacbo_observation_space.space  # gym observation space
        self._validate_action_feature_space()

        # If previous_param is in obs, define the observation space for it
        if "previous_param" in self._dacbo_observation_space._keys:  # type: ignore
            self._dacbo_observation_space._observation_space["previous_param"] = self.action_space
            if isinstance(
                self._action_space,
                WEIDiscreteActionSpace
                | WEITempoRLActionSpace
                | PosteriorQuantileActionSpace
                | PosteriorModeActionSpace,
            ):
                self._dacbo_observation_space._observation_space["previous_param"] = Box(low=0, high=1)

        # Setup reward
        self._setup_reward(task_id)

        if not self._evaluation_mode:
            # Work off new initial design
            # This is important for training DAC policies because for the phase of the initial design, no action can
            # be taken and this might lead to misleading signals.
            # In evaluation, however, the initial design counts towards the total number of trials, controlled by
            # carps optimizer.
            initial_design_size = len(self._smac_instance.intensifier.config_selector._initial_design_configs)
            # One ask does not necessarily advance to a new configuration:
            # intensifiers may re-queue a configuration for additional calls.
            # Count configurations actually consumed instead of counting asks.
            while (
                len(self._smac_instance.runhistory.get_configs()) < initial_design_size
                and self._smac_instance.runhistory.finished < self._smac_instance._scenario.n_trials
            ):
                trial_info = self._smac_instance.ask()
                _, trial_value = self._smac_instance._runner.run_wrapper(trial_info)
                self._smac_instance.tell(trial_info, trial_value)
                self._check_reference_breach(trial_value)

        # The initial design is already part of the BO state. Returning hard
        # coded defaults would hide the consumed budget, fitted surrogate and
        # action consequences from the first policy decision.
        initial_obs = self._dacbo_observation_space.get_initial_observation()
        initial_obs = self.modify_obs(obs=initial_obs)

        info = self._context_info()
        diagnostics_fn = getattr(self._dacbo_observation_space, "get_action_feature_diagnostics", None)
        if callable(diagnostics_fn):
            info.update(diagnostics_fn())
        logger.info(f"Selected episode context: {info}")
        return initial_obs, info

    def get_n_finished_trials(self) -> int:
        """Get the number of told trials from the SMAC instance.

        Returns
        -------
        int
            Number of observations
        """
        return self._smac_instance._runhistory._finished
