from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from ConfigSpace import Configuration
from smac import BlackBoxFacade
from smac.acquisition.maximizer import LocalAndSortedRandomSearch
from smac.model.gaussian_process import GaussianProcess
from smac.model.random_forest import RandomForest
from smac.scenario import Scenario
from smac.utils.logging import get_logger

from dacboenv.utils.confidence_bound import LCB, UCB

if TYPE_CHECKING:
    from ConfigSpace import Configuration, ConfigurationSpace
    from smac.main.smbo import SMBO
    from smac.model import AbstractModel
    from smac.runhistory import TrialInfo, TrialValue

logger = get_logger(__name__)

from collections.abc import Iterable


def model_fitted(model: AbstractModel | None) -> bool:
    """Check whether the surrogate model is fitted.

    Parameters
    ----------
    model : AbstractModel
        Surrogate model.

    Returns:
    --------
    bool
        Model fitted or not.
    """
    fitted = False
    if model is not None:
        fitted = (type(model) == GaussianProcess and model._is_trained) or (
            type(model) == RandomForest and model._rf is not None
        )
    return fitted


def calculate_ubr(
    trial_infos: list[TrialInfo] | None,
    trial_values: list[TrialValue] | None,
    configspace: ConfigurationSpace | None,
    seed: int | None = None,
    top_p: float = 0.5,
    smbo: SMBO | None = None,
) -> dict[str, Any]:
    def dummy_fn(config: Configuration, seed: int | None) -> float:
        return 0

    if smbo is None:
        assert isinstance(trial_infos, Iterable)
        assert isinstance(trial_values, Iterable)

        scenario = Scenario(n_trials=10000, configspace=configspace, seed=seed)
        optimizer = BlackBoxFacade(
            scenario=scenario, target_function=dummy_fn, overwrite=True, logging_level=logging.WARNING
        )

        # Tell configs
        for info, value in zip(trial_infos, trial_values, strict=True):
            optimizer.tell(info, value)

        smbo = optimizer.optimizer

    model = smbo.intensifier.config_selector._model

    # Fit model if still model-free
    if len(smbo.intensifier.config_selector._initial_design_configs) > 0:
        X, Y, X_configurations = smbo.intensifier.config_selector._collect_data()
        smbo.intensifier.config_selector._runhistory.get_configs()
        smbo.intensifier.config_selector._model.train(X, Y)
        model = smbo.intensifier.config_selector._model

    rh = smbo.runhistory
    evaluated_configs = rh.get_configs(sort_by="cost")
    evaluated_configs = evaluated_configs[: int(np.ceil(len(evaluated_configs) * top_p))]

    ucb_aq = UCB()
    lcb_aq = LCB()

    kwargs = {"model": model, "num_data": rh.finished}
    ucb_aq.update(**kwargs)  # type: ignore[arg-type]
    lcb_aq.update(**kwargs)  # type: ignore[arg-type]

    # Minimize UCB (max -UCB) for all evaluated configs
    acq_values = ucb_aq(evaluated_configs)
    min_ucb = -float(np.squeeze(np.amax(acq_values)))

    # Minimize LCB (max -LCB) on config space
    acq_maximizer = LocalAndSortedRandomSearch(
        configspace=smbo._configspace,
        seed=smbo._scenario.seed,
        acquisition_function=lcb_aq,
    )
    challengers = np.array(
        acq_maximizer._maximize(
            previous_configs=[],
            n_points=1,
        ),
        dtype=object,
    )
    acq_values = challengers[:, 0]
    min_lcb = -float(np.squeeze(np.amax(acq_values)))

    ubr = min_ucb - min_lcb

    return {
        "n_evaluated": smbo.runhistory.finished,
        "ubr": ubr,
        "min_ucb": min_ucb,
        "min_lcb": min_lcb,
    }
