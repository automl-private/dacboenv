"""Handles the MSE (Mean Squared Error)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from smac.facade.blackbox_facade import BlackBoxFacade
from smac.model.gaussian_process import GaussianProcess

if TYPE_CHECKING:
    from smac.main.smbo import SMBO


def calculate_mse(smbo: SMBO, k: int = 10) -> float:
    """Computes the MSE (model-fit) of a model.

    Parameters
    ----------
    model : SMBO
        SMAC instance
    k : int
        How many folds to use for cross-validation

    Returns:
    ----------
    float
        MSE after k-fold cross-validation
    """
    if len(smbo.runhistory) < k:
        return np.nan

    X, y, _ = smbo.intensifier.config_selector._collect_data()

    # Only take best quartile
    sort_indices = np.argsort(y.squeeze())
    X = X[sort_indices].squeeze()
    y = y[sort_indices].squeeze()

    quartile = len(X) // 4

    # If quartile is too small, best k configs
    quartile = max(k, quartile)

    X = X[:quartile]
    y = y[:quartile]

    if len(X) < k or len(X) != len(y):
        return np.nan

    kf = KFold(n_splits=k, shuffle=True)
    scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        gp = GaussianProcess(
            configspace=smbo._intensifier._config_selector._model._configspace,
            kernel=BlackBoxFacade.get_kernel(smbo._scenario),
            seed=smbo._intensifier._config_selector._model._seed,
        )

        gp.train(X_train, y_train)

        y_pred, _ = gp.predict(X_test)
        scores.append(mean_squared_error(y_pred, y_test))

    return np.mean(scores)
