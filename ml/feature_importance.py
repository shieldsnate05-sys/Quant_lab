"""
Quant-Lab Feature Importance.

Ranks features by a fitted model's built-in importances and by
model-agnostic permutation importance.
"""

from __future__ import annotations

import pandas as pd
from sklearn.inspection import permutation_importance

from config.logging_config import get_logger
from config.settings import settings
from core.exceptions import ModelError
from ml.base_model import BaseModel
from ml.dataset import Dataset

logger = get_logger(__name__)


def built_in_importance(model: BaseModel) -> pd.Series:
    """
    Return ``model``'s built-in feature importances, descending.

    A thin, explicit wrapper around
    :meth:`ml.base_model.BaseModel.feature_importances`, so callers
    importing from :mod:`ml.feature_importance` get both importance
    methods from one module.
    """
    return model.feature_importances()


def permutation_feature_importance(
    model: BaseModel,
    dataset: Dataset,
    *,
    n_repeats: int = 10,
    scoring: str = "f1_macro",
    random_state: int | None = None,
) -> pd.Series:
    """
    Compute model-agnostic permutation importance on ``dataset``.

    Unlike built-in importances (which reflect how a tree model used a
    feature internally), permutation importance measures the drop in
    ``scoring`` when a feature's values are randomly shuffled - so it
    works for any fitted estimator and better reflects predictive
    usefulness.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        A fitted model.
    dataset : ml.dataset.Dataset
        Data to permute features on (typically a held-out partition).
    n_repeats : int, optional
        Number of times to permute each feature. Defaults to 10.
    scoring : str, optional
        An sklearn scoring string. Defaults to ``"f1_macro"``.
    random_state : int | None, optional
        Seed for the permutations. Defaults to
        :data:`config.settings.settings.ml.random_state`.

    Returns
    -------
    pandas.Series
        Mean importance (score drop) per feature, descending.

    Raises
    ------
    core.exceptions.ModelError
        If ``model`` is not fitted.
    """
    if not model.is_fitted:
        raise ModelError(
            f"Cannot compute permutation importance: {model.name} is not fitted."
        )

    random_state = (
        random_state if random_state is not None else settings.ml.random_state
    )

    result = permutation_importance(
        model.estimator,
        dataset.X,
        dataset.y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=settings.ml.n_jobs,
    )

    logger.info(
        "Computed permutation importance for %s over %d repeats.", model.name, n_repeats
    )

    importances = pd.Series(
        result.importances_mean, index=dataset.feature_names, name="importance"
    )

    return importances.sort_values(ascending=False)
