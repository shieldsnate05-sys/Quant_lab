"""
Quant-Lab SHAP Analysis.

Computes SHAP (SHapley Additive exPlanations) values for tree-based
models (:class:`~ml.random_forest.RandomForestModel`,
:class:`~ml.xgboost_model.XGBoostModel`) via ``shap.TreeExplainer``,
and summarizes them into a per-feature importance ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap

from config.logging_config import get_logger
from core.exceptions import ModelError
from core.types import FeatureFrame
from ml.base_model import BaseModel

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ShapResult:
    """SHAP values for a batch of samples, for a single (target) class."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    base_value: float

    def mean_absolute_importance(self) -> pd.Series:
        """Per-feature mean |SHAP value|, descending - a global importance summary."""
        importance = pd.Series(
            np.abs(self.values).mean(axis=0),
            index=self.feature_names,
            name="mean_abs_shap",
        )
        return importance.sort_values(ascending=False)


def compute_shap_values(
    model: BaseModel, X: FeatureFrame, *, class_index: int = -1
) -> ShapResult:
    """
    Compute SHAP values for ``X`` using ``model``'s fitted tree estimator.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        A fitted tree-based model.
    X : core.types.FeatureFrame
        Samples to explain.
    class_index : int, optional
        For multi-class models, which class's SHAP values to return
        (indexing into the last axis of the raw SHAP output, which
        follows the estimator's internal class order). Defaults to
        ``-1`` (the last/highest-index class - the "up" label when
        classes are ``{-1, 0, 1}``, sorted ascending). Ignored for
        estimators that already produce a single 2D SHAP matrix (e.g.
        binary XGBoost classifiers).

    Returns
    -------
    ShapResult

    Raises
    ------
    core.exceptions.ModelError
        If ``model`` is not fitted or ``X`` is empty.
    """
    if not model.is_fitted:
        raise ModelError(f"Cannot compute SHAP values: {model.name} is not fitted.")
    if X.empty:
        raise ModelError("Cannot compute SHAP values on an empty feature frame.")

    explainer = shap.TreeExplainer(model.estimator)
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        values = np.asarray(raw[class_index])
    elif raw.ndim == 3:
        values = raw[:, :, class_index]
    else:
        values = raw

    base_value = explainer.expected_value
    if isinstance(base_value, list | np.ndarray):
        base_value = base_value[class_index]

    logger.info(
        "Computed SHAP values for %s over %d samples, %d features.",
        model.name,
        len(X),
        X.shape[1],
    )

    return ShapResult(
        values=values, feature_names=tuple(X.columns), base_value=float(base_value)
    )
