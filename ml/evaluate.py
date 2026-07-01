"""
Quant-Lab Model Evaluation.

Produces a full evaluation report for a fitted model against a
held-out dataset: predictions, metrics, and feature importances in one
place.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.logging_config import get_logger
from core.exceptions import ModelError
from core.types import PredictionArray
from ml.base_model import BaseModel
from ml.dataset import Dataset
from ml.metrics import ClassificationMetrics, compute_classification_metrics

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    """
    Full evaluation of a fitted model against a dataset.

    Attributes
    ----------
    feature_importances : pandas.Series
        Empty if ``model``'s current estimator does not expose
        importances (e.g. after :func:`ml.train.calibrate_model`
        wraps it in a ``CalibratedClassifierCV``).
    """

    predictions: pd.Series
    probabilities: PredictionArray
    metrics: ClassificationMetrics
    feature_importances: pd.Series


def evaluate_model(model: BaseModel, dataset: Dataset) -> EvaluationReport:
    """
    Evaluate ``model`` against ``dataset``.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        A fitted model.
    dataset : ml.dataset.Dataset
        Held-out data (e.g. the ``test`` partition of a
        :class:`~ml.dataset.ChronologicalSplit`) to score against.

    Returns
    -------
    EvaluationReport
        Predictions, probabilities, metrics, and feature importances.

    Raises
    ------
    core.exceptions.ModelError
        If ``model`` is not fitted.
    """
    predictions_array = model.predict(dataset.X)
    probabilities = model.predict_proba(dataset.X)

    predictions = pd.Series(predictions_array, index=dataset.X.index, name="prediction")
    metrics = compute_classification_metrics(
        dataset.y, predictions_array, probabilities
    )

    logger.info(
        "Evaluated %s on %d samples: accuracy=%.4f f1=%.4f",
        model.name,
        len(dataset),
        metrics.accuracy,
        metrics.f1,
    )

    try:
        feature_importances = model.feature_importances()
    except ModelError:
        logger.warning(
            "%s's current estimator does not expose feature importances; "
            "EvaluationReport.feature_importances will be empty.",
            model.name,
        )
        feature_importances = pd.Series(dtype="float64", name="importance")

    return EvaluationReport(
        predictions=predictions,
        probabilities=probabilities,
        metrics=metrics,
        feature_importances=feature_importances,
    )
