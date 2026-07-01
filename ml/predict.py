"""
Quant-Lab Prediction API.

Runs a fitted model over new data and (optionally) persists the
predictions to disk as Parquet, reusing
:class:`data.storage.ParquetStorage` rather than a second I/O layer.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from config.logging_config import get_logger
from config.paths import PREDICTIONS
from core.exceptions import ModelError
from core.types import FeatureFrame
from data.storage import ParquetStorage
from ml.base_model import BaseModel

logger = get_logger(__name__)


def predict(model: BaseModel, X: FeatureFrame) -> pd.Series:
    """
    Predict class labels for ``X``.

    Returns
    -------
    pandas.Series
        Predicted labels, named ``"prediction"``, aligned to ``X.index``.

    Raises
    ------
    core.exceptions.ModelError
        If ``X`` is empty.
    """
    if X.empty:
        raise ModelError("Cannot predict on an empty feature frame.")

    return pd.Series(model.predict(X), index=X.index, name="prediction")


def predict_proba(model: BaseModel, X: FeatureFrame) -> pd.DataFrame:
    """
    Predict class probabilities for ``X``.

    Returns
    -------
    pandas.DataFrame
        One ``proba_<class>`` column per class in
        :attr:`~ml.base_model.BaseModel.classes_`, aligned to ``X.index``.

    Raises
    ------
    core.exceptions.ModelError
        If ``X`` is empty.
    """
    if X.empty:
        raise ModelError("Cannot predict on an empty feature frame.")

    proba = model.predict_proba(X)
    columns = [f"proba_{label}" for label in model.classes_]

    return pd.DataFrame(proba, index=X.index, columns=columns)


def predict_with_confidence(model: BaseModel, X: FeatureFrame) -> pd.DataFrame:
    """
    Predict labels alongside the model's confidence in that label.

    Returns
    -------
    pandas.DataFrame
        Columns ``prediction`` and ``confidence`` (the winning class's
        predicted probability), aligned to ``X.index``.
    """
    labels = predict(model, X)
    proba = predict_proba(model, X)
    confidence = proba.max(axis=1).rename("confidence")

    return pd.concat([labels, confidence], axis=1)


def save_predictions(
    predictions: pd.DataFrame,
    model_name: str,
    *,
    storage: ParquetStorage | None = None,
    timestamp: datetime | None = None,
) -> None:
    """
    Persist ``predictions`` to :data:`config.paths.PREDICTIONS` as Parquet.

    Parameters
    ----------
    predictions : pandas.DataFrame
        The predictions to persist (e.g. from :func:`predict_with_confidence`).
    model_name : str
        The model's :attr:`~ml.base_model.BaseModel.name`, used in the file name.
    storage : data.storage.ParquetStorage | None, optional
        Storage backend to write through. Defaults to a
        :class:`~data.storage.ParquetStorage` rooted at
        :data:`config.paths.PREDICTIONS`.
    timestamp : datetime.datetime | None, optional
        Timestamp used in the file name. Defaults to now (UTC).
    """
    storage = storage or ParquetStorage(base_dir=PREDICTIONS)
    timestamp = timestamp or datetime.now(UTC)

    path = storage.resolve_path(f"{model_name}_{timestamp:%Y%m%dT%H%M%SZ}.parquet")
    storage.write(path, predictions)

    logger.info(
        "Saved %d predictions for %s to %s.", len(predictions), model_name, path
    )
