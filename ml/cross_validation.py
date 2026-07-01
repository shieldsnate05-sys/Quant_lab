"""
Quant-Lab Cross-Validation.

Time-series-aware cross-validation: wraps scikit-learn's
``TimeSeriesSplit`` (which never lets a fold's training data come
after its test data) and runs a model factory across folds,
aggregating metrics per fold.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sklearn.model_selection import TimeSeriesSplit

from config.logging_config import get_logger
from core.exceptions import ModelError
from ml.base_model import BaseModel
from ml.dataset import Dataset
from ml.metrics import ClassificationMetrics, compute_classification_metrics

logger = get_logger(__name__)


def make_time_series_split(n_splits: int = 5, *, gap: int = 0) -> TimeSeriesSplit:
    """
    Build a ``sklearn.model_selection.TimeSeriesSplit`` for chronological CV.

    Parameters
    ----------
    n_splits : int, optional
        Number of folds. Defaults to 5.
    gap : int, optional
        Number of samples to exclude between the end of a training
        fold and the start of its test fold, to reduce leakage from
        overlapping label horizons. Defaults to 0.

    Returns
    -------
    sklearn.model_selection.TimeSeriesSplit

    Raises
    ------
    core.exceptions.ModelError
        If ``n_splits`` is less than 2.
    """
    if n_splits < 2:
        raise ModelError(f"n_splits must be at least 2, got {n_splits}.")

    return TimeSeriesSplit(n_splits=n_splits, gap=gap)


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Per-fold metrics from a cross-validation run."""

    fold_metrics: tuple[ClassificationMetrics, ...]

    @property
    def mean_accuracy(self) -> float:
        """Mean accuracy across all folds."""
        return sum(m.accuracy for m in self.fold_metrics) / len(self.fold_metrics)

    @property
    def mean_f1(self) -> float:
        """Mean macro-averaged F1 score across all folds."""
        return sum(m.f1 for m in self.fold_metrics) / len(self.fold_metrics)


def cross_validate_model(
    model_factory: Callable[[], BaseModel],
    dataset: Dataset,
    *,
    n_splits: int = 5,
    gap: int = 0,
) -> CrossValidationResult:
    """
    Run time-series cross-validation for a model.

    Parameters
    ----------
    model_factory : Callable[[], BaseModel]
        Builds a fresh, unfitted model instance for each fold, so
        folds don't share state (e.g. ``lambda: RandomForestModel()``).
    dataset : ml.dataset.Dataset
        The full dataset to split into folds.
    n_splits : int, optional
        Number of folds. Defaults to 5.
    gap : int, optional
        Samples to exclude between train and test within each fold.
        Defaults to 0.

    Returns
    -------
    CrossValidationResult
        Metrics for every fold.
    """
    splitter = make_time_series_split(n_splits, gap=gap)
    fold_metrics: list[ClassificationMetrics] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(dataset.X), start=1):
        X_train, y_train = dataset.X.iloc[train_idx], dataset.y.iloc[train_idx]
        X_test, y_test = dataset.X.iloc[test_idx], dataset.y.iloc[test_idx]

        model = model_factory().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(metrics)

        logger.info(
            "Fold %d/%d: accuracy=%.4f f1=%.4f",
            fold,
            n_splits,
            metrics.accuracy,
            metrics.f1,
        )

    return CrossValidationResult(fold_metrics=tuple(fold_metrics))
