"""
Quant-Lab Walk-Forward Validation.

Repeatedly trains on a window of history and evaluates strictly
out-of-sample on the window immediately following it, then advances
through time - the standard quant-research validation scheme. This is
distinct from :mod:`ml.cross_validation`'s ``TimeSeriesSplit`` folds:
walk-forward windows are fixed-size (or expanding) and slide by a
fixed step, mirroring how a strategy would actually be retrained
periodically in production.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from config.logging_config import get_logger
from core.exceptions import ModelError
from ml.base_model import BaseModel
from ml.config import WalkForwardConfig
from ml.dataset import Dataset
from ml.metrics import ClassificationMetrics, compute_classification_metrics

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    """One train/test window from a walk-forward run."""

    window_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    metrics: ClassificationMetrics
    predictions: pd.Series


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Full walk-forward validation result: every window, in time order."""

    windows: tuple[WalkForwardWindow, ...]

    @property
    def out_of_sample_predictions(self) -> pd.Series:
        """All windows' test-set predictions, concatenated in time order."""
        concatenated: pd.Series = pd.concat([w.predictions for w in self.windows])
        return concatenated.sort_index()

    @property
    def mean_accuracy(self) -> float:
        """Mean accuracy across all windows."""
        return sum(w.metrics.accuracy for w in self.windows) / len(self.windows)

    @property
    def mean_f1(self) -> float:
        """Mean macro-averaged F1 across all windows."""
        return sum(w.metrics.f1 for w in self.windows) / len(self.windows)


def run_walk_forward(
    model_factory: Callable[[], BaseModel],
    dataset: Dataset,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """
    Run walk-forward validation over ``dataset``.

    Parameters
    ----------
    model_factory : Callable[[], BaseModel]
        Builds a fresh, unfitted model for every window.
    dataset : ml.dataset.Dataset
        The full, chronologically-ordered dataset.
    config : ml.config.WalkForwardConfig | None, optional
        Window sizes and step. Defaults to ``WalkForwardConfig()``.

    Returns
    -------
    WalkForwardResult
        Every window's out-of-sample metrics and predictions.

    Raises
    ------
    core.exceptions.ModelError
        If ``dataset`` has fewer rows than ``train_window + test_window``.
    """
    config = config or WalkForwardConfig()
    n = len(dataset)

    if n < config.train_window + config.test_window:
        raise ModelError(
            f"Dataset has {n} rows, fewer than train_window + test_window "
            f"({config.train_window + config.test_window})."
        )

    windows: list[WalkForwardWindow] = []
    test_start_pos = config.train_window
    window_index = 0

    while test_start_pos + config.test_window <= n:
        train_start_pos = (
            0 if config.expanding else test_start_pos - config.train_window
        )
        test_end_pos = test_start_pos + config.test_window

        X_train = dataset.X.iloc[train_start_pos:test_start_pos]
        y_train = dataset.y.iloc[train_start_pos:test_start_pos]
        X_test = dataset.X.iloc[test_start_pos:test_end_pos]
        y_test = dataset.y.iloc[test_start_pos:test_end_pos]

        model = model_factory().fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        predictions = pd.Series(y_pred, index=X_test.index, name="prediction")

        windows.append(
            WalkForwardWindow(
                window_index=window_index,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                metrics=metrics,
                predictions=predictions,
            )
        )

        logger.info(
            "Walk-forward window %d: train=[%s, %s] test=[%s, %s] accuracy=%.4f",
            window_index,
            X_train.index[0],
            X_train.index[-1],
            X_test.index[0],
            X_test.index[-1],
            metrics.accuracy,
        )

        window_index += 1
        test_start_pos += config.step

    if not windows:
        raise ModelError(
            "No walk-forward windows were produced; check config against dataset size."
        )

    return WalkForwardResult(windows=tuple(windows))
