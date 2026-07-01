"""
Quant-Lab Model Training.

Orchestrates fitting a :class:`~ml.base_model.BaseModel`, optional
hyperparameter tuning (via time-series-aware search), and probability
calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

from config.logging_config import get_logger
from core.exceptions import ModelError
from ml.base_model import BaseModel
from ml.config import CalibrationConfig
from ml.cross_validation import make_time_series_split
from ml.dataset import Dataset

logger = get_logger(__name__)


def train_model(model: BaseModel, dataset: Dataset) -> BaseModel:
    """
    Fit ``model`` on ``dataset``.

    A thin, explicit entry point so training always goes through one
    place in the codebase, even though it is a one-line call to
    :meth:`~ml.base_model.BaseModel.fit`.

    Returns
    -------
    ml.base_model.BaseModel
        The fitted ``model`` (fit mutates and returns ``self``).
    """
    return model.fit(dataset.X, dataset.y)


@dataclass(slots=True)
class TuningResult:
    """Result of a hyperparameter search."""

    model: BaseModel
    best_params: dict[str, Any]
    best_score: float


def tune_hyperparameters(
    model: BaseModel,
    dataset: Dataset,
    param_grid: dict[str, list[Any]],
    *,
    n_splits: int = 5,
    scoring: str = "f1_macro",
) -> TuningResult:
    """
    Search ``param_grid`` for the best hyperparameters via time-series CV.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        The (unfitted) model whose estimator will be tuned.
    dataset : ml.dataset.Dataset
        Training data to search over.
    param_grid : dict[str, list[Any]]
        Parameter grid, e.g. from
        :meth:`ml.config.RandomForestSearchSpace.as_param_grid`.
    n_splits : int, optional
        Number of ``TimeSeriesSplit`` folds used to score each
        parameter combination. Defaults to 5.
    scoring : str, optional
        An sklearn scoring string. Defaults to ``"f1_macro"``.

    Returns
    -------
    TuningResult
        A new model wrapping the best-found estimator, its parameters,
        and its cross-validated score.

    Raises
    ------
    core.exceptions.ModelError
        If ``param_grid`` is empty.
    """
    if not param_grid:
        raise ModelError("param_grid must not be empty.")

    splitter = make_time_series_split(n_splits)
    search = GridSearchCV(
        model.estimator, param_grid=param_grid, cv=splitter, scoring=scoring
    )
    search.fit(dataset.X, dataset.y)

    tuned_model = model.with_estimator(search.best_estimator_)

    logger.info(
        "Hyperparameter search for %s complete: best_score=%.4f params=%s",
        model.name,
        search.best_score_,
        search.best_params_,
    )

    return TuningResult(
        model=tuned_model,
        best_params=dict(search.best_params_),
        best_score=float(search.best_score_),
    )


def calibrate_model(
    model: BaseModel,
    dataset: Dataset,
    config: CalibrationConfig | None = None,
) -> BaseModel:
    """
    Calibrate ``model``'s predicted probabilities via ``CalibratedClassifierCV``.

    Parameters
    ----------
    model : ml.base_model.BaseModel
        The model whose estimator is calibrated. The estimator is
        cloned and refit internally by ``CalibratedClassifierCV``'s
        own cross-validation, so ``model`` need not be pre-fitted.
    dataset : ml.dataset.Dataset
        Data used to fit the calibrator.
    config : ml.config.CalibrationConfig | None, optional
        Calibration method and CV folds. Defaults to
        ``CalibrationConfig()`` (sigmoid, 5-fold).

    Returns
    -------
    ml.base_model.BaseModel
        A new model instance wrapping the calibrated estimator.
    """
    config = config or CalibrationConfig()

    calibrated_estimator = CalibratedClassifierCV(
        estimator=model.estimator, method=config.method, cv=config.cv
    )
    calibrated_estimator.fit(dataset.X, dataset.y)

    logger.info(
        "Calibrated %s using method=%s, cv=%d.", model.name, config.method, config.cv
    )

    return model.with_estimator(calibrated_estimator)
