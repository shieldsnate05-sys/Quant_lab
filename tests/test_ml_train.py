"""Tests for ml.train."""

from __future__ import annotations

import pytest

from core.exceptions import ModelError
from ml.config import CalibrationConfig, RandomForestConfig, RandomForestSearchSpace
from ml.dataset import ChronologicalSplit
from ml.random_forest import RandomForestModel
from ml.train import calibrate_model, train_model, tune_hyperparameters


def test_train_model_fits_and_returns_the_model(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20))
    trained = train_model(model, ml_split.train)

    assert trained is model
    assert trained.is_fitted


def test_tune_hyperparameters_rejects_empty_grid(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20))
    with pytest.raises(ModelError, match="param_grid must not be empty"):
        tune_hyperparameters(model, ml_split.train, {}, n_splits=3)


def test_tune_hyperparameters_returns_a_model_and_best_params(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20))
    search_space = RandomForestSearchSpace(
        n_estimators=(10, 20), max_depth=(3, 4), min_samples_leaf=(10, 20)
    )

    result = tune_hyperparameters(
        model, ml_split.train, search_space.as_param_grid(), n_splits=3
    )

    assert result.model.is_fitted
    assert set(result.best_params) == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert isinstance(result.best_score, float)


def test_calibrate_model_returns_a_new_instance_with_valid_probabilities(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )

    calibrated = calibrate_model(model, ml_split.validation, CalibrationConfig(cv=3))

    assert calibrated is not model
    assert calibrated.is_fitted

    proba = calibrated.predict_proba(ml_split.test.X)
    assert (proba.sum(axis=1) - 1.0 < 1e-6).all()


def test_calibrate_model_uses_default_config_when_none_given(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )
    calibrated = calibrate_model(model, ml_split.validation)
    assert calibrated.is_fitted
