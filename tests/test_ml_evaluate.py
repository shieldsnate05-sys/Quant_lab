"""Tests for ml.evaluate."""

from __future__ import annotations

import pandas as pd

from ml.config import CalibrationConfig, RandomForestConfig
from ml.dataset import ChronologicalSplit
from ml.evaluate import evaluate_model
from ml.random_forest import RandomForestModel
from ml.train import calibrate_model


def test_evaluate_model_returns_a_full_report(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )

    report = evaluate_model(model, ml_split.test)

    assert len(report.predictions) == len(ml_split.test)
    assert report.probabilities.shape[0] == len(ml_split.test)
    assert 0 <= report.metrics.accuracy <= 1
    assert not report.feature_importances.empty
    assert set(report.feature_importances.index) == set(ml_split.test.feature_names)


def test_evaluate_model_after_calibration_has_empty_importances(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )
    calibrated = calibrate_model(model, ml_split.validation, CalibrationConfig(cv=3))

    report = evaluate_model(calibrated, ml_split.test)

    assert isinstance(report.feature_importances, pd.Series)
    assert report.feature_importances.empty
    assert 0 <= report.metrics.accuracy <= 1
