"""Tests for ml.predict."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.exceptions import ModelError
from data.storage import ParquetStorage
from ml.config import RandomForestConfig
from ml.dataset import ChronologicalSplit
from ml.predict import predict, predict_proba, predict_with_confidence, save_predictions
from ml.random_forest import RandomForestModel


@pytest.fixture
def fitted_model(ml_split: ChronologicalSplit) -> RandomForestModel:
    return RandomForestModel(RandomForestConfig(n_estimators=20)).fit(
        ml_split.train.X, ml_split.train.y
    )


def test_predict_returns_series_aligned_to_index(
    fitted_model: RandomForestModel, ml_split: ChronologicalSplit
) -> None:
    result = predict(fitted_model, ml_split.test.X)
    assert result.name == "prediction"
    pd.testing.assert_index_equal(result.index, ml_split.test.X.index)


def test_predict_rejects_empty_frame(fitted_model: RandomForestModel) -> None:
    with pytest.raises(ModelError, match="empty"):
        predict(fitted_model, pd.DataFrame())


def test_predict_proba_columns_match_classes(
    fitted_model: RandomForestModel, ml_split: ChronologicalSplit
) -> None:
    result = predict_proba(fitted_model, ml_split.test.X)
    expected_columns = [f"proba_{label}" for label in fitted_model.classes_]
    assert list(result.columns) == expected_columns
    assert ((result.sum(axis=1) - 1.0).abs() < 1e-6).all()


def test_predict_with_confidence_matches_max_proba(
    fitted_model: RandomForestModel, ml_split: ChronologicalSplit
) -> None:
    result = predict_with_confidence(fitted_model, ml_split.test.X)
    proba = predict_proba(fitted_model, ml_split.test.X)

    assert list(result.columns) == ["prediction", "confidence"]
    pd.testing.assert_series_equal(
        result["confidence"], proba.max(axis=1).rename("confidence")
    )


def test_save_predictions_writes_parquet(
    fitted_model: RandomForestModel, ml_split: ChronologicalSplit, tmp_path: Path
) -> None:
    predictions = predict_with_confidence(fitted_model, ml_split.test.X)
    storage = ParquetStorage(base_dir=tmp_path)
    fixed_timestamp = pd.Timestamp("2026-01-01T00:00:00Z").to_pydatetime()

    save_predictions(
        predictions, "RandomForest", storage=storage, timestamp=fixed_timestamp
    )

    expected_path = storage.resolve_path("RandomForest_20260101T000000Z.parquet")
    assert expected_path.exists()
