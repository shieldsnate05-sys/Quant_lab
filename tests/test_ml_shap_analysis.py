"""Tests for ml.shap_analysis."""

from __future__ import annotations

import pytest

from core.exceptions import ModelError
from ml.config import RandomForestConfig, XGBoostConfig
from ml.dataset import ChronologicalSplit
from ml.random_forest import RandomForestModel
from ml.shap_analysis import compute_shap_values
from ml.xgboost_model import XGBoostModel


def test_compute_shap_values_rejects_unfitted_model(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10))
    with pytest.raises(ModelError, match="not fitted"):
        compute_shap_values(model, ml_split.test.X)


def test_compute_shap_values_rejects_empty_frame(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    with pytest.raises(ModelError, match="empty"):
        compute_shap_values(model, ml_split.test.X.iloc[:0])


def test_compute_shap_values_random_forest_shape(ml_split: ChronologicalSplit) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    sample = ml_split.test.X.iloc[:10]

    result = compute_shap_values(model, sample)

    assert result.values.shape == (10, sample.shape[1])
    assert result.feature_names == tuple(sample.columns)


def test_compute_shap_values_xgboost_shape(ml_split: ChronologicalSplit) -> None:
    model = XGBoostModel(XGBoostConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    sample = ml_split.test.X.iloc[:10]

    result = compute_shap_values(model, sample)

    assert result.values.shape == (10, sample.shape[1])


def test_mean_absolute_importance_is_non_negative_and_covers_all_features(
    ml_split: ChronologicalSplit,
) -> None:
    model = RandomForestModel(RandomForestConfig(n_estimators=10)).fit(
        ml_split.train.X, ml_split.train.y
    )
    sample = ml_split.test.X.iloc[:15]

    result = compute_shap_values(model, sample)
    importance = result.mean_absolute_importance()

    assert set(importance.index) == set(sample.columns)
    assert (importance >= 0).all()
    assert list(importance.values) == sorted(importance.values, reverse=True)
