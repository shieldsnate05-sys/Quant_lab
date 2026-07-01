"""Tests for ml.xgboost_model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.exceptions import ModelError
from ml.config import XGBoostConfig
from ml.xgboost_model import XGBoostModel


@pytest.fixture
def toy_data() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(np.where(X["f1"] + X["f2"] > 0, 1, -1))
    return X, y


def test_fit_sets_is_fitted(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    model = XGBoostModel(XGBoostConfig(n_estimators=20))
    assert not model.is_fitted
    model.fit(X, y)
    assert model.is_fitted


def test_predict_before_fit_raises(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, _ = toy_data
    model = XGBoostModel(XGBoostConfig(n_estimators=20))
    with pytest.raises(ModelError, match="not fitted"):
        model.predict(X)


def test_fit_rejects_empty_frame() -> None:
    model = XGBoostModel(XGBoostConfig(n_estimators=20))
    with pytest.raises(ModelError, match="empty"):
        model.fit(pd.DataFrame(), pd.Series(dtype="int64"))


def test_predict_and_predict_proba_shapes(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(XGBoostConfig(n_estimators=20)).fit(X, y)

    predictions = model.predict(X)
    proba = model.predict_proba(X)

    assert predictions.shape == (len(X),)
    assert proba.shape == (len(X), 2)
    assert set(predictions).issubset({-1, 1})


def test_handles_negative_and_zero_labels_via_internal_encoding() -> None:
    # XGBoost's sklearn API natively requires labels in [0, n_classes) -
    # this is the label space {-1, 0, 1} that would fail without the
    # internal LabelEncoder round-trip (see class docstring).
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(150, 2)), columns=["f1", "f2"])
    y = pd.Series(np.select([X["f1"] > 0.3, X["f1"] < -0.3], [1, -1], default=0))

    model = XGBoostModel(XGBoostConfig(n_estimators=20)).fit(X, y)

    assert set(model.classes_) == {-1, 0, 1}
    assert set(model.predict(X)).issubset({-1, 0, 1})


def test_feature_importances_order(toy_data: tuple[pd.DataFrame, pd.Series]) -> None:
    X, y = toy_data
    model = XGBoostModel(XGBoostConfig(n_estimators=50)).fit(X, y)

    importances = model.feature_importances()

    assert set(importances.index) == set(X.columns)
    assert list(importances.values) == sorted(importances.values, reverse=True)


def test_with_estimator_produces_independent_fitted_copy(
    toy_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_data
    model = XGBoostModel(XGBoostConfig(n_estimators=20)).fit(X, y)

    clone = model.with_estimator(model.estimator)

    assert clone is not model
    assert clone.is_fitted
    np.testing.assert_array_equal(clone.predict(X), model.predict(X))
