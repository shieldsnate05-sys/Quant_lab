"""Tests for ml.base_model."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from core.enums import ModelType
from core.exceptions import ModelError
from core.types import FeatureFrame, PredictionArray
from ml.base_model import BaseModel, SklearnEstimator


class _StubModel(BaseModel):
    """Minimal concrete BaseModel for exercising the shared ABC behavior."""

    name: ClassVar[str] = "Stub"
    model_type: ClassVar[ModelType] = ModelType.RANDOM_FOREST

    def __init__(self) -> None:
        self._fitted = False

    @property
    def estimator(self) -> SklearnEstimator:
        return self  # type: ignore[return-value]

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def classes_(self) -> np.ndarray:
        self._require_fitted()
        return np.array([0, 1])

    def fit(self, X: FeatureFrame, y: pd.Series) -> _StubModel:
        self._fitted = True
        return self

    def predict(self, X: FeatureFrame) -> PredictionArray:
        self._require_fitted()
        return np.zeros(len(X))

    def predict_proba(self, X: FeatureFrame) -> PredictionArray:
        self._require_fitted()
        return np.zeros((len(X), 2))

    def feature_importances(self) -> pd.Series:
        self._require_fitted()
        return pd.Series(dtype="float64")

    def with_estimator(self, estimator: SklearnEstimator) -> _StubModel:
        clone = _StubModel()
        clone._fitted = True
        return clone


def test_base_model_cannot_be_instantiated_directly() -> None:
    with pytest.raises(TypeError):
        BaseModel()  # type: ignore[abstract]


def test_require_fitted_raises_before_fit() -> None:
    model = _StubModel()
    with pytest.raises(ModelError, match="not fitted"):
        model.predict(pd.DataFrame({"a": [1.0]}))


def test_require_fitted_passes_after_fit() -> None:
    model = _StubModel()
    model.fit(pd.DataFrame({"a": [1.0]}), pd.Series([1]))
    assert model.is_fitted
    assert model.predict(pd.DataFrame({"a": [1.0]})).shape == (1,)


def test_with_estimator_returns_a_fitted_copy() -> None:
    model = _StubModel()
    clone = model.with_estimator(model.estimator)
    assert clone.is_fitted
    assert clone is not model
