"""Tests for ml.model_registry."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from core.enums import ModelType
from core.exceptions import ModelError
from core.types import FeatureFrame, PredictionArray
from ml.base_model import BaseModel, SklearnEstimator
from ml.model_registry import MODEL_REGISTRY, ModelRegistry
from ml.random_forest import RandomForestModel
from ml.xgboost_model import XGBoostModel


def test_random_forest_and_xgboost_are_registered() -> None:
    assert {"RandomForest", "XGBoost"} <= set(MODEL_REGISTRY.names())


def test_get_returns_the_registered_class() -> None:
    assert MODEL_REGISTRY.get("RandomForest") is RandomForestModel
    assert MODEL_REGISTRY.get("XGBoost") is XGBoostModel


def test_get_raises_on_unknown_name() -> None:
    with pytest.raises(ModelError, match="No model named 'NOPE'"):
        MODEL_REGISTRY.get("NOPE")


def test_create_instantiates_with_kwargs() -> None:
    model = MODEL_REGISTRY.create("RandomForest")
    assert isinstance(model, RandomForestModel)


def test_contains_and_len() -> None:
    assert "RandomForest" in MODEL_REGISTRY
    assert "NOPE" not in MODEL_REGISTRY
    assert len(MODEL_REGISTRY) >= 2


def test_register_raises_on_duplicate_name() -> None:
    registry = ModelRegistry()

    class _Dummy(BaseModel):
        name: ClassVar[str] = "DUMMY"
        model_type: ClassVar[ModelType] = ModelType.RANDOM_FOREST

        @property
        def estimator(self) -> SklearnEstimator:
            return self  # type: ignore[return-value]

        @property
        def is_fitted(self) -> bool:
            return False

        @property
        def classes_(self) -> np.ndarray:
            return np.array([])

        def fit(self, X: FeatureFrame, y: pd.Series) -> BaseModel:
            return self

        def predict(self, X: FeatureFrame) -> PredictionArray:
            return np.array([])

        def predict_proba(self, X: FeatureFrame) -> PredictionArray:
            return np.array([])

        def feature_importances(self) -> pd.Series:
            return pd.Series(dtype="float64")

        def with_estimator(self, estimator: SklearnEstimator) -> BaseModel:
            return self

    registry.register(_Dummy)

    with pytest.raises(ModelError, match="already registered"):
        registry.register(_Dummy)
