"""
Quant-Lab XGBoost Model.

A :class:`~ml.base_model.BaseModel` wrapping ``xgboost.XGBClassifier``.

XGBoost's sklearn API requires integer class labels in
``[0, n_classes)``, unlike scikit-learn's own classifiers which accept
arbitrary labels (e.g. ``{-1, 0, 1}``). This wrapper transparently
label-encodes on :meth:`fit` and decodes on :meth:`predict`, so callers
see the original label space everywhere, exactly like
:class:`~ml.random_forest.RandomForestModel`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from config.logging_config import get_logger
from core.enums import ModelType
from core.exceptions import ModelError
from core.types import FeatureFrame, PredictionArray
from ml.base_model import BaseModel, SklearnEstimator
from ml.config import XGBoostConfig
from ml.model_registry import MODEL_REGISTRY

logger = get_logger(__name__)


@MODEL_REGISTRY.register
@dataclass(slots=True)
class XGBoostModel(BaseModel):
    """Gradient-boosted tree classifier for direction/label prediction."""

    name: ClassVar[str] = "XGBoost"
    model_type: ClassVar[ModelType] = ModelType.XGBOOST

    config: XGBoostConfig = field(default_factory=XGBoostConfig)

    _estimator: XGBClassifier = field(init=False, repr=False)
    _label_encoder: LabelEncoder = field(init=False, repr=False)
    _is_fitted: bool = field(init=False, repr=False, default=False)
    _feature_names: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._estimator = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            reg_alpha=self.config.reg_alpha,
            reg_lambda=self.config.reg_lambda,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
            eval_metric="logloss",
        )
        self._label_encoder = LabelEncoder()

    @property
    def estimator(self) -> SklearnEstimator:
        """The underlying ``XGBClassifier``."""
        return self._estimator

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called on this model."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Class labels (original label space) used by :meth:`predict_proba` columns."""
        self._require_fitted()
        return np.asarray(self._label_encoder.classes_)

    def fit(self, X: FeatureFrame, y: pd.Series) -> XGBoostModel:
        """
        Fit the gradient-boosted classifier on ``X``/``y``.

        Raises
        ------
        core.exceptions.ModelError
            If ``X`` is empty.
        """
        if X.empty:
            raise ModelError("Cannot fit XGBoostModel on an empty feature frame.")

        self._feature_names = list(X.columns)
        encoded_y = self._label_encoder.fit_transform(y)
        self._estimator.fit(X, encoded_y)
        self._is_fitted = True

        logger.info(
            "Fitted XGBoostModel on %d rows, %d features.",
            len(X),
            len(self._feature_names),
        )

        return self

    def predict(self, X: FeatureFrame) -> PredictionArray:
        """Predict class labels for ``X``, decoded back to the original label space."""
        self._require_fitted()
        encoded = self._estimator.predict(X)
        return np.asarray(self._label_encoder.inverse_transform(encoded))

    def predict_proba(self, X: FeatureFrame) -> PredictionArray:
        """Predict class probabilities for ``X``."""
        self._require_fitted()
        return np.asarray(self._estimator.predict_proba(X))

    def feature_importances(self) -> pd.Series:
        """
        Return gain-based feature importances, indexed by feature name, descending.

        Raises
        ------
        core.exceptions.ModelError
            If the model is not fitted, or its estimator was replaced
            (e.g. by :func:`ml.train.calibrate_model`) with one that
            does not expose ``feature_importances_``.
        """
        self._require_fitted()

        try:
            importances = self._estimator.feature_importances_
        except AttributeError as exc:
            raise ModelError(
                f"{self.name}'s current estimator ({type(self._estimator).__name__}) "
                "does not expose feature_importances_."
            ) from exc

        return pd.Series(
            importances, index=self._feature_names, name="importance"
        ).sort_values(ascending=False)

    def with_estimator(self, estimator: SklearnEstimator) -> XGBoostModel:
        """Return a copy of this model wrapping a different fitted estimator."""
        clone = XGBoostModel(config=self.config)
        clone._estimator = estimator  # type: ignore[assignment]
        clone._label_encoder = self._label_encoder
        clone._is_fitted = True
        clone._feature_names = list(self._feature_names)
        return clone
