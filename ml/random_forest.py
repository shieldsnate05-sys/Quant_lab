"""
Quant-Lab Random Forest Model.

A :class:`~ml.base_model.BaseModel` wrapping scikit-learn's
``RandomForestClassifier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config.logging_config import get_logger
from core.enums import ModelType
from core.exceptions import ModelError
from core.types import FeatureFrame, PredictionArray
from ml.base_model import BaseModel, SklearnEstimator
from ml.config import RandomForestConfig
from ml.model_registry import MODEL_REGISTRY

logger = get_logger(__name__)


@MODEL_REGISTRY.register
@dataclass(slots=True)
class RandomForestModel(BaseModel):
    """Random forest classifier for direction/label prediction."""

    name: ClassVar[str] = "RandomForest"
    model_type: ClassVar[ModelType] = ModelType.RANDOM_FOREST

    config: RandomForestConfig = field(default_factory=RandomForestConfig)

    _estimator: RandomForestClassifier = field(init=False, repr=False)
    _is_fitted: bool = field(init=False, repr=False, default=False)
    _feature_names: list[str] = field(init=False, repr=False, default_factory=list)

    def __post_init__(self) -> None:
        self._estimator = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )

    @property
    def estimator(self) -> SklearnEstimator:
        """The underlying ``RandomForestClassifier``."""
        return cast(SklearnEstimator, self._estimator)

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called on this model."""
        return self._is_fitted

    @property
    def classes_(self) -> np.ndarray:
        """Class labels in the order used by :meth:`predict_proba` columns."""
        self._require_fitted()
        return np.asarray(self._estimator.classes_)

    def fit(self, X: FeatureFrame, y: pd.Series) -> RandomForestModel:
        """
        Fit the random forest on ``X``/``y``.

        Raises
        ------
        core.exceptions.ModelError
            If ``X`` is empty.
        """
        if X.empty:
            raise ModelError("Cannot fit RandomForestModel on an empty feature frame.")

        self._feature_names = list(X.columns)
        self._estimator.fit(X, y)
        self._is_fitted = True

        logger.info(
            "Fitted RandomForestModel on %d rows, %d features.",
            len(X),
            len(self._feature_names),
        )

        return self

    def predict(self, X: FeatureFrame) -> PredictionArray:
        """Predict class labels for ``X``."""
        self._require_fitted()
        return np.asarray(self._estimator.predict(X))

    def predict_proba(self, X: FeatureFrame) -> PredictionArray:
        """Predict class probabilities for ``X``."""
        self._require_fitted()
        return np.asarray(self._estimator.predict_proba(X))

    def feature_importances(self) -> pd.Series:
        """
        Return Gini-based feature importances, indexed by feature name, descending.

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

    def with_estimator(self, estimator: SklearnEstimator) -> RandomForestModel:
        """Return a copy of this model wrapping a different fitted estimator."""
        clone = RandomForestModel(config=self.config)
        clone._estimator = estimator
        clone._is_fitted = True
        clone._feature_names = list(self._feature_names)
        return clone
