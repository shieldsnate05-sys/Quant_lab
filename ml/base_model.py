"""
Quant-Lab ML Model Interface.

Defines the common contract every ML model wrapper implements, so
training, evaluation, persistence, and explainability code can depend
on :class:`BaseModel` rather than a concrete estimator implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from core.enums import ModelType
from core.exceptions import ModelError
from core.types import FeatureFrame, PredictionArray


@runtime_checkable
class SklearnEstimator(Protocol):
    """Structural type for the subset of the sklearn estimator API we depend on."""

    def fit(self, X: object, y: object) -> object: ...

    def predict(self, X: object) -> object: ...

    def get_params(self, deep: bool = ...) -> dict[str, object]: ...

    def set_params(self, **params: object) -> object: ...


class BaseModel(ABC):
    """
    Abstract base class for all ML model wrappers.

    Concrete subclasses wrap a scikit-learn-compatible classifier
    (accessible via :attr:`estimator`), so generic sklearn utilities
    (``TimeSeriesSplit``-driven search, ``CalibratedClassifierCV``,
    SHAP's ``TreeExplainer``) can operate on it directly, while
    training/evaluation/persistence code depends only on this uniform
    interface.
    """

    #: Short, unique name this model is registered under (e.g. ``"RandomForest"``).
    name: ClassVar[str]

    #: The model family this wrapper belongs to.
    model_type: ClassVar[ModelType]

    @property
    @abstractmethod
    def estimator(self) -> SklearnEstimator:
        """The underlying (possibly unfitted) scikit-learn-compatible estimator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called on this model."""
        raise NotImplementedError

    @property
    @abstractmethod
    def classes_(self) -> np.ndarray:
        """
        Class labels in the order used by :meth:`predict_proba` columns.

        Raises
        ------
        core.exceptions.ModelError
            If the model has not been fitted yet.
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, X: FeatureFrame, y: pd.Series) -> BaseModel:
        """
        Fit this model on ``X``/``y``.

        Returns
        -------
        BaseModel
            ``self``, for chaining.

        Raises
        ------
        core.exceptions.ModelError
            If ``X`` is empty or ``X``/``y`` are misaligned.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: FeatureFrame) -> PredictionArray:
        """Predict class labels for ``X``."""
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, X: FeatureFrame) -> PredictionArray:
        """Predict class probabilities for ``X``, ordered per :attr:`classes_`."""
        raise NotImplementedError

    @abstractmethod
    def feature_importances(self) -> pd.Series:
        """Return per-feature importance scores, indexed by feature name, descending."""
        raise NotImplementedError

    @abstractmethod
    def with_estimator(self, estimator: SklearnEstimator) -> BaseModel:
        """
        Return a copy of this model wrapping a different fitted estimator.

        Used by :func:`ml.train.calibrate_model` to swap in a
        ``CalibratedClassifierCV``-wrapped estimator without mutating
        the original model in place.
        """
        raise NotImplementedError

    def _require_fitted(self) -> None:
        """
        Raise if this model has not been fitted yet.

        Raises
        ------
        core.exceptions.ModelError
            If :attr:`is_fitted` is ``False``.
        """
        if not self.is_fitted:
            raise ModelError(f"{self.name} model is not fitted yet. Call fit() first.")
