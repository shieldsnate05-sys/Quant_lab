"""
Quant-Lab ML Preprocessing.

Feature scaling and missing-value imputation, wrapped in a typed,
logged interface around scikit-learn transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config.logging_config import get_logger
from core.exceptions import ModelError
from core.types import FeatureFrame

logger = get_logger(__name__)


@dataclass(slots=True)
class Preprocessor:
    """
    Imputes missing values and (optionally) standardizes features.

    Fits an imputer and scaler on training data via :meth:`fit`, then
    applies both consistently to any subsequent data via
    :meth:`transform`.

    Parameters
    ----------
    impute_strategy : str, optional
        Strategy passed to ``sklearn.impute.SimpleImputer``. Defaults
        to ``"median"``.
    scale : bool, optional
        Whether to standardize features (zero mean, unit variance)
        after imputation. Defaults to ``True``.
    """

    impute_strategy: str = "median"
    scale: bool = True

    _imputer: SimpleImputer = field(init=False, repr=False)
    _scaler: StandardScaler | None = field(init=False, repr=False, default=None)
    _feature_names: list[str] = field(init=False, repr=False, default_factory=list)
    _is_fitted: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._imputer = SimpleImputer(strategy=self.impute_strategy)
        self._scaler = StandardScaler() if self.scale else None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called on this preprocessor."""
        return self._is_fitted

    def fit(self, X: FeatureFrame) -> Preprocessor:
        """
        Fit the imputer (and scaler, if enabled) on ``X``.

        Returns
        -------
        Preprocessor
            ``self``, for chaining.

        Raises
        ------
        core.exceptions.ModelError
            If ``X`` is empty.
        """
        if X.empty:
            raise ModelError("Cannot fit a Preprocessor on an empty feature frame.")

        self._feature_names = list(X.columns)
        imputed = self._imputer.fit_transform(X)
        if self._scaler is not None:
            self._scaler.fit(imputed)
        self._is_fitted = True

        logger.info(
            "Fitted Preprocessor on %d rows, %d features.",
            len(X),
            len(self._feature_names),
        )

        return self

    def transform(self, X: FeatureFrame) -> FeatureFrame:
        """
        Apply the fitted imputer (and scaler) to ``X``.

        Raises
        ------
        core.exceptions.ModelError
            If this preprocessor is not fitted, or ``X``'s columns
            don't match the columns it was fitted on.
        """
        self._require_fitted()

        if list(X.columns) != self._feature_names:
            raise ModelError(
                f"X has columns {list(X.columns)}, expected {self._feature_names}."
            )

        transformed = self._imputer.transform(X)
        if self._scaler is not None:
            transformed = self._scaler.transform(transformed)

        return pd.DataFrame(transformed, index=X.index, columns=self._feature_names)

    def fit_transform(self, X: FeatureFrame) -> FeatureFrame:
        """Fit this preprocessor on ``X`` and immediately transform it."""
        return self.fit(X).transform(X)

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise ModelError("Preprocessor is not fitted yet. Call fit() first.")
