"""
Quant-Lab ML Configuration.

Model- and validation-specific configuration dataclasses. Global,
cross-cutting settings (train/validation/test fractions, random_state,
n_jobs) live in :class:`config.settings.MLSettings`; this module holds
parameters specific to individual estimators and validation schemes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from config.settings import settings
from core.exceptions import ConfigurationError


@dataclass(slots=True)
class RandomForestConfig:
    """Hyperparameters for :class:`~ml.random_forest.RandomForestModel`."""

    n_estimators: int = 300
    max_depth: int | None = 6
    min_samples_leaf: int = 20
    max_features: str | float = "sqrt"
    class_weight: str | None = "balanced"
    random_state: int = field(default_factory=lambda: settings.ml.random_state)
    n_jobs: int = field(default_factory=lambda: settings.ml.n_jobs)


@dataclass(slots=True)
class RandomForestSearchSpace:
    """Hyperparameter search space for tuning :class:`RandomForestConfig`."""

    n_estimators: tuple[int, ...] = (100, 200, 300, 500)
    max_depth: tuple[int | None, ...] = (4, 6, 8, None)
    min_samples_leaf: tuple[int, ...] = (5, 10, 20, 50)

    def as_param_grid(self) -> dict[str, list[Any]]:
        """Return this search space as an sklearn-compatible ``param_grid``."""
        return {
            "n_estimators": list(self.n_estimators),
            "max_depth": list(self.max_depth),
            "min_samples_leaf": list(self.min_samples_leaf),
        }


@dataclass(slots=True)
class XGBoostConfig:
    """Hyperparameters for :class:`~ml.xgboost_model.XGBoostModel`."""

    n_estimators: int = 300
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = field(default_factory=lambda: settings.ml.random_state)
    n_jobs: int = field(default_factory=lambda: settings.ml.n_jobs)


@dataclass(slots=True)
class XGBoostSearchSpace:
    """Hyperparameter search space for tuning :class:`XGBoostConfig`."""

    n_estimators: tuple[int, ...] = (100, 200, 300, 500)
    max_depth: tuple[int, ...] = (3, 4, 6, 8)
    learning_rate: tuple[float, ...] = (0.01, 0.05, 0.1, 0.2)

    def as_param_grid(self) -> dict[str, list[Any]]:
        """Return this search space as an sklearn-compatible ``param_grid``."""
        return {
            "n_estimators": list(self.n_estimators),
            "max_depth": list(self.max_depth),
            "learning_rate": list(self.learning_rate),
        }


@dataclass(slots=True)
class WalkForwardConfig:
    """
    Configuration for walk-forward validation.

    Attributes
    ----------
    train_window : int
        Number of bars in each training window.
    test_window : int
        Number of bars in each out-of-sample test window.
    step : int
        Number of bars to advance between successive windows.
    expanding : bool
        If ``True``, the training window grows from the start of the
        data on every step (anchored/expanding). If ``False`` (the
        default), it is a fixed-size rolling window.
    """

    train_window: int = 252
    test_window: int = 63
    step: int = 63
    expanding: bool = False

    def __post_init__(self) -> None:
        if self.train_window <= 0 or self.test_window <= 0 or self.step <= 0:
            raise ConfigurationError(
                "train_window, test_window, and step must all be positive."
            )


@dataclass(slots=True)
class CalibrationConfig:
    """
    Configuration for probability calibration.

    Attributes
    ----------
    method : str
        ``"sigmoid"`` (Platt scaling) or ``"isotonic"``.
    cv : int
        Number of cross-validation folds used to fit the calibrator.
    """

    method: str = "sigmoid"
    cv: int = 5

    def __post_init__(self) -> None:
        if self.method not in {"sigmoid", "isotonic"}:
            raise ConfigurationError(
                f"method must be 'sigmoid' or 'isotonic', got '{self.method}'."
            )
        if self.cv < 2:
            raise ConfigurationError(f"cv must be at least 2, got {self.cv}.")
