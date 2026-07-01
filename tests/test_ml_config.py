"""Tests for ml.config."""

from __future__ import annotations

import pytest

from core.exceptions import ConfigurationError
from ml.config import (
    CalibrationConfig,
    RandomForestConfig,
    RandomForestSearchSpace,
    WalkForwardConfig,
    XGBoostConfig,
    XGBoostSearchSpace,
)


def test_random_forest_config_defaults() -> None:
    config = RandomForestConfig()
    assert config.n_estimators > 0
    assert config.random_state >= 0


def test_random_forest_search_space_as_param_grid() -> None:
    grid = RandomForestSearchSpace().as_param_grid()
    assert set(grid) == {"n_estimators", "max_depth", "min_samples_leaf"}
    assert all(isinstance(v, list) for v in grid.values())


def test_xgboost_config_defaults() -> None:
    config = XGBoostConfig()
    assert config.n_estimators > 0
    assert 0 < config.learning_rate <= 1


def test_xgboost_search_space_as_param_grid() -> None:
    grid = XGBoostSearchSpace().as_param_grid()
    assert set(grid) == {"n_estimators", "max_depth", "learning_rate"}


def test_walk_forward_config_rejects_non_positive_windows() -> None:
    with pytest.raises(ConfigurationError):
        WalkForwardConfig(train_window=0)
    with pytest.raises(ConfigurationError):
        WalkForwardConfig(test_window=-1)
    with pytest.raises(ConfigurationError):
        WalkForwardConfig(step=0)


def test_calibration_config_rejects_invalid_method() -> None:
    with pytest.raises(ConfigurationError, match="sigmoid"):
        CalibrationConfig(method="not-a-method")


def test_calibration_config_rejects_cv_below_two() -> None:
    with pytest.raises(ConfigurationError, match="cv"):
        CalibrationConfig(cv=1)


def test_calibration_config_accepts_isotonic() -> None:
    config = CalibrationConfig(method="isotonic", cv=3)
    assert config.method == "isotonic"
