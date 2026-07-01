"""Tests for ml.feature_selection."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import ModelError
from ml.feature_selection import (
    drop_highly_correlated_features,
    drop_low_variance_features,
    select_top_k_by_importance,
)


@pytest.fixture
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "constant": [1.0, 1.0, 1.0, 1.0],
            "varying": [1.0, 2.0, 3.0, 4.0],
            "duplicate_of_varying": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_drop_low_variance_features_removes_constant_column(
    features: pd.DataFrame,
) -> None:
    result = drop_low_variance_features(features)
    assert "constant" not in result.columns
    assert "varying" in result.columns


def test_drop_low_variance_features_raises_when_all_dropped() -> None:
    all_constant = pd.DataFrame({"a": [1.0, 1.0], "b": [2.0, 2.0]})
    with pytest.raises(ModelError, match="variance"):
        drop_low_variance_features(all_constant)


def test_drop_highly_correlated_features_removes_duplicate(
    features: pd.DataFrame,
) -> None:
    result = drop_highly_correlated_features(
        features[["varying", "duplicate_of_varying"]]
    )
    assert list(result.columns) == ["varying"]


def test_drop_highly_correlated_features_keeps_uncorrelated_columns() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 1.0, 3.0, 2.0]})
    result = drop_highly_correlated_features(frame, threshold=0.5)
    assert list(result.columns) == list(frame.columns)


def test_select_top_k_by_importance(features: pd.DataFrame) -> None:
    importances = pd.Series(
        {"constant": 0.0, "varying": 0.9, "duplicate_of_varying": 0.5}
    )
    result = select_top_k_by_importance(features, importances, k=2)
    assert list(result.columns) == ["varying", "duplicate_of_varying"]


def test_select_top_k_by_importance_rejects_non_positive_k(
    features: pd.DataFrame,
) -> None:
    importances = pd.Series(dict.fromkeys(features.columns, 1.0))
    with pytest.raises(ModelError, match="k must be positive"):
        select_top_k_by_importance(features, importances, k=0)


def test_select_top_k_by_importance_rejects_missing_scores(
    features: pd.DataFrame,
) -> None:
    importances = pd.Series({"varying": 1.0})
    with pytest.raises(ModelError, match="missing scores"):
        select_top_k_by_importance(features, importances, k=1)
