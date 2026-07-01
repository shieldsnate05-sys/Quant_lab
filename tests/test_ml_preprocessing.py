"""Tests for ml.preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.exceptions import ModelError
from ml.preprocessing import Preprocessor


@pytest.fixture
def feature_frame_with_nans() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, np.nan, 30.0, 40.0, 50.0],
        }
    )


def test_fit_transform_fills_nans(feature_frame_with_nans: pd.DataFrame) -> None:
    preprocessor = Preprocessor()
    transformed = preprocessor.fit_transform(feature_frame_with_nans)
    assert not transformed.isnull().any().any()


def test_fit_transform_scales_to_zero_mean(
    feature_frame_with_nans: pd.DataFrame,
) -> None:
    preprocessor = Preprocessor(scale=True)
    transformed = preprocessor.fit_transform(feature_frame_with_nans)
    assert transformed["a"].mean() == pytest.approx(0.0, abs=1e-9)


def test_scale_false_skips_standardization(
    feature_frame_with_nans: pd.DataFrame,
) -> None:
    preprocessor = Preprocessor(scale=False)
    transformed = preprocessor.fit_transform(feature_frame_with_nans)
    assert transformed["a"].mean() != pytest.approx(0.0, abs=1e-9)


def test_transform_before_fit_raises(feature_frame_with_nans: pd.DataFrame) -> None:
    preprocessor = Preprocessor()
    with pytest.raises(ModelError, match="not fitted"):
        preprocessor.transform(feature_frame_with_nans)


def test_fit_rejects_empty_frame() -> None:
    preprocessor = Preprocessor()
    with pytest.raises(ModelError, match="empty"):
        preprocessor.fit(pd.DataFrame())


def test_transform_rejects_mismatched_columns(
    feature_frame_with_nans: pd.DataFrame,
) -> None:
    preprocessor = Preprocessor().fit(feature_frame_with_nans)
    with pytest.raises(ModelError, match="expected"):
        preprocessor.transform(pd.DataFrame({"a": [1.0], "c": [2.0]}))


def test_is_fitted_flag(feature_frame_with_nans: pd.DataFrame) -> None:
    preprocessor = Preprocessor()
    assert not preprocessor.is_fitted
    preprocessor.fit(feature_frame_with_nans)
    assert preprocessor.is_fitted


def test_transform_preserves_index_and_columns(
    feature_frame_with_nans: pd.DataFrame,
) -> None:
    preprocessor = Preprocessor().fit(feature_frame_with_nans)
    transformed = preprocessor.transform(feature_frame_with_nans)
    pd.testing.assert_index_equal(transformed.index, feature_frame_with_nans.index)
    assert list(transformed.columns) == list(feature_frame_with_nans.columns)
