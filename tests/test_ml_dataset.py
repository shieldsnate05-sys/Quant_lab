"""Tests for ml.dataset."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import ModelError
from ml.dataset import (
    Dataset,
    build_dataset,
    make_forward_return_labels,
    train_validation_test_split,
)


def test_make_forward_return_labels_values(ohlcv_frame: pd.DataFrame) -> None:
    labels = make_forward_return_labels(ohlcv_frame, horizon=1, threshold=0.0)

    close = ohlcv_frame["close"]
    expected_up = (close.shift(-1) / close - 1.0) > 0
    assert (labels[expected_up] == 1.0).all()
    assert labels.iloc[-1:].isna().all()


def test_make_forward_return_labels_respects_threshold(
    ohlcv_frame: pd.DataFrame,
) -> None:
    labels = make_forward_return_labels(ohlcv_frame, horizon=1, threshold=0.05)
    valid = labels.dropna()
    assert (valid == 0.0).any()  # a 5% threshold should flatten most bars


def test_make_forward_return_labels_rejects_non_positive_horizon(
    ohlcv_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ModelError, match="horizon must be positive"):
        make_forward_return_labels(ohlcv_frame, horizon=0)


def test_make_forward_return_labels_rejects_negative_threshold(
    ohlcv_frame: pd.DataFrame,
) -> None:
    with pytest.raises(ModelError, match="threshold must be non-negative"):
        make_forward_return_labels(ohlcv_frame, threshold=-0.1)


def test_build_dataset_drops_nan_rows(
    ml_features: pd.DataFrame, ml_labels: pd.Series
) -> None:
    dataset = build_dataset(ml_features, ml_labels)
    assert len(dataset) < len(ml_features)
    assert not dataset.X.isnull().any().any()
    assert not dataset.y.isnull().any()


def test_build_dataset_rejects_misaligned_index(
    ml_features: pd.DataFrame, ml_labels: pd.Series
) -> None:
    with pytest.raises(ModelError, match="same index"):
        build_dataset(ml_features, ml_labels.iloc[:-1])


def test_dataset_rejects_length_mismatch() -> None:
    X = pd.DataFrame({"a": [1.0, 2.0]}, index=pd.RangeIndex(2))
    y = pd.Series([1], index=pd.RangeIndex(1))
    with pytest.raises(ModelError, match="must match"):
        Dataset(X, y)


def test_dataset_rejects_empty_frame() -> None:
    empty_index = pd.RangeIndex(0)
    with pytest.raises(ModelError, match="empty"):
        Dataset(
            pd.DataFrame(index=empty_index), pd.Series(dtype="int64", index=empty_index)
        )


def test_dataset_feature_names_and_len(ml_dataset: Dataset) -> None:
    assert ml_dataset.feature_names == list(ml_dataset.X.columns)
    assert len(ml_dataset) == len(ml_dataset.X)


def test_train_validation_test_split_partitions_are_contiguous_in_time(
    ml_dataset: Dataset,
) -> None:
    split = train_validation_test_split(ml_dataset)

    assert len(split.train) + len(split.validation) + len(split.test) == len(ml_dataset)
    assert split.train.X.index[-1] < split.validation.X.index[0]
    assert split.validation.X.index[-1] < split.test.X.index[0]


def test_train_validation_test_split_rejects_invalid_fractions(
    ml_dataset: Dataset,
) -> None:
    with pytest.raises(ModelError, match="in \\(0, 1\\)"):
        train_validation_test_split(
            ml_dataset, train_fraction=1.5, validation_fraction=0.1
        )

    with pytest.raises(ModelError, match="leave room"):
        train_validation_test_split(
            ml_dataset, train_fraction=0.6, validation_fraction=0.5
        )


def test_train_validation_test_split_rejects_too_small_dataset(
    ml_dataset: Dataset,
) -> None:
    tiny = Dataset(ml_dataset.X.iloc[:2], ml_dataset.y.iloc[:2])
    with pytest.raises(ModelError, match="too few"):
        train_validation_test_split(tiny)
