"""Shared pytest fixtures for the Quant-Lab test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import SCHEMA
from ml.dataset import (
    ChronologicalSplit,
    Dataset,
    build_dataset,
    make_forward_return_labels,
    train_validation_test_split,
)


def make_ohlcv_frame(
    n: int = 300,
    start_price: float = 100.0,
    seed: int = 7,
    trend: float = 0.0005,
) -> pd.DataFrame:
    """Build a deterministic, schema-valid synthetic OHLCV frame."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-02", periods=n, freq="1D", tz="UTC")

    returns = rng.normal(loc=trend, scale=0.01, size=n)
    close = start_price * np.cumprod(1.0 + returns)
    open_ = np.concatenate([[start_price], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.003, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.003, size=n))
    volume = rng.integers(1_000, 10_000, size=n).astype(float)

    frame = pd.DataFrame(
        {
            SCHEMA.open: open_,
            SCHEMA.high: high,
            SCHEMA.low: low,
            SCHEMA.close: close,
            SCHEMA.volume: volume,
        },
        index=index,
    )
    frame.index.name = SCHEMA.timestamp

    return frame


@pytest.fixture
def ohlcv_frame() -> pd.DataFrame:
    """A deterministic synthetic OHLCV frame with a mild uptrend."""
    return make_ohlcv_frame()


@pytest.fixture
def trending_ohlcv_frame() -> pd.DataFrame:
    """A short, strongly trending synthetic OHLCV frame."""
    return make_ohlcv_frame(n=120, trend=0.004, seed=11)


def make_ml_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build a small, deterministic feature frame from OHLCV bars, for ML tests."""
    close = frame[SCHEMA.close]
    features = pd.DataFrame(
        {
            "ret_1": close.pct_change(),
            "ret_5": close.pct_change(5),
            "vol_10": close.pct_change().rolling(10).std(),
        },
        index=frame.index,
    )
    features.index.name = SCHEMA.timestamp
    return features


@pytest.fixture
def ml_features(ohlcv_frame: pd.DataFrame) -> pd.DataFrame:
    """A small, deterministic feature frame aligned to `ohlcv_frame`."""
    return make_ml_features(ohlcv_frame)


@pytest.fixture
def ml_labels(ohlcv_frame: pd.DataFrame) -> pd.Series:
    """Forward-return classification labels aligned to `ohlcv_frame`."""
    return make_forward_return_labels(ohlcv_frame, horizon=1, threshold=0.0)


@pytest.fixture
def ml_dataset(ml_features: pd.DataFrame, ml_labels: pd.Series) -> Dataset:
    """A `ml.dataset.Dataset` built from `ml_features`/`ml_labels`, NaNs dropped."""
    return build_dataset(ml_features, ml_labels)


@pytest.fixture
def ml_split(ml_dataset: Dataset) -> ChronologicalSplit:
    """A chronological train/validation/test split of `ml_dataset`."""
    return train_validation_test_split(ml_dataset)
