"""Shared pytest fixtures for the Quant-Lab test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.schema import SCHEMA


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
