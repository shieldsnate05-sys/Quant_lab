"""Tests for indicators.momentum: RSI, Stochastic."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import IndicatorError
from data.schema import SCHEMA
from indicators.momentum import RSI, Stochastic


def test_rsi_bounds(ohlcv_frame: pd.DataFrame) -> None:
    result = RSI().compute(ohlcv_frame)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()
    assert result.name == "RSI"


def test_rsi_is_high_for_monotonically_rising_prices() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    frame = pd.DataFrame(
        {
            SCHEMA.open: range(100, 130),
            SCHEMA.high: range(101, 131),
            SCHEMA.low: range(99, 129),
            SCHEMA.close: range(100, 130),
            SCHEMA.volume: [10.0] * 30,
        },
        index=index,
    ).astype(float)
    frame.index.name = SCHEMA.timestamp

    result = RSI(period=14).compute(frame)

    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_rejects_insufficient_rows(ohlcv_frame: pd.DataFrame) -> None:
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        RSI(period=1000).compute(ohlcv_frame)


def test_stochastic_bounds_and_columns(ohlcv_frame: pd.DataFrame) -> None:
    result = Stochastic().compute(ohlcv_frame)

    assert list(result.columns) == ["percent_k", "percent_d"]
    valid = result.dropna()
    assert not valid.empty
    assert (valid["percent_k"] >= 0).all()
    assert (valid["percent_k"] <= 100).all()
    assert (valid["percent_d"] >= 0).all()
    assert (valid["percent_d"] <= 100).all()
