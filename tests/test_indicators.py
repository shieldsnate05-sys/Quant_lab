"""Tests for the indicators package."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from core.exceptions import IndicatorError
from indicators.momentum import rsi
from indicators.trend import ema, sma


@pytest.fixture
def close_series(ohlcv_frame: pd.DataFrame) -> pd.Series:
    return ohlcv_frame["close"]


def test_sma_matches_manual_rolling_mean(close_series: pd.Series) -> None:
    result = sma(close_series, period=10)
    expected = close_series.rolling(window=10, min_periods=10).mean()
    pd.testing.assert_series_equal(result, expected)


def test_ema_matches_manual_ewm(close_series: pd.Series) -> None:
    result = ema(close_series, period=12)
    expected = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
    pd.testing.assert_series_equal(result, expected)


def test_ema_warm_up_period_is_nan(close_series: pd.Series) -> None:
    result = ema(close_series, period=20)
    assert result.iloc[:19].isnull().all()
    assert result.iloc[19:].notnull().all()


def test_rsi_bounds(close_series: pd.Series) -> None:
    result = rsi(close_series, period=14)
    valid = result.dropna()
    assert (valid >= 0).all()
    assert (valid <= 100).all()


def test_rsi_is_high_for_monotonically_rising_prices() -> None:
    index = pd.date_range("2024-01-01", periods=30, freq="1D", tz="UTC")
    prices = pd.Series(range(100, 130), index=index, dtype=float)

    result = rsi(prices, period=14)

    assert result.iloc[-1] == pytest.approx(100.0)


Indicator = Callable[..., pd.Series]


@pytest.mark.parametrize("indicator", [sma, ema, rsi])
def test_indicator_rejects_non_positive_period(
    indicator: Indicator, close_series: pd.Series
) -> None:
    with pytest.raises(IndicatorError, match="period must be positive"):
        indicator(close_series, period=0)


@pytest.mark.parametrize("indicator", [sma, ema, rsi])
def test_indicator_rejects_empty_series(indicator: Indicator) -> None:
    empty = pd.Series([], dtype=float)
    with pytest.raises(IndicatorError, match="empty"):
        indicator(empty, period=5)


@pytest.mark.parametrize("indicator", [sma, ema, rsi])
def test_indicator_rejects_series_shorter_than_period(indicator: Indicator) -> None:
    short = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        indicator(short, period=10)
