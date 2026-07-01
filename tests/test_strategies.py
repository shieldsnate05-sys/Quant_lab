"""Tests for the strategies package."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import StrategyError
from indicators.trend import EMA
from strategies.ema_cross import EMACrossStrategy


def test_post_init_rejects_fast_not_less_than_slow() -> None:
    with pytest.raises(StrategyError, match="must be less than"):
        EMACrossStrategy(fast_period=20, slow_period=20)


def test_post_init_rejects_non_positive_periods() -> None:
    with pytest.raises(StrategyError, match="must be positive"):
        EMACrossStrategy(fast_period=0, slow_period=10)


def test_generate_signals_matches_manual_crossover(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    strategy = EMACrossStrategy(fast_period=5, slow_period=20)
    signals = strategy.generate_signals(trending_ohlcv_frame)

    fast = EMA(period=5).compute(trending_ohlcv_frame)
    slow = EMA(period=20).compute(trending_ohlcv_frame)
    warmed_up = fast.notna() & slow.notna()

    expected = pd.Series(0, index=trending_ohlcv_frame.index, dtype="int64")
    expected[warmed_up & (fast > slow)] = 1
    expected[warmed_up & (fast < slow)] = -1

    pd.testing.assert_series_equal(signals, expected, check_names=False)


def test_generate_signals_produces_no_shorts_when_disallowed(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    strategy = EMACrossStrategy(fast_period=5, slow_period=20, allow_shorting=False)
    signals = strategy.generate_signals(trending_ohlcv_frame)
    assert (signals >= 0).all()


def test_generate_signals_raises_on_insufficient_bars(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    strategy = EMACrossStrategy(fast_period=5, slow_period=20)
    with pytest.raises(StrategyError, match="fewer than the required"):
        strategy.generate_signals(trending_ohlcv_frame.iloc[:10])
