"""Tests for the backtesting package."""

from __future__ import annotations

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from config.settings import BacktestSettings
from core.exceptions import BacktestError
from strategies.base import Strategy


class _AlwaysLongStrategy(Strategy):
    """Test double: constant long signal on every bar."""

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(1, index=frame.index, dtype="int64")


class _AlwaysFlatStrategy(Strategy):
    """Test double: constant flat signal on every bar."""

    def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=frame.index, dtype="int64")


def test_engine_rejects_frames_shorter_than_two_bars(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    engine = BacktestEngine()
    with pytest.raises(BacktestError, match="fewer than two bars"):
        engine.run(trending_ohlcv_frame.iloc[:1], _AlwaysLongStrategy())


def test_always_long_equity_curve_matches_manual_calculation(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    zero_cost = BacktestSettings(
        initial_cash=10_000.0,
        commission_per_trade=0.0,
        slippage=0.0,
        allow_shorting=True,
    )
    engine = BacktestEngine(backtest_settings=zero_cost)

    result = engine.run(trending_ohlcv_frame, _AlwaysLongStrategy())

    close = trending_ohlcv_frame["close"]
    market_returns = close.pct_change().fillna(0.0)
    position = (
        pd.Series(1, index=trending_ohlcv_frame.index, dtype=float).shift(1).fillna(0.0)
    )
    expected_equity = (
        zero_cost.initial_cash * (1.0 + position * market_returns).cumprod()
    )

    pd.testing.assert_series_equal(
        result.equity_curve, expected_equity, check_names=False
    )
    assert result.metrics.total_return == pytest.approx(
        expected_equity.iloc[-1] / expected_equity.iloc[0] - 1.0
    )


def test_always_flat_strategy_has_zero_return(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    engine = BacktestEngine()
    result = engine.run(trending_ohlcv_frame, _AlwaysFlatStrategy())

    assert result.metrics.total_return == pytest.approx(0.0)
    assert result.metrics.num_trades == 0


def test_commission_reduces_returns_relative_to_zero_cost(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    zero_cost = BacktestSettings(commission_per_trade=0.0, slippage=0.0)
    with_cost = BacktestSettings(commission_per_trade=0.01, slippage=0.0)

    zero_cost_result = BacktestEngine(backtest_settings=zero_cost).run(
        trending_ohlcv_frame, _AlwaysLongStrategy()
    )
    with_cost_result = BacktestEngine(backtest_settings=with_cost).run(
        trending_ohlcv_frame, _AlwaysLongStrategy()
    )

    assert (
        with_cost_result.equity_curve.iloc[-1] < zero_cost_result.equity_curve.iloc[-1]
    )


def test_disallowing_shorts_clips_position_to_non_negative(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    class _AlwaysShortStrategy(Strategy):
        def generate_signals(self, frame: pd.DataFrame) -> pd.Series:
            return pd.Series(-1, index=frame.index, dtype="int64")

    settings_no_short = BacktestSettings(allow_shorting=False)
    engine = BacktestEngine(backtest_settings=settings_no_short)

    result = engine.run(trending_ohlcv_frame, _AlwaysShortStrategy())

    assert (result.positions >= 0).all()
