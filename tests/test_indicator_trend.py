"""Tests for indicators.trend: SMA, EMA, MACD, ParabolicSAR."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pytest

from core.enums import IndicatorGroup
from core.exceptions import IndicatorError
from data.schema import SCHEMA
from indicators.base import Indicator
from indicators.trend import EMA, MACD, SMA, ParabolicSAR

IndicatorFactory = Callable[..., Indicator]


def test_sma_matches_manual_rolling_mean(ohlcv_frame: pd.DataFrame) -> None:
    result = SMA(period=10).compute(ohlcv_frame)
    expected = ohlcv_frame[SCHEMA.close].rolling(window=10, min_periods=10).mean()
    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert result.name == "SMA"


def test_ema_matches_manual_ewm(ohlcv_frame: pd.DataFrame) -> None:
    result = EMA(period=12).compute(ohlcv_frame)
    expected = (
        ohlcv_frame[SCHEMA.close].ewm(span=12, adjust=False, min_periods=12).mean()
    )
    pd.testing.assert_series_equal(result, expected, check_names=False)


def test_ema_warm_up_period_is_nan(ohlcv_frame: pd.DataFrame) -> None:
    result = EMA(period=20).compute(ohlcv_frame)
    assert result.iloc[:19].isnull().all()
    assert result.iloc[19:].notnull().all()


def test_indicator_class_attributes() -> None:
    assert EMA.name == "EMA"
    assert EMA.group is IndicatorGroup.TREND
    assert SMA.name == "SMA"


@pytest.mark.parametrize("indicator_cls", [SMA, EMA])
def test_indicator_rejects_insufficient_rows(
    indicator_cls: IndicatorFactory, ohlcv_frame: pd.DataFrame
) -> None:
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        indicator_cls(period=1000).compute(ohlcv_frame)


def test_macd_columns_and_relationship(ohlcv_frame: pd.DataFrame) -> None:
    result = MACD().compute(ohlcv_frame)

    assert list(result.columns) == ["macd", "signal", "histogram"]
    valid = result.dropna()
    assert not valid.empty
    pd.testing.assert_series_equal(
        valid["histogram"],
        valid["macd"] - valid["signal"],
        check_names=False,
    )


def test_macd_rejects_fast_not_less_than_slow() -> None:
    with pytest.raises(IndicatorError, match="must be less than"):
        MACD(fast_period=26, slow_period=12)


def test_parabolic_sar_stays_within_high_low_envelope(
    trending_ohlcv_frame: pd.DataFrame,
) -> None:
    result = ParabolicSAR().compute(trending_ohlcv_frame)

    assert result.name == "PSAR"
    assert len(result) == len(trending_ohlcv_frame)
    assert result.notna().all()
    envelope_low = trending_ohlcv_frame[SCHEMA.low].min()
    envelope_high = trending_ohlcv_frame[SCHEMA.high].max()
    assert (result >= envelope_low - 1e-6).all()
    assert (result <= envelope_high + 1e-6).all()


def test_parabolic_sar_requires_at_least_two_bars(ohlcv_frame: pd.DataFrame) -> None:
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        ParabolicSAR().compute(ohlcv_frame.iloc[:1])
