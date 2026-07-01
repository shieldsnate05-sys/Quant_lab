"""Tests for indicators.market_structure: Ichimoku."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import IndicatorError
from indicators.market_structure import Ichimoku


def test_ichimoku_columns(ohlcv_frame: pd.DataFrame) -> None:
    result = Ichimoku().compute(ohlcv_frame)
    assert list(result.columns) == [
        "tenkan_sen",
        "kijun_sen",
        "senkou_span_a",
        "senkou_span_b",
        "chikou_span",
    ]
    assert len(result) == len(ohlcv_frame)


def test_ichimoku_has_non_null_values_in_the_middle_of_the_series(
    ohlcv_frame: pd.DataFrame,
) -> None:
    result = Ichimoku().compute(ohlcv_frame)
    midpoint = len(result) // 2
    assert result.iloc[midpoint].notna().all()


def test_ichimoku_chikou_span_is_shifted_backward(ohlcv_frame: pd.DataFrame) -> None:
    result = Ichimoku(kijun_period=26).compute(ohlcv_frame)
    # chikou_span at bar i is close at bar i + kijun_period.
    assert result["chikou_span"].iloc[-27] == pytest.approx(
        ohlcv_frame["close"].iloc[-1]
    )
    assert result["chikou_span"].iloc[-26:].isnull().all()


def test_ichimoku_rejects_insufficient_rows(ohlcv_frame: pd.DataFrame) -> None:
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        Ichimoku(senkou_b_period=1000).compute(ohlcv_frame)
