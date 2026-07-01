"""Tests for indicators.volume: VWAP."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import IndicatorError
from data.schema import SCHEMA
from indicators.volume import VWAP


def test_vwap_anchored_matches_manual_calculation(ohlcv_frame: pd.DataFrame) -> None:
    result = VWAP().compute(ohlcv_frame)

    typical_price = (
        ohlcv_frame[SCHEMA.high] + ohlcv_frame[SCHEMA.low] + ohlcv_frame[SCHEMA.close]
    ) / 3.0
    expected = (typical_price * ohlcv_frame[SCHEMA.volume]).cumsum() / ohlcv_frame[
        SCHEMA.volume
    ].cumsum()

    pd.testing.assert_series_equal(result, expected, check_names=False)
    assert result.name == "VWAP"


def test_vwap_rolling_window_has_warm_up_nans(ohlcv_frame: pd.DataFrame) -> None:
    result = VWAP(window=20).compute(ohlcv_frame)
    assert result.iloc[:19].isnull().all()
    assert result.iloc[19:].notnull().all()


def test_vwap_lies_within_cumulative_high_low_envelope(
    ohlcv_frame: pd.DataFrame,
) -> None:
    # An anchored VWAP is a volume-weighted average of each bar's typical
    # price, so it can only drift outside the *running* high/low envelope
    # accumulated so far - not necessarily the current bar's own range.
    result = VWAP().compute(ohlcv_frame)
    running_low = ohlcv_frame[SCHEMA.low].cummin()
    running_high = ohlcv_frame[SCHEMA.high].cummax()
    assert (result >= running_low - 1e-9).all()
    assert (result <= running_high + 1e-9).all()


def test_vwap_rejects_non_positive_window(ohlcv_frame: pd.DataFrame) -> None:
    with pytest.raises(IndicatorError, match="window must be positive"):
        VWAP(window=0).compute(ohlcv_frame)
