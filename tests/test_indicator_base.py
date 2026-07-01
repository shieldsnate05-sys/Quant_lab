"""Tests for indicators.base."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import DataError, IndicatorError
from indicators.base import validate_ohlcv_for_indicator


def test_validate_ohlcv_for_indicator_accepts_valid_frame(
    ohlcv_frame: pd.DataFrame,
) -> None:
    validate_ohlcv_for_indicator(ohlcv_frame, period=10)  # should not raise


def test_validate_ohlcv_for_indicator_allows_no_period(
    ohlcv_frame: pd.DataFrame,
) -> None:
    validate_ohlcv_for_indicator(ohlcv_frame)  # should not raise


def test_validate_ohlcv_for_indicator_rejects_non_positive_period(
    ohlcv_frame: pd.DataFrame,
) -> None:
    with pytest.raises(IndicatorError, match="period must be positive"):
        validate_ohlcv_for_indicator(ohlcv_frame, period=0)


def test_validate_ohlcv_for_indicator_rejects_insufficient_rows(
    ohlcv_frame: pd.DataFrame,
) -> None:
    with pytest.raises(IndicatorError, match="fewer than the required period"):
        validate_ohlcv_for_indicator(ohlcv_frame.iloc[:5], period=10)


def test_validate_ohlcv_for_indicator_propagates_structural_errors() -> None:
    with pytest.raises(DataError):
        validate_ohlcv_for_indicator(pd.DataFrame(), period=5)
