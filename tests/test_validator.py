"""Tests for data.schema and data.validator."""

from __future__ import annotations

import pandas as pd
import pytest

from core.exceptions import DataError
from data.schema import SCHEMA
from data.validator import validate_ohlcv_frame


def test_schema_columns() -> None:
    assert SCHEMA.columns == ("open", "high", "low", "close", "volume")
    assert SCHEMA.all_columns == ("timestamp", "open", "high", "low", "close", "volume")


def test_validate_ohlcv_frame_accepts_valid_frame(ohlcv_frame: pd.DataFrame) -> None:
    validate_ohlcv_frame(ohlcv_frame)  # should not raise


def test_validate_ohlcv_frame_rejects_empty_frame() -> None:
    with pytest.raises(DataError):
        validate_ohlcv_frame(pd.DataFrame())


def test_validate_ohlcv_frame_allows_empty_when_requested() -> None:
    validate_ohlcv_frame(pd.DataFrame(), allow_empty=True)  # should not raise


def test_validate_ohlcv_frame_rejects_missing_columns(
    ohlcv_frame: pd.DataFrame,
) -> None:
    broken = ohlcv_frame.drop(columns=[SCHEMA.volume])
    with pytest.raises(DataError, match="missing required columns"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_non_datetime_index(
    ohlcv_frame: pd.DataFrame,
) -> None:
    broken = ohlcv_frame.reset_index(drop=True)
    with pytest.raises(DataError, match="DatetimeIndex"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_unsorted_index(ohlcv_frame: pd.DataFrame) -> None:
    broken = ohlcv_frame.iloc[::-1]
    with pytest.raises(DataError, match="ascending order"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_null_values(ohlcv_frame: pd.DataFrame) -> None:
    broken = ohlcv_frame.copy()
    broken.loc[broken.index[0], SCHEMA.close] = None
    with pytest.raises(DataError, match="null values"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_high_below_low(ohlcv_frame: pd.DataFrame) -> None:
    broken = ohlcv_frame.copy()
    low_value = float(broken.loc[broken.index[0], SCHEMA.low])
    broken.loc[broken.index[0], SCHEMA.high] = low_value - 1.0
    with pytest.raises(DataError, match="high < low"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_negative_volume(
    ohlcv_frame: pd.DataFrame,
) -> None:
    broken = ohlcv_frame.copy()
    broken.loc[broken.index[0], SCHEMA.volume] = -1.0
    with pytest.raises(DataError, match="negative volume"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_rejects_timezone_naive_index(
    ohlcv_frame: pd.DataFrame,
) -> None:
    broken = ohlcv_frame.tz_localize(None)
    with pytest.raises(DataError, match="timezone-aware"):
        validate_ohlcv_frame(broken)


def test_validate_ohlcv_frame_allows_timezone_naive_when_not_required(
    ohlcv_frame: pd.DataFrame,
) -> None:
    naive = ohlcv_frame.tz_localize(None)
    validate_ohlcv_frame(naive, require_tz_aware=False)  # should not raise
