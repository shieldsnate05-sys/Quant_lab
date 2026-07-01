"""Tests for data.sessions."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from core.exceptions import DataError
from data.schema import SCHEMA
from data.sessions import filter_regular_session, is_trading_day, regular_session_mask


def _hourly_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(index)
    frame = pd.DataFrame(
        {
            SCHEMA.open: [100.0] * n,
            SCHEMA.high: [101.0] * n,
            SCHEMA.low: [99.0] * n,
            SCHEMA.close: [100.0] * n,
            SCHEMA.volume: [10.0] * n,
        },
        index=index,
    )
    frame.index.name = SCHEMA.timestamp
    return frame


def test_regular_session_mask_flags_hours_within_session() -> None:
    index = pd.date_range(
        "2024-01-02 00:00", periods=24, freq="1h", tz="America/New_York"
    )
    mask = regular_session_mask(index, timezone="America/New_York")
    assert index[mask].hour.tolist() == [10, 11, 12, 13, 14, 15, 16]


def test_regular_session_mask_raises_on_naive_index() -> None:
    index = pd.date_range("2024-01-02", periods=5, freq="1h")
    with pytest.raises(DataError, match="timezone-naive"):
        regular_session_mask(index)


def test_is_trading_day_true_for_weekday() -> None:
    tuesday = datetime(2024, 1, 2, 12, 0, tzinfo=UTC)
    assert is_trading_day(tuesday) is True


def test_is_trading_day_false_for_weekend() -> None:
    saturday = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)
    assert is_trading_day(saturday) is False


def test_is_trading_day_raises_on_naive_datetime() -> None:
    with pytest.raises(DataError, match="timezone-naive"):
        is_trading_day(datetime(2024, 1, 2, 12, 0))


def test_filter_regular_session_excludes_after_hours_and_weekends() -> None:
    weekday_index = pd.date_range(  # Tuesday
        "2024-01-02 00:00", periods=24, freq="1h", tz="America/New_York"
    )
    weekend_index = pd.date_range(  # Saturday, an in-session hour
        "2024-01-06 10:00", periods=1, freq="1h", tz="America/New_York"
    )
    frame = _hourly_ohlcv(weekday_index.union(weekend_index))

    filtered = filter_regular_session(frame, timezone="America/New_York")
    filtered_index = pd.DatetimeIndex(filtered.index)

    assert len(filtered) == 7
    assert filtered_index.tz_convert("America/New_York").weekday.max() < 5
