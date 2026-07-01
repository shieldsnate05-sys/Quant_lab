"""Tests for data.resampler."""

from __future__ import annotations

import pandas as pd
import pytest

from core.enums import TimeFrame
from core.exceptions import DataError
from data.resampler import resample_ohlcv
from data.schema import SCHEMA


def _minute_frame(n: int = 10) -> pd.DataFrame:
    index = pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            SCHEMA.open: [100.0 + i for i in range(n)],
            SCHEMA.high: [101.0 + i for i in range(n)],
            SCHEMA.low: [99.0 + i for i in range(n)],
            SCHEMA.close: [100.5 + i for i in range(n)],
            SCHEMA.volume: [10.0] * n,
        },
        index=index,
    )
    frame.index.name = SCHEMA.timestamp
    return frame


def test_resample_aggregates_ohlcv_correctly() -> None:
    frame = _minute_frame(10)

    resampled = resample_ohlcv(frame, TimeFrame.ONE_MINUTE, TimeFrame.FIVE_MINUTE)

    assert len(resampled) == 2

    first, second = resampled.iloc[0], resampled.iloc[1]

    assert first[SCHEMA.open] == pytest.approx(100.0)
    assert first[SCHEMA.high] == pytest.approx(105.0)
    assert first[SCHEMA.low] == pytest.approx(99.0)
    assert first[SCHEMA.close] == pytest.approx(104.5)
    assert first[SCHEMA.volume] == pytest.approx(50.0)

    assert second[SCHEMA.open] == pytest.approx(105.0)
    assert second[SCHEMA.high] == pytest.approx(110.0)
    assert second[SCHEMA.low] == pytest.approx(104.0)
    assert second[SCHEMA.close] == pytest.approx(109.5)
    assert second[SCHEMA.volume] == pytest.approx(50.0)


def test_resample_rejects_non_coarser_target() -> None:
    frame = _minute_frame(10)
    with pytest.raises(DataError, match="must be coarser"):
        resample_ohlcv(frame, TimeFrame.FIVE_MINUTE, TimeFrame.ONE_MINUTE)


def test_resample_rejects_equal_timeframe() -> None:
    frame = _minute_frame(10)
    with pytest.raises(DataError, match="must be coarser"):
        resample_ohlcv(frame, TimeFrame.ONE_MINUTE, TimeFrame.ONE_MINUTE)


def test_resample_output_passes_validation() -> None:
    frame = _minute_frame(30)
    resampled = resample_ohlcv(frame, TimeFrame.ONE_MINUTE, TimeFrame.FIFTEEN_MINUTE)
    assert isinstance(resampled.index, pd.DatetimeIndex)
    assert resampled.index.name == SCHEMA.timestamp
