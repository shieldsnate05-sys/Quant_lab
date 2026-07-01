"""Tests for data.cache.ParquetCache and data.loader.CachedDataLoader."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from core.enums import TimeFrame
from core.types import OHLCVFrame, Symbol
from data.cache import ParquetCache
from data.loader import CachedDataLoader, DataLoader
from data.storage import ParquetStorage


def _cache(tmp_path: Path) -> ParquetCache:
    return ParquetCache(storage=ParquetStorage(base_dir=tmp_path))


def test_cache_returns_none_on_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache.read("QQQ", TimeFrame.DAILY) is None


def test_cache_round_trip(tmp_path: Path, ohlcv_frame: pd.DataFrame) -> None:
    cache = _cache(tmp_path)
    cache.write("QQQ", TimeFrame.DAILY, ohlcv_frame, source="test")

    loaded = cache.read("QQQ", TimeFrame.DAILY)

    assert loaded is not None
    # Parquet does not preserve DatetimeIndex.freq metadata, only values.
    pd.testing.assert_frame_equal(
        loaded, ohlcv_frame[list(loaded.columns)], check_freq=False
    )


def test_cache_write_records_metadata(
    tmp_path: Path, ohlcv_frame: pd.DataFrame
) -> None:
    cache = _cache(tmp_path)
    cache.write("QQQ", TimeFrame.DAILY, ohlcv_frame, source="test")

    metadata = cache.read_metadata("QQQ", TimeFrame.DAILY)

    assert metadata is not None
    assert metadata.symbol == "QQQ"
    assert metadata.timeframe is TimeFrame.DAILY
    assert metadata.row_count == len(ohlcv_frame)
    assert metadata.source == "test"


def test_cache_read_metadata_returns_none_on_miss(tmp_path: Path) -> None:
    cache = _cache(tmp_path)
    assert cache.read_metadata("QQQ", TimeFrame.DAILY) is None


class _StubLoader(DataLoader):
    """Test double that counts calls and returns a fixed frame."""

    def __init__(self, frame: OHLCVFrame) -> None:
        self.frame = frame
        self.call_count = 0

    def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> OHLCVFrame:
        self.call_count += 1
        return self.frame


def test_cached_loader_serves_second_request_from_cache(
    tmp_path: Path, ohlcv_frame: pd.DataFrame
) -> None:
    stub = _StubLoader(ohlcv_frame)
    cached_loader = CachedDataLoader(stub, cache=_cache(tmp_path))

    start = ohlcv_frame.index.min()
    end = ohlcv_frame.index.max()

    first = cached_loader.fetch_ohlcv("QQQ", TimeFrame.DAILY, start, end)
    second = cached_loader.fetch_ohlcv("QQQ", TimeFrame.DAILY, start, end)

    assert stub.call_count == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cached_loader_refetches_when_range_not_covered(
    tmp_path: Path, ohlcv_frame: pd.DataFrame
) -> None:
    stub = _StubLoader(ohlcv_frame)
    cached_loader = CachedDataLoader(stub, cache=_cache(tmp_path))

    start = ohlcv_frame.index.min()
    end = ohlcv_frame.index.max()

    cached_loader.fetch_ohlcv("QQQ", TimeFrame.DAILY, start, end)
    cached_loader.fetch_ohlcv("QQQ", TimeFrame.DAILY, start - timedelta(days=30), end)

    assert stub.call_count == 2
