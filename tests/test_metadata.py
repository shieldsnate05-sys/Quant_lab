"""Tests for data.metadata."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from core.enums import TimeFrame
from core.exceptions import DataError
from data.metadata import DatasetMetadata, build_metadata, read_metadata, write_metadata


def test_build_metadata_raises_on_empty_frame() -> None:
    with pytest.raises(DataError, match="empty"):
        build_metadata(
            "QQQ",
            TimeFrame.DAILY,
            pd.DataFrame(),
            source="test",
            fetched_at=datetime.now(UTC),
        )


def test_build_metadata_captures_coverage(ohlcv_frame: pd.DataFrame) -> None:
    fetched_at = datetime.now(UTC)

    metadata = build_metadata(
        "QQQ", TimeFrame.DAILY, ohlcv_frame, source="alpaca", fetched_at=fetched_at
    )

    assert metadata.symbol == "QQQ"
    assert metadata.timeframe is TimeFrame.DAILY
    assert metadata.row_count == len(ohlcv_frame)
    assert metadata.source == "alpaca"
    assert metadata.start == ohlcv_frame.index[0].to_pydatetime()
    assert metadata.end == ohlcv_frame.index[-1].to_pydatetime()


def test_metadata_round_trips_through_json(
    tmp_path: Path, ohlcv_frame: pd.DataFrame
) -> None:
    fetched_at = datetime.now(UTC)
    metadata = build_metadata(
        "QQQ", TimeFrame.DAILY, ohlcv_frame, source="alpaca", fetched_at=fetched_at
    )
    path = tmp_path / "QQQ_1Day.meta.json"

    write_metadata(path, metadata)
    loaded = read_metadata(path)

    assert loaded == metadata


def test_read_metadata_returns_none_when_missing(tmp_path: Path) -> None:
    assert read_metadata(tmp_path / "missing.meta.json") is None


def test_read_metadata_raises_on_corrupt_file(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.meta.json"
    path.write_text("not json", encoding="utf-8")

    with pytest.raises(DataError, match="Failed to read metadata"):
        read_metadata(path)


def test_dataset_metadata_dict_round_trip() -> None:
    metadata = DatasetMetadata(
        symbol="QQQ",
        timeframe=TimeFrame.DAILY,
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 12, 31, tzinfo=UTC),
        row_count=252,
        source="alpaca",
        fetched_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    assert DatasetMetadata.from_dict(metadata.to_dict()) == metadata
