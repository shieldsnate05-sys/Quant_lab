"""Tests for data.storage.ParquetStorage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.exceptions import DataError
from data.storage import ParquetStorage


def test_base_dir_is_created(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "storage"
    ParquetStorage(base_dir=target)
    assert target.is_dir()


def test_resolve_path_is_relative_to_base_dir(tmp_path: Path) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    assert storage.resolve_path("foo.parquet") == tmp_path / "foo.parquet"


def test_exists_false_for_missing_file(tmp_path: Path) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    assert storage.exists(storage.resolve_path("missing.parquet")) is False


def test_read_raises_on_missing_file(tmp_path: Path) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    with pytest.raises(DataError, match="does not exist"):
        storage.read(storage.resolve_path("missing.parquet"))


def test_write_then_read_round_trip(tmp_path: Path, ohlcv_frame: pd.DataFrame) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    path = storage.resolve_path("qqq.parquet")

    storage.write(path, ohlcv_frame)

    assert storage.exists(path)
    loaded = storage.read(path)
    pd.testing.assert_frame_equal(loaded, ohlcv_frame, check_freq=False)


def test_delete_removes_file(tmp_path: Path, ohlcv_frame: pd.DataFrame) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    path = storage.resolve_path("qqq.parquet")
    storage.write(path, ohlcv_frame)

    storage.delete(path)

    assert not storage.exists(path)


def test_delete_is_a_no_op_when_file_missing(tmp_path: Path) -> None:
    storage = ParquetStorage(base_dir=tmp_path)
    storage.delete(storage.resolve_path("missing.parquet"))  # should not raise
