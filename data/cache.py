"""
Quant-Lab Parquet Cache.

Disk-backed cache for OHLCV market data, stored as Parquet files under
``config.paths.PARQUET``. Keeping fetched bars on disk avoids redundant
API calls across research sessions.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.logging_config import get_logger
from config.paths import PARQUET
from core.enums import TimeFrame
from core.exceptions import DataError
from core.types import OHLCVFrame, Symbol
from data.schema import SCHEMA

logger = get_logger(__name__)


class ParquetCache:
    """
    Reads and writes OHLCV frames to Parquet files on disk.

    Parameters
    ----------
    cache_dir : pathlib.Path, optional
        Directory Parquet files are stored under. Defaults to
        :data:`config.paths.PARQUET`.
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir: Path = cache_dir or PARQUET
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        filename = f"{symbol.upper()}_{timeframe.value}.parquet"
        return self._cache_dir / filename

    def read(self, symbol: Symbol, timeframe: TimeFrame) -> OHLCVFrame | None:
        """
        Read cached bars for ``symbol``/``timeframe``, if present.

        Returns
        -------
        core.types.OHLCVFrame | None
            The cached frame, or ``None`` if no cache entry exists.

        Raises
        ------
        core.exceptions.DataError
            If the cache file exists but cannot be read.
        """
        path = self._key_path(symbol, timeframe)

        if not path.exists():
            return None

        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataError(f"Failed to read cache file {path}: {exc}") from exc

        logger.debug("Cache hit: %s (%d rows).", path, len(frame))

        return frame

    def write(self, symbol: Symbol, timeframe: TimeFrame, frame: OHLCVFrame) -> None:
        """
        Write ``frame`` to the cache for ``symbol``/``timeframe``.

        Raises
        ------
        core.exceptions.DataError
            If the frame cannot be written to disk.
        """
        path = self._key_path(symbol, timeframe)

        try:
            frame[list(SCHEMA.columns)].to_parquet(path)
        except Exception as exc:
            raise DataError(f"Failed to write cache file {path}: {exc}") from exc

        logger.debug("Cache write: %s (%d rows).", path, len(frame))
