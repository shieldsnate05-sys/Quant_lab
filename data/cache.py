"""
Quant-Lab Parquet Cache.

Key-based (symbol, timeframe) cache for OHLCV market data. Delegates
the actual file I/O to :class:`~data.storage.ParquetStorage` and
records a :class:`~data.metadata.DatasetMetadata` sidecar alongside
every cached dataset.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from config.logging_config import get_logger
from config.paths import PARQUET
from core.enums import TimeFrame
from core.types import OHLCVFrame, Symbol
from data.metadata import DatasetMetadata, build_metadata, read_metadata, write_metadata
from data.schema import SCHEMA
from data.storage import ParquetStorage

logger = get_logger(__name__)


class ParquetCache:
    """
    Caches OHLCV frames to disk, keyed by symbol and timeframe.

    Parameters
    ----------
    storage : data.storage.ParquetStorage, optional
        Storage backend to read/write through. Defaults to a
        :class:`~data.storage.ParquetStorage` rooted at
        :data:`config.paths.PARQUET`.
    """

    def __init__(self, storage: ParquetStorage | None = None) -> None:
        self._storage = storage or ParquetStorage(base_dir=PARQUET)

    def _data_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        return self._storage.resolve_path(f"{symbol.upper()}_{timeframe.value}.parquet")

    def _metadata_path(self, symbol: Symbol, timeframe: TimeFrame) -> Path:
        return self._storage.resolve_path(
            f"{symbol.upper()}_{timeframe.value}.meta.json"
        )

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
        path = self._data_path(symbol, timeframe)

        if not self._storage.exists(path):
            return None

        frame = self._storage.read(path)
        logger.info("Cache hit: %s/%s (%d rows).", symbol, timeframe.value, len(frame))

        return frame

    def read_metadata(
        self, symbol: Symbol, timeframe: TimeFrame
    ) -> DatasetMetadata | None:
        """Read the metadata sidecar for ``symbol``/``timeframe``, if present."""
        return read_metadata(self._metadata_path(symbol, timeframe))

    def write(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        frame: OHLCVFrame,
        *,
        source: str = "unknown",
    ) -> None:
        """
        Write ``frame`` to the cache for ``symbol``/``timeframe``.

        Parameters
        ----------
        symbol : core.types.Symbol
            Ticker symbol the frame covers.
        timeframe : core.enums.TimeFrame
            Bar timeframe of the frame.
        frame : core.types.OHLCVFrame
            The OHLCV frame to cache.
        source : str, optional
            Identifier for the data source, recorded in the metadata
            sidecar (e.g. ``"alpaca"``). Defaults to ``"unknown"``.

        Raises
        ------
        core.exceptions.DataError
            If the frame or its metadata cannot be written to disk.
        """
        self._storage.write(
            self._data_path(symbol, timeframe), frame[list(SCHEMA.columns)]
        )

        metadata = build_metadata(
            symbol,
            timeframe,
            frame,
            source=source,
            fetched_at=datetime.now(UTC),
        )
        write_metadata(self._metadata_path(symbol, timeframe), metadata)

        logger.info(
            "Cache write: %s/%s (%d rows).", symbol, timeframe.value, len(frame)
        )
