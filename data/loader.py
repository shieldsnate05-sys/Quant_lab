"""
Quant-Lab Data Loader Interface and Caching Decorator.

Defines the abstract contract every market data source must implement
(:class:`DataLoader`), so that strategies, backtests, and feature
pipelines can depend on it rather than a concrete vendor
implementation, plus :class:`CachedDataLoader`, which decorates any
:class:`DataLoader` with :class:`~data.cache.ParquetCache` caching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings
from core.enums import TimeFrame
from core.types import OHLCVFrame, Symbol
from data.cache import ParquetCache
from data.validator import validate_ohlcv_frame

logger = get_logger(__name__)


class DataLoader(ABC):
    """Abstract base class for OHLCV market data sources."""

    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> OHLCVFrame:
        """
        Fetch OHLCV bars for ``symbol`` between ``start`` and ``end``.

        Parameters
        ----------
        symbol : core.types.Symbol
            Ticker symbol to fetch.
        timeframe : core.enums.TimeFrame
            Bar timeframe to fetch.
        start : datetime.datetime
            Inclusive start of the date range.
        end : datetime.datetime
            Inclusive end of the date range.

        Returns
        -------
        core.types.OHLCVFrame
            OHLCV bars indexed by a timezone-aware ``DatetimeIndex`` and
            conforming to :data:`data.schema.SCHEMA`.

        Raises
        ------
        core.exceptions.DataError
            If the data cannot be retrieved.
        """
        raise NotImplementedError


class CachedDataLoader(DataLoader):
    """
    Decorates a :class:`DataLoader` with Parquet caching.

    Parameters
    ----------
    loader : DataLoader
        The underlying data source to fetch from on a cache miss.
    cache : data.cache.ParquetCache, optional
        The cache to read from / write to. A new cache using the
        default location is created if not provided.
    """

    def __init__(self, loader: DataLoader, cache: ParquetCache | None = None) -> None:
        self._loader = loader
        self._cache = cache or ParquetCache()

    def fetch_ohlcv(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        start: datetime,
        end: datetime,
    ) -> OHLCVFrame:
        """
        Fetch OHLCV bars for ``symbol``, preferring the on-disk cache.

        Parameters
        ----------
        symbol : core.types.Symbol
            Ticker symbol to fetch.
        timeframe : core.enums.TimeFrame
            Bar timeframe to fetch.
        start : datetime.datetime
            Inclusive start of the date range.
        end : datetime.datetime
            Inclusive end of the date range.

        Returns
        -------
        core.types.OHLCVFrame
            OHLCV bars covering at least ``[start, end]``.

        Raises
        ------
        core.exceptions.DataError
            If neither the cache nor the underlying loader can supply
            the requested data.
        """
        if settings.data.cache_data:
            cached = self._cache.read(symbol, timeframe)
            if cached is not None and not cached.empty:
                covers_range = cached.index.min() <= start and cached.index.max() >= end
                if covers_range:
                    logger.info("Serving %s (%s) from cache.", symbol, timeframe.value)
                    return cached.loc[start:end]

        frame = self._loader.fetch_ohlcv(symbol, timeframe, start, end)
        validate_ohlcv_frame(frame)

        if settings.data.cache_data:
            self._cache.write(
                symbol, timeframe, frame, source=type(self._loader).__name__
            )

        return frame
