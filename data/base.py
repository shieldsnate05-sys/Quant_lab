"""
Quant-Lab Data Loader Interface.

Defines the abstract contract every market data source must implement,
so that strategies, backtests, and feature pipelines can depend on
:class:`DataLoader` rather than a concrete vendor implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from core.enums import TimeFrame
from core.types import OHLCVFrame, Symbol


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
