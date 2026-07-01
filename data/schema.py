"""
Quant-Lab OHLCV Schema.

Defines the canonical column layout for OHLCV (open, high, low, close,
volume) market data used throughout the platform.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OHLCVSchema:
    """
    Canonical column names for an OHLCV market data frame.

    The frame is expected to be indexed by a timezone-aware
    ``DatetimeIndex`` named :attr:`timestamp`; :attr:`columns` lists the
    five price/volume columns that must be present alongside it.
    """

    timestamp: str = "timestamp"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str = "volume"

    @property
    def columns(self) -> tuple[str, ...]:
        """Return the ordered OHLCV column names, excluding the timestamp."""
        return (self.open, self.high, self.low, self.close, self.volume)

    @property
    def all_columns(self) -> tuple[str, ...]:
        """Return every column name, including the timestamp index name."""
        return (self.timestamp, *self.columns)


#: The default OHLCV schema used across Quant-Lab.
SCHEMA = OHLCVSchema()
