"""
Quant-Lab Data Package.

Market data acquisition, caching, and validation. All OHLCV data
flowing through the platform conforms to :data:`data.schema.SCHEMA`.
"""

from __future__ import annotations

from data.alpaca_loader import AlpacaDataLoader
from data.base import DataLoader
from data.cache import ParquetCache
from data.cached_loader import CachedDataLoader
from data.schema import SCHEMA, OHLCVSchema
from data.validation import validate_ohlcv_frame

__all__ = [
    "SCHEMA",
    "AlpacaDataLoader",
    "CachedDataLoader",
    "DataLoader",
    "OHLCVSchema",
    "ParquetCache",
    "validate_ohlcv_frame",
]
