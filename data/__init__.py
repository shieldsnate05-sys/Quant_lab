"""
Quant-Lab Data Package.

Market data acquisition, caching, validation, resampling, and session
filtering. All OHLCV data flowing through the platform conforms to
:data:`data.schema.SCHEMA`.
"""

from __future__ import annotations

from data.cache import ParquetCache
from data.downloader import AlpacaDownloader
from data.loader import CachedDataLoader, DataLoader
from data.metadata import DatasetMetadata, build_metadata, read_metadata, write_metadata
from data.resampler import resample_ohlcv
from data.schema import SCHEMA, OHLCVSchema
from data.sessions import filter_regular_session, is_trading_day, regular_session_mask
from data.storage import ParquetStorage
from data.validator import validate_ohlcv_frame

__all__ = [
    "SCHEMA",
    "AlpacaDownloader",
    "CachedDataLoader",
    "DataLoader",
    "DatasetMetadata",
    "OHLCVSchema",
    "ParquetCache",
    "ParquetStorage",
    "build_metadata",
    "filter_regular_session",
    "is_trading_day",
    "read_metadata",
    "regular_session_mask",
    "resample_ohlcv",
    "validate_ohlcv_frame",
    "write_metadata",
]
