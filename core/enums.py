"""
Quant-Lab Core Enumerations.

Domain enumerations shared across the platform: timeframes, position
sides, order types, indicator groupings, and model types.
"""

from __future__ import annotations

from enum import Enum


class TimeFrame(Enum):
    """Bar timeframe used for market data and backtesting."""

    ONE_MINUTE = "1Min"
    FIVE_MINUTE = "5Min"
    FIFTEEN_MINUTE = "15Min"
    THIRTY_MINUTE = "30Min"
    ONE_HOUR = "1Hour"
    DAILY = "1Day"


class PositionSide(Enum):
    """Direction of an open position."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class OrderType(Enum):
    """Order execution type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class IndicatorGroup(Enum):
    """Category an indicator belongs to."""

    TREND = "Trend"
    MOMENTUM = "Momentum"
    VOLATILITY = "Volatility"
    VOLUME = "Volume"
    MARKET_STRUCTURE = "Market Structure"


class ModelType(Enum):
    """Supported machine-learning model families."""

    RANDOM_FOREST = "RandomForest"
    XGBOOST = "XGBoost"
