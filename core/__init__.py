"""
Quant-Lab Core Package.

Cross-cutting primitives shared by every other package in the platform:
custom exceptions, type aliases, enumerations, and domain constants.

No module outside :mod:`core` may redefine these primitives. Every other
package (``data``, ``indicators``, ``features``, ``strategies``,
``backtesting``, ``optimization``, ``ml``, ``reports``, ``visualization``)
depends on ``core`` and never the other way around.
"""

from __future__ import annotations

from core.constants import (
    ADX_PERIOD,
    ATR_PERIOD,
    BBANDS_PERIOD,
    BBANDS_STD,
    EMA_FAST,
    EMA_SLOW,
    FLOAT_TOLERANCE,
    ICHIMOKU_KIJUN,
    ICHIMOKU_SPAN_B,
    ICHIMOKU_TENKAN,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    MARKET_CLOSE,
    MARKET_OPEN,
    RSI_PERIOD,
    STOCH_PERIOD,
)
from core.enums import (
    IndicatorGroup,
    ModelType,
    OrderType,
    PositionSide,
    TimeFrame,
)
from core.exceptions import (
    BacktestError,
    ConfigurationError,
    DataError,
    FeatureError,
    IndicatorError,
    ModelError,
    OptimizationError,
    QuantLabError,
    ReportError,
    StrategyError,
    ValidationError,
)
from core.types import (
    FeatureFrame,
    OHLCVFrame,
    PredictionArray,
    PriceSeries,
    Symbol,
    VolumeSeries,
)

__all__ = [
    "ADX_PERIOD",
    "ATR_PERIOD",
    "BBANDS_PERIOD",
    "BBANDS_STD",
    "EMA_FAST",
    "EMA_SLOW",
    "FLOAT_TOLERANCE",
    "ICHIMOKU_KIJUN",
    "ICHIMOKU_SPAN_B",
    "ICHIMOKU_TENKAN",
    "MACD_FAST",
    "MACD_SIGNAL",
    "MACD_SLOW",
    "MARKET_CLOSE",
    "MARKET_OPEN",
    "RSI_PERIOD",
    "STOCH_PERIOD",
    "BacktestError",
    "ConfigurationError",
    "DataError",
    "FeatureError",
    "FeatureFrame",
    "IndicatorError",
    "IndicatorGroup",
    "ModelError",
    "ModelType",
    "OHLCVFrame",
    "OptimizationError",
    "OrderType",
    "PositionSide",
    "PredictionArray",
    "PriceSeries",
    "QuantLabError",
    "ReportError",
    "StrategyError",
    "Symbol",
    "TimeFrame",
    "ValidationError",
    "VolumeSeries",
]
