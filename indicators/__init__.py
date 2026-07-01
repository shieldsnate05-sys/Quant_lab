"""
Quant-Lab Indicators Package.

Vectorized technical indicators, grouped by :class:`core.enums.IndicatorGroup`.
Every concrete indicator is a dataclass implementing
:class:`~indicators.base.Indicator` and self-registers with
:data:`~indicators.registry.REGISTRY` on import, so it can also be
looked up and instantiated by name via ``REGISTRY.create("EMA", ...)``.
"""

from __future__ import annotations

from indicators.base import Indicator
from indicators.market_structure import Ichimoku
from indicators.momentum import RSI, Stochastic
from indicators.registry import REGISTRY, IndicatorRegistry
from indicators.trend import EMA, MACD, SMA, ParabolicSAR
from indicators.volatility import ADX, ATR, BollingerBands
from indicators.volume import VWAP

__all__ = [
    "ADX",
    "ATR",
    "EMA",
    "MACD",
    "REGISTRY",
    "RSI",
    "SMA",
    "VWAP",
    "BollingerBands",
    "Ichimoku",
    "Indicator",
    "IndicatorRegistry",
    "ParabolicSAR",
    "Stochastic",
]
