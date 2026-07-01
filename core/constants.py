"""
Quant-Lab Core Constants.

Domain constants shared across the platform: the default symbol,
regular trading hours, and default indicator periods.
"""

from __future__ import annotations

from datetime import time

#: Default ticker symbol used when none is specified.
DEFAULT_SYMBOL = "QQQ"

#: US regular market session (America/New_York), used as the default
#: trading-hours window unless overridden by :mod:`config.settings`.
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)

# EMA
EMA_FAST = 9
EMA_SLOW = 21

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# RSI
RSI_PERIOD = 14

# ATR
ATR_PERIOD = 14

# ADX
ADX_PERIOD = 14

# Stochastic Oscillator
STOCH_PERIOD = 14
STOCH_SMOOTH_K = 3
STOCH_SMOOTH_D = 3

# Bollinger Bands
BBANDS_PERIOD = 20
BBANDS_STD = 2

# Ichimoku Cloud
ICHIMOKU_TENKAN = 9
ICHIMOKU_KIJUN = 26
ICHIMOKU_SPAN_B = 52

# Parabolic SAR
PSAR_AF_START = 0.02
PSAR_AF_INCREMENT = 0.02
PSAR_AF_MAX = 0.2

#: Tolerance used when comparing floating point values for equality.
FLOAT_TOLERANCE = 1e-9
