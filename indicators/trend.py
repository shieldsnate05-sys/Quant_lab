"""
Quant-Lab Trend Indicators.

Trend-following indicators: simple and exponential moving averages,
MACD, and Parabolic SAR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from core.constants import (
    EMA_FAST,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PSAR_AF_INCREMENT,
    PSAR_AF_MAX,
    PSAR_AF_START,
)
from core.enums import IndicatorGroup
from core.exceptions import IndicatorError
from core.types import FeatureFrame, OHLCVFrame, PriceSeries
from data.schema import SCHEMA
from indicators.base import Indicator, validate_ohlcv_for_indicator
from indicators.registry import REGISTRY


@REGISTRY.register
@dataclass(slots=True)
class SMA(Indicator):
    """Simple moving average."""

    name: ClassVar[str] = "SMA"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.TREND

    period: int
    column: str = SCHEMA.close

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the simple moving average of ``frame[self.column]``.

        Returns
        -------
        core.types.PriceSeries
            The rolling mean, with the first ``period - 1`` values
            equal to ``NaN``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        series = frame[self.column]
        result = series.rolling(window=self.period, min_periods=self.period).mean()
        return result.rename(self.name)


@REGISTRY.register
@dataclass(slots=True)
class EMA(Indicator):
    """Exponential moving average."""

    name: ClassVar[str] = "EMA"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.TREND

    period: int = EMA_FAST
    column: str = SCHEMA.close

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the exponential moving average of ``frame[self.column]``.

        Returns
        -------
        core.types.PriceSeries
            The exponentially weighted moving average, with the first
            ``period - 1`` values equal to ``NaN``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        series = frame[self.column]
        result = series.ewm(
            span=self.period, adjust=False, min_periods=self.period
        ).mean()
        return result.rename(self.name)


@REGISTRY.register
@dataclass(slots=True)
class MACD(Indicator):
    """Moving Average Convergence Divergence."""

    name: ClassVar[str] = "MACD"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.TREND

    fast_period: int = MACD_FAST
    slow_period: int = MACD_SLOW
    signal_period: int = MACD_SIGNAL
    column: str = SCHEMA.close

    def __post_init__(self) -> None:
        if self.fast_period >= self.slow_period:
            raise IndicatorError(
                f"fast_period ({self.fast_period}) must be less than "
                f"slow_period ({self.slow_period})."
            )

    def compute(self, frame: OHLCVFrame) -> FeatureFrame:
        """
        Compute the MACD line, signal line, and histogram.

        Returns
        -------
        core.types.FeatureFrame
            Columns ``macd``, ``signal``, and ``histogram``.
        """
        validate_ohlcv_for_indicator(frame, self.slow_period)

        fast_ema = EMA(period=self.fast_period, column=self.column).compute(frame)
        slow_ema = EMA(period=self.slow_period, column=self.column).compute(frame)

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(
            span=self.signal_period, adjust=False, min_periods=self.signal_period
        ).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame(
            {"macd": macd_line, "signal": signal_line, "histogram": histogram},
            index=frame.index,
        )


@REGISTRY.register
@dataclass(slots=True)
class ParabolicSAR(Indicator):
    """
    Parabolic Stop-and-Reverse (Wilder).

    Unlike the platform's other indicators, Parabolic SAR is a
    recursive, stateful algorithm: each bar's value depends on the
    previous bar's SAR, trend direction, and extreme point, so it
    cannot be expressed as an elementwise or rolling-window vectorized
    pandas operation. This implementation runs a single sequential
    pass over numpy arrays (the standard approach used by reference
    implementations), which is still substantially faster than
    row-by-row pandas indexing.
    """

    name: ClassVar[str] = "PSAR"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.TREND

    af_start: float = PSAR_AF_START
    af_increment: float = PSAR_AF_INCREMENT
    af_max: float = PSAR_AF_MAX

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the Parabolic SAR stop-and-reverse series.

        Returns
        -------
        core.types.PriceSeries
            The SAR value at each bar.
        """
        validate_ohlcv_for_indicator(frame, period=2)

        high = frame[SCHEMA.high].to_numpy(dtype=np.float64)
        low = frame[SCHEMA.low].to_numpy(dtype=np.float64)
        n = len(frame)

        sar = np.empty(n, dtype=np.float64)
        trend_up = np.empty(n, dtype=np.bool_)

        trend_up[0] = True
        sar[0] = low[0]
        extreme_point = high[0]
        af = self.af_start

        for i in range(1, n):
            prior_sar = sar[i - 1]
            prior_low = low[i - 2] if i >= 2 else low[i - 1]
            prior_high = high[i - 2] if i >= 2 else high[i - 1]

            if trend_up[i - 1]:
                candidate = prior_sar + af * (extreme_point - prior_sar)
                candidate = min(candidate, low[i - 1], prior_low)

                if high[i] > extreme_point:
                    extreme_point = high[i]
                    af = min(af + self.af_increment, self.af_max)

                if low[i] < candidate:
                    trend_up[i] = False
                    sar[i] = extreme_point
                    extreme_point = low[i]
                    af = self.af_start
                else:
                    trend_up[i] = True
                    sar[i] = candidate
            else:
                candidate = prior_sar - af * (prior_sar - extreme_point)
                candidate = max(candidate, high[i - 1], prior_high)

                if low[i] < extreme_point:
                    extreme_point = low[i]
                    af = min(af + self.af_increment, self.af_max)

                if high[i] > candidate:
                    trend_up[i] = True
                    sar[i] = extreme_point
                    extreme_point = high[i]
                    af = self.af_start
                else:
                    trend_up[i] = False
                    sar[i] = candidate

        return pd.Series(sar, index=frame.index, name=self.name)
