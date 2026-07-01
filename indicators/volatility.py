"""
Quant-Lab Volatility Indicators.

Volatility and trend-strength indicators: Average True Range (ATR),
Average Directional Index (ADX), and Bollinger Bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import pandas as pd

from core.constants import ADX_PERIOD, ATR_PERIOD, BBANDS_PERIOD, BBANDS_STD
from core.enums import IndicatorGroup
from core.types import FeatureFrame, OHLCVFrame, PriceSeries
from data.schema import SCHEMA
from indicators.base import Indicator, validate_ohlcv_for_indicator
from indicators.registry import REGISTRY
from indicators.trend import SMA


def _true_range(frame: OHLCVFrame) -> pd.Series:
    """Compute the true range (max of three candidate ranges) for each bar."""
    high, low, close = frame[SCHEMA.high], frame[SCHEMA.low], frame[SCHEMA.close]
    prev_close = close.shift(1)

    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Apply Wilder's smoothing (an EWM with alpha = 1 / period)."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


@REGISTRY.register
@dataclass(slots=True)
class ATR(Indicator):
    """Average True Range, computed with Wilder's smoothing."""

    name: ClassVar[str] = "ATR"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.VOLATILITY

    period: int = ATR_PERIOD

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the Average True Range.

        Returns
        -------
        core.types.PriceSeries
            The Wilder-smoothed true range, with the first ``period``
            values equal to ``NaN``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        true_range = _true_range(frame)
        result = _wilder_smooth(true_range, self.period)
        return result.rename(self.name)


@REGISTRY.register
@dataclass(slots=True)
class ADX(Indicator):
    """Average Directional Index, with +DI and -DI."""

    name: ClassVar[str] = "ADX"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.VOLATILITY

    period: int = ADX_PERIOD

    def compute(self, frame: OHLCVFrame) -> FeatureFrame:
        """
        Compute the ADX along with the +DI and -DI directional indicators.

        Returns
        -------
        core.types.FeatureFrame
            Columns ``adx``, ``plus_di``, and ``minus_di``.
        """
        validate_ohlcv_for_indicator(frame, self.period * 2)

        high, low = frame[SCHEMA.high], frame[SCHEMA.low]

        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
            index=frame.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
            index=frame.index,
        )

        smoothed_tr = _wilder_smooth(_true_range(frame), self.period)
        smoothed_plus_dm = _wilder_smooth(plus_dm, self.period)
        smoothed_minus_dm = _wilder_smooth(minus_dm, self.period)

        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr

        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = _wilder_smooth(dx, self.period)

        return pd.DataFrame(
            {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}, index=frame.index
        )


@REGISTRY.register
@dataclass(slots=True)
class BollingerBands(Indicator):
    """Bollinger Bands: an SMA with upper/lower bands at N standard deviations."""

    name: ClassVar[str] = "BBANDS"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.VOLATILITY

    period: int = BBANDS_PERIOD
    num_std: float = BBANDS_STD
    column: str = SCHEMA.close

    def compute(self, frame: OHLCVFrame) -> FeatureFrame:
        """
        Compute the Bollinger Bands.

        Returns
        -------
        core.types.FeatureFrame
            Columns ``upper``, ``middle``, and ``lower``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        series = frame[self.column]
        middle = SMA(period=self.period, column=self.column).compute(frame)
        std = series.rolling(window=self.period, min_periods=self.period).std(ddof=0)

        upper = middle + self.num_std * std
        lower = middle - self.num_std * std

        return pd.DataFrame(
            {"upper": upper, "middle": middle, "lower": lower}, index=frame.index
        )
