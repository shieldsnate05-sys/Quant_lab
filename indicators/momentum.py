"""
Quant-Lab Momentum Indicators.

Momentum oscillators: the Relative Strength Index (RSI, using Wilder's
smoothing) and the Stochastic Oscillator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from core.constants import RSI_PERIOD, STOCH_PERIOD, STOCH_SMOOTH_D, STOCH_SMOOTH_K
from core.enums import IndicatorGroup
from core.types import FeatureFrame, OHLCVFrame, PriceSeries
from data.schema import SCHEMA
from indicators.base import Indicator, validate_ohlcv_for_indicator
from indicators.registry import REGISTRY


@REGISTRY.register
@dataclass(slots=True)
class RSI(Indicator):
    """Relative Strength Index, computed with Wilder's smoothing."""

    name: ClassVar[str] = "RSI"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.MOMENTUM

    period: int = RSI_PERIOD
    column: str = SCHEMA.close

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the Relative Strength Index of ``frame[self.column]``.

        Returns
        -------
        core.types.PriceSeries
            RSI values in ``[0, 100]``. Bars where the average loss is
            zero are set to ``100.0``. The first ``period`` values are
            ``NaN``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        prices = frame[self.column]
        delta = prices.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)

        avg_gain = gain.ewm(
            alpha=1.0 / self.period, adjust=False, min_periods=self.period
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / self.period, adjust=False, min_periods=self.period
        ).mean()

        relative_strength = avg_gain / avg_loss
        result = 100.0 - (100.0 / (1.0 + relative_strength))
        result = result.where(avg_loss != 0.0, 100.0)

        return result.rename(self.name)


@REGISTRY.register
@dataclass(slots=True)
class Stochastic(Indicator):
    """Stochastic Oscillator (%K, %D)."""

    name: ClassVar[str] = "STOCH"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.MOMENTUM

    period: int = STOCH_PERIOD
    smooth_k: int = STOCH_SMOOTH_K
    smooth_d: int = STOCH_SMOOTH_D

    def compute(self, frame: OHLCVFrame) -> FeatureFrame:
        """
        Compute the stochastic oscillator's %K and %D lines.

        Returns
        -------
        core.types.FeatureFrame
            Columns ``percent_k`` and ``percent_d``.
        """
        validate_ohlcv_for_indicator(frame, self.period)

        high, low, close = frame[SCHEMA.high], frame[SCHEMA.low], frame[SCHEMA.close]

        lowest_low = low.rolling(window=self.period, min_periods=self.period).min()
        highest_high = high.rolling(window=self.period, min_periods=self.period).max()

        raw_k = 100.0 * (close - lowest_low) / (highest_high - lowest_low)
        percent_k = raw_k.rolling(
            window=self.smooth_k, min_periods=self.smooth_k
        ).mean()
        percent_d = percent_k.rolling(
            window=self.smooth_d, min_periods=self.smooth_d
        ).mean()

        return pd.DataFrame(
            {"percent_k": percent_k, "percent_d": percent_d}, index=frame.index
        )
