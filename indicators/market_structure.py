"""
Quant-Lab Market Structure Indicators.

Support/resistance and trend-cloud indicators: the Ichimoku Kinko Hyo
system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import pandas as pd

from core.constants import ICHIMOKU_KIJUN, ICHIMOKU_SPAN_B, ICHIMOKU_TENKAN
from core.enums import IndicatorGroup
from core.types import FeatureFrame, OHLCVFrame
from data.schema import SCHEMA
from indicators.base import Indicator, validate_ohlcv_for_indicator
from indicators.registry import REGISTRY


def _midpoint_channel(high: pd.Series, low: pd.Series, period: int) -> pd.Series:
    """Return the midpoint of the highest high and lowest low over ``period`` bars."""
    return (
        high.rolling(window=period, min_periods=period).max()
        + low.rolling(window=period, min_periods=period).min()
    ) / 2.0


@REGISTRY.register
@dataclass(slots=True)
class Ichimoku(Indicator):
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud).

    ``senkou_span_a`` and ``senkou_span_b`` are projected ``kijun_period``
    bars forward, and ``chikou_span`` is projected ``kijun_period`` bars
    backward, per the standard Ichimoku construction - so the leading
    and trailing edges of the returned frame are ``NaN`` by design, not
    a validation gap.
    """

    name: ClassVar[str] = "ICHIMOKU"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.MARKET_STRUCTURE

    tenkan_period: int = ICHIMOKU_TENKAN
    kijun_period: int = ICHIMOKU_KIJUN
    senkou_b_period: int = ICHIMOKU_SPAN_B

    def compute(self, frame: OHLCVFrame) -> FeatureFrame:
        """
        Compute the Ichimoku Cloud components.

        Returns
        -------
        core.types.FeatureFrame
            Columns ``tenkan_sen``, ``kijun_sen``, ``senkou_span_a``,
            ``senkou_span_b``, and ``chikou_span``.
        """
        validate_ohlcv_for_indicator(frame, self.senkou_b_period)

        high, low, close = frame[SCHEMA.high], frame[SCHEMA.low], frame[SCHEMA.close]

        tenkan_sen = _midpoint_channel(high, low, self.tenkan_period)
        kijun_sen = _midpoint_channel(high, low, self.kijun_period)

        senkou_span_a = ((tenkan_sen + kijun_sen) / 2.0).shift(self.kijun_period)
        senkou_span_b = _midpoint_channel(high, low, self.senkou_b_period).shift(
            self.kijun_period
        )
        chikou_span = close.shift(-self.kijun_period)

        return pd.DataFrame(
            {
                "tenkan_sen": tenkan_sen,
                "kijun_sen": kijun_sen,
                "senkou_span_a": senkou_span_a,
                "senkou_span_b": senkou_span_b,
                "chikou_span": chikou_span,
            },
            index=frame.index,
        )
