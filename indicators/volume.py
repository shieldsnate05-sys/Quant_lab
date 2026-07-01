"""
Quant-Lab Volume Indicators.

Volume-weighted price indicators: the Volume-Weighted Average Price (VWAP).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from core.enums import IndicatorGroup
from core.exceptions import IndicatorError
from core.types import OHLCVFrame, PriceSeries
from data.schema import SCHEMA
from indicators.base import Indicator, validate_ohlcv_for_indicator
from indicators.registry import REGISTRY


@REGISTRY.register
@dataclass(slots=True)
class VWAP(Indicator):
    """
    Volume-Weighted Average Price.

    With ``window=None`` (the default), this is an *anchored* VWAP:
    cumulative from the first bar in ``frame``. With ``window`` set, it
    is a rolling VWAP over the trailing ``window`` bars instead.
    Neither variant resets at session boundaries; combine with
    :func:`data.sessions.filter_regular_session` or pre-split ``frame``
    by session if a session-anchored VWAP is required.
    """

    name: ClassVar[str] = "VWAP"
    group: ClassVar[IndicatorGroup] = IndicatorGroup.VOLUME

    window: int | None = None

    def compute(self, frame: OHLCVFrame) -> PriceSeries:
        """
        Compute the Volume-Weighted Average Price.

        Returns
        -------
        core.types.PriceSeries
            The VWAP at each bar.
        """
        if self.window is not None and self.window <= 0:
            raise IndicatorError(f"window must be positive, got {self.window}.")

        validate_ohlcv_for_indicator(frame, self.window)

        typical_price = (
            frame[SCHEMA.high] + frame[SCHEMA.low] + frame[SCHEMA.close]
        ) / 3.0
        price_volume = typical_price * frame[SCHEMA.volume]
        volume = frame[SCHEMA.volume]

        if self.window is None:
            cumulative_price_volume = price_volume.cumsum()
            cumulative_volume = volume.cumsum()
        else:
            cumulative_price_volume = price_volume.rolling(
                window=self.window, min_periods=self.window
            ).sum()
            cumulative_volume = volume.rolling(
                window=self.window, min_periods=self.window
            ).sum()

        result = cumulative_price_volume / cumulative_volume
        return result.rename(self.name)
