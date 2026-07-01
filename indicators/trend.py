"""
Quant-Lab Trend Indicators.

Vectorized trend-following indicators: simple and exponential moving
averages.
"""

from __future__ import annotations

from core.constants import EMA_FAST
from core.types import PriceSeries
from indicators.base import validate_price_series


def sma(prices: PriceSeries, period: int) -> PriceSeries:
    """
    Compute the simple moving average of ``prices``.

    Parameters
    ----------
    prices : core.types.PriceSeries
        Input price series (e.g. close prices).
    period : int
        Number of bars in the rolling window.

    Returns
    -------
    core.types.PriceSeries
        The rolling mean of ``prices``, with the first ``period - 1``
        values equal to ``NaN``.

    Raises
    ------
    core.exceptions.IndicatorError
        If ``period`` is not positive or ``prices`` is too short.
    """
    validate_price_series(prices, period)

    return prices.rolling(window=period, min_periods=period).mean()


def ema(prices: PriceSeries, period: int = EMA_FAST) -> PriceSeries:
    """
    Compute the exponential moving average of ``prices``.

    Parameters
    ----------
    prices : core.types.PriceSeries
        Input price series (e.g. close prices).
    period : int, optional
        The EMA span. Defaults to :data:`core.constants.EMA_FAST`.

    Returns
    -------
    core.types.PriceSeries
        The exponentially weighted moving average of ``prices``, with
        the first ``period - 1`` values equal to ``NaN``.

    Raises
    ------
    core.exceptions.IndicatorError
        If ``period`` is not positive or ``prices`` is too short.
    """
    validate_price_series(prices, period)

    return prices.ewm(span=period, adjust=False, min_periods=period).mean()
