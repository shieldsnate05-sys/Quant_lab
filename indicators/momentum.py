"""
Quant-Lab Momentum Indicators.

Vectorized momentum indicators: the Relative Strength Index (RSI),
computed with Wilder's smoothing method.
"""

from __future__ import annotations

from core.constants import RSI_PERIOD
from core.types import PriceSeries
from indicators.base import validate_price_series


def rsi(prices: PriceSeries, period: int = RSI_PERIOD) -> PriceSeries:
    """
    Compute the Relative Strength Index (RSI) of ``prices``.

    Uses Wilder's smoothing (an exponential moving average with
    ``alpha = 1 / period``), the standard RSI formulation.

    Parameters
    ----------
    prices : core.types.PriceSeries
        Input price series (e.g. close prices).
    period : int, optional
        The lookback period. Defaults to :data:`core.constants.RSI_PERIOD`.

    Returns
    -------
    core.types.PriceSeries
        RSI values in ``[0, 100]``. Bars where the average loss is zero
        are set to ``100.0``. The first ``period`` values are ``NaN``.

    Raises
    ------
    core.exceptions.IndicatorError
        If ``period`` is not positive or ``prices`` is too short.
    """
    validate_price_series(prices, period)

    delta = prices.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    relative_strength = avg_gain / avg_loss
    rsi_values = 100.0 - (100.0 / (1.0 + relative_strength))

    return rsi_values.where(avg_loss != 0.0, 100.0)
