"""
Quant-Lab Indicator Validation Helpers.

Shared input validation for every indicator function in this package,
so each indicator does not re-implement the same checks.
"""

from __future__ import annotations

from core.exceptions import IndicatorError
from core.types import PriceSeries


def validate_price_series(series: PriceSeries, period: int) -> None:
    """
    Validate that ``series`` is usable for a rolling indicator of ``period``.

    Parameters
    ----------
    series : core.types.PriceSeries
        The input price series.
    period : int
        The lookback period the caller intends to use.

    Raises
    ------
    core.exceptions.IndicatorError
        If ``period`` is not positive, ``series`` is empty, ``series``
        has fewer rows than ``period``, or ``series`` contains null
        values.
    """
    if period <= 0:
        raise IndicatorError(f"period must be positive, got {period}.")

    if series.empty:
        raise IndicatorError("Cannot compute indicator on an empty series.")

    if len(series) < period:
        raise IndicatorError(
            f"Series has {len(series)} rows, fewer than the required period {period}."
        )

    if series.isnull().any():
        raise IndicatorError("Series contains null values.")
