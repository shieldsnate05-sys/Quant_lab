"""
Quant-Lab Indicator Interface.

Defines the abstract contract every technical indicator implements, so
that features, strategies, and reports can depend on :class:`Indicator`
rather than a concrete indicator implementation, plus shared input
validation used by every concrete indicator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from core.enums import IndicatorGroup
from core.exceptions import IndicatorError
from core.types import FeatureFrame, OHLCVFrame, PriceSeries
from data.validator import validate_ohlcv_frame


class Indicator(ABC):
    """
    Abstract base class for all technical indicators.

    Concrete subclasses are dataclasses whose fields are the
    indicator's parameters (periods, smoothing constants, the source
    column, etc.), with defaults sourced from :mod:`core.constants`.
    Every subclass declares :attr:`name` and :attr:`group` as
    ``ClassVar`` and implements :meth:`compute`.
    """

    #: Short, unique name this indicator is registered under (e.g. ``"EMA"``).
    name: ClassVar[str]

    #: The category this indicator belongs to.
    group: ClassVar[IndicatorGroup]

    @abstractmethod
    def compute(self, frame: OHLCVFrame) -> PriceSeries | FeatureFrame:
        """
        Compute this indicator from OHLCV bars.

        Parameters
        ----------
        frame : core.types.OHLCVFrame
            OHLCV bars conforming to :data:`data.schema.SCHEMA`.

        Returns
        -------
        core.types.PriceSeries | core.types.FeatureFrame
            A single series for single-output indicators (e.g. EMA,
            RSI), or a multi-column frame for indicators that produce
            more than one output (e.g. MACD, Bollinger Bands).

        Raises
        ------
        core.exceptions.IndicatorError
            If ``frame`` does not have enough bars, or otherwise
            cannot be used to compute this indicator.
        """
        raise NotImplementedError


def validate_ohlcv_for_indicator(frame: OHLCVFrame, period: int | None = None) -> None:
    """
    Validate that ``frame`` is usable to compute an indicator.

    Parameters
    ----------
    frame : core.types.OHLCVFrame
        The input OHLCV frame.
    period : int | None, optional
        If given, ``frame`` must have at least this many rows.

    Raises
    ------
    core.exceptions.IndicatorError
        If ``period`` is not positive, or ``frame`` has fewer than
        ``period`` rows.
    core.exceptions.DataError
        If ``frame`` fails structural OHLCV validation.
    """
    validate_ohlcv_frame(frame)

    if period is None:
        return

    if period <= 0:
        raise IndicatorError(f"period must be positive, got {period}.")

    if len(frame) < period:
        raise IndicatorError(
            f"Frame has {len(frame)} rows, fewer than the required period {period}."
        )
