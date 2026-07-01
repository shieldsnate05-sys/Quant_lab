"""
Quant-Lab EMA Crossover Strategy.

A trend-following strategy that goes long when a fast EMA is above a
slow EMA, and short (or flat, if shorting is disallowed) when it is
below.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from core.constants import EMA_FAST, EMA_SLOW
from core.exceptions import StrategyError
from core.types import OHLCVFrame
from data.schema import SCHEMA
from data.validator import validate_ohlcv_frame
from indicators.trend import ema
from strategies.base import Strategy

logger = get_logger(__name__)


@dataclass(slots=True)
class EMACrossStrategy(Strategy):
    """
    Long/short EMA crossover strategy.

    Parameters
    ----------
    fast_period : int, optional
        Lookback period of the fast EMA. Defaults to
        :data:`core.constants.EMA_FAST`.
    slow_period : int, optional
        Lookback period of the slow EMA. Defaults to
        :data:`core.constants.EMA_SLOW`.
    allow_shorting : bool, optional
        If ``False``, bearish crossovers produce a flat (``0``) signal
        instead of a short (``-1``) signal. Defaults to ``True``.
    """

    fast_period: int = EMA_FAST
    slow_period: int = EMA_SLOW
    allow_shorting: bool = True

    def __post_init__(self) -> None:
        if self.fast_period <= 0 or self.slow_period <= 0:
            raise StrategyError("fast_period and slow_period must be positive.")

        if self.fast_period >= self.slow_period:
            raise StrategyError(
                f"fast_period ({self.fast_period}) must be less than "
                f"slow_period ({self.slow_period})."
            )

    def generate_signals(self, frame: OHLCVFrame) -> pd.Series:
        """
        Generate an EMA-crossover position signal for every bar in ``frame``.

        Parameters
        ----------
        frame : core.types.OHLCVFrame
            OHLCV bars conforming to :data:`data.schema.SCHEMA`.

        Returns
        -------
        pandas.Series
            Integer-valued series aligned to ``frame.index``: ``1`` when
            the fast EMA is above the slow EMA, ``-1`` (or ``0`` if
            :attr:`allow_shorting` is ``False``) when it is below, and
            ``0`` while either EMA is undefined (warm-up period).

        Raises
        ------
        core.exceptions.StrategyError
            If ``frame`` does not have enough bars to compute the slow EMA.
        """
        validate_ohlcv_frame(frame)

        if len(frame) < self.slow_period:
            raise StrategyError(
                f"Frame has {len(frame)} bars, fewer than the required "
                f"slow_period {self.slow_period}."
            )

        close = frame[SCHEMA.close]
        fast_ema = ema(close, self.fast_period)
        slow_ema = ema(close, self.slow_period)

        bullish = fast_ema > slow_ema
        bearish = fast_ema < slow_ema
        warmed_up = fast_ema.notna() & slow_ema.notna()

        short_value = -1 if self.allow_shorting else 0

        signal = np.where(
            bullish & warmed_up, 1, np.where(bearish & warmed_up, short_value, 0)
        )

        logger.info(
            "Generated %d EMA(%d/%d) signals for %d bars.",
            int((signal != 0).sum()),
            self.fast_period,
            self.slow_period,
            len(frame),
        )

        return pd.Series(signal, index=frame.index, name="signal", dtype="int64")
