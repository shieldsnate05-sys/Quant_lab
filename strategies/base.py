"""
Quant-Lab Strategy Interface.

Defines the abstract contract every trading strategy must implement so
that the backtesting engine can depend on :class:`Strategy` rather
than a concrete strategy implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from core.types import OHLCVFrame


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @abstractmethod
    def generate_signals(self, frame: OHLCVFrame) -> pd.Series:
        """
        Generate a target-position signal for every bar in ``frame``.

        Parameters
        ----------
        frame : core.types.OHLCVFrame
            OHLCV bars conforming to :data:`data.schema.SCHEMA`.

        Returns
        -------
        pandas.Series
            Integer-valued series aligned to ``frame.index`` where each
            entry is one of ``-1`` (short), ``0`` (flat), or ``1``
            (long) - the target position held from that bar's close
            until the next signal change.

        Raises
        ------
        core.exceptions.StrategyError
            If signals cannot be generated for ``frame``.
        """
        raise NotImplementedError
