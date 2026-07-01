"""
Quant-Lab Backtest Result.

Container for everything a single backtest run produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from backtesting.metrics import PerformanceMetrics


@dataclass(slots=True)
class BacktestResult:
    """
    The full output of a single :class:`~backtesting.engine.BacktestEngine` run.

    Attributes
    ----------
    equity_curve : pandas.Series
        Portfolio value at each bar, starting at the strategy's initial cash.
    returns : pandas.Series
        Per-bar net strategy returns (after commission and slippage).
    positions : pandas.Series
        Realized position (``-1``, ``0``, or ``1``) held at each bar, after
        applying the one-bar execution delay.
    trades : pandas.DataFrame
        One row per completed (or still-open) trade, with columns
        ``entry_time``, ``exit_time``, ``side``, ``entry_price``,
        ``exit_price``, and ``return_pct``.
    metrics : backtesting.metrics.PerformanceMetrics
        Summary performance statistics.
    """

    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: PerformanceMetrics
