"""
Quant-Lab Backtest Performance Metrics.

Computes standard performance statistics (return, risk, and trade
statistics) from a backtest's equity curve and trade log.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.constants import FLOAT_TOLERANCE
from core.exceptions import BacktestError


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """Summary performance statistics for a single backtest run."""

    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int


def compute_metrics(
    equity_curve: pd.Series,
    returns: pd.Series,
    trades: pd.DataFrame,
    periods_per_year: float,
) -> PerformanceMetrics:
    """
    Compute :class:`PerformanceMetrics` from a backtest's outputs.

    Parameters
    ----------
    equity_curve : pandas.Series
        Portfolio value at each bar.
    returns : pandas.Series
        Per-bar net strategy returns, aligned to ``equity_curve``.
    trades : pandas.DataFrame
        Trade log with a ``return_pct`` column, as produced by
        :func:`backtesting.engine.BacktestEngine.run`.
    periods_per_year : float
        Number of bars per year, used to annualize return and
        volatility.

    Returns
    -------
    PerformanceMetrics
        The computed statistics.

    Raises
    ------
    core.exceptions.BacktestError
        If ``equity_curve`` is empty.
    """
    if equity_curve.empty:
        raise BacktestError("Cannot compute metrics on an empty equity curve.")

    start_value = float(equity_curve.iloc[0])
    end_value = float(equity_curve.iloc[-1])

    total_return = end_value / start_value - 1.0

    num_periods = len(returns)
    years = num_periods / periods_per_year if periods_per_year > 0 else 0.0
    if years > 0 and start_value > 0:
        cagr = (end_value / start_value) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    volatility = float(returns.std(ddof=0) * np.sqrt(periods_per_year))
    mean_return = float(returns.mean() * periods_per_year)
    sharpe_ratio = mean_return / volatility if volatility > FLOAT_TOLERANCE else 0.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min())

    if trades.empty:
        win_rate = 0.0
        profit_factor = 0.0
    else:
        win_rate = float((trades["return_pct"] > 0).mean())
        gross_profit = float(trades.loc[trades["return_pct"] > 0, "return_pct"].sum())
        gross_loss = float(-trades.loc[trades["return_pct"] < 0, "return_pct"].sum())
        if gross_loss > FLOAT_TOLERANCE:
            profit_factor = gross_profit / gross_loss
        else:
            profit_factor = float("inf") if gross_profit > 0 else 0.0

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        annualized_volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(trades),
    )
