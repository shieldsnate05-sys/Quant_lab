"""
Quant-Lab Backtesting Package.

Vectorized strategy backtesting: :class:`~backtesting.engine.BacktestEngine`
runs a :class:`~strategies.base.Strategy` against OHLCV bars and produces a
:class:`~backtesting.results.BacktestResult`.
"""

from __future__ import annotations

from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics, compute_metrics
from backtesting.results import BacktestResult

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "PerformanceMetrics",
    "compute_metrics",
]
