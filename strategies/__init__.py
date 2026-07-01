"""
Quant-Lab Strategies Package.

Trading strategies implementing the :class:`~strategies.base.Strategy`
interface, consumed by :mod:`backtesting`.
"""

from __future__ import annotations

from strategies.base import Strategy
from strategies.ema_cross import EMACrossStrategy

__all__ = [
    "EMACrossStrategy",
    "Strategy",
]
