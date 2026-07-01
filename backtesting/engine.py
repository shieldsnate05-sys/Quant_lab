"""
Quant-Lab Vectorized Backtest Engine.

Runs a :class:`~strategies.base.Strategy` against OHLCV bars using
vectorized pandas/numpy operations (no per-bar Python loop for the
return calculation).

Execution model
----------------
Signals are shifted forward by one bar before being applied to
returns, so a signal generated from bar ``t``'s close is only realized
in bar ``t + 1``'s return - this avoids look-ahead bias. Commission and
slippage (both expressed as a fraction of notional value, see
:class:`config.settings.BacktestSettings`) are charged proportionally
to the change in position (turnover) at each bar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import cast

import pandas as pd

from backtesting.metrics import PerformanceMetrics, compute_metrics
from backtesting.results import BacktestResult
from config.logging_config import get_logger
from config.settings import BacktestSettings, settings
from core.enums import PositionSide
from core.exceptions import BacktestError
from core.types import OHLCVFrame
from data.schema import SCHEMA
from data.validator import validate_ohlcv_frame
from strategies.base import Strategy

logger = get_logger(__name__)

_SECONDS_PER_YEAR = 365.25 * 24 * 3600


def _infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate the number of bars per year from the median bar spacing."""
    if len(index) < 2:
        return 252.0

    median_delta = index.to_series().diff().dropna().median()
    seconds = median_delta.total_seconds()

    if seconds <= 0:
        return 252.0

    return _SECONDS_PER_YEAR / seconds


def _extract_trades(position: pd.Series, close: pd.Series) -> pd.DataFrame:
    """Build a trade log from a realized (already shifted) position series."""
    previous = position.shift(1).fillna(0.0)
    change_index = position.index[position != previous]

    records: list[dict[str, object]] = []
    open_side = 0.0
    open_time: pd.Timestamp | None = None
    open_price = 0.0

    for timestamp in change_index:
        new_side = float(position.loc[timestamp])
        price = float(close.loc[timestamp])

        if open_side != 0.0:
            return_pct = open_side * (price / open_price - 1.0)
            records.append(
                {
                    "entry_time": open_time,
                    "exit_time": timestamp,
                    "side": PositionSide.LONG if open_side > 0 else PositionSide.SHORT,
                    "entry_price": open_price,
                    "exit_price": price,
                    "return_pct": return_pct,
                    "closed": True,
                }
            )

        if new_side != 0.0:
            open_side, open_time, open_price = new_side, timestamp, price
        else:
            open_side, open_time, open_price = 0.0, None, 0.0

    if open_side != 0.0 and open_time is not None:
        timestamp = position.index[-1]
        price = float(close.loc[timestamp])
        return_pct = open_side * (price / open_price - 1.0)
        records.append(
            {
                "entry_time": open_time,
                "exit_time": timestamp,
                "side": PositionSide.LONG if open_side > 0 else PositionSide.SHORT,
                "entry_price": open_price,
                "exit_price": price,
                "return_pct": return_pct,
                "closed": False,
            }
        )

    columns = [
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "return_pct",
        "closed",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


@dataclass(slots=True)
class BacktestEngine:
    """
    Runs strategies against OHLCV bars and produces a
    :class:`~backtesting.results.BacktestResult`.

    Parameters
    ----------
    backtest_settings : config.settings.BacktestSettings, optional
        Initial cash, commission, slippage, and shorting configuration.
        Defaults to ``config.settings.settings.backtest``.
    """

    backtest_settings: BacktestSettings = field(
        default_factory=lambda: settings.backtest
    )

    def run(self, frame: OHLCVFrame, strategy: Strategy) -> BacktestResult:
        """
        Run ``strategy`` against ``frame`` and return the full result.

        Parameters
        ----------
        frame : core.types.OHLCVFrame
            OHLCV bars conforming to :data:`data.schema.SCHEMA`.
        strategy : strategies.base.Strategy
            The strategy to evaluate.

        Returns
        -------
        backtesting.results.BacktestResult
            Equity curve, returns, realized positions, trade log, and
            performance metrics.

        Raises
        ------
        core.exceptions.BacktestError
            If ``frame`` has fewer than two bars.
        """
        validate_ohlcv_frame(frame)

        if len(frame) < 2:
            raise BacktestError("Cannot run a backtest on fewer than two bars.")

        signal = strategy.generate_signals(frame)

        close = frame[SCHEMA.close]
        market_returns = close.pct_change().fillna(0.0)

        position = signal.shift(1).fillna(0).astype(float)
        if not self.backtest_settings.allow_shorting:
            position = position.clip(lower=0.0)

        gross_returns = position * market_returns

        turnover = position.diff().abs()
        turnover.iloc[0] = abs(position.iloc[0])
        cost_rate = (
            self.backtest_settings.commission_per_trade
            + self.backtest_settings.slippage
        )
        transaction_costs = turnover * cost_rate

        net_returns = gross_returns - transaction_costs
        equity_curve = (
            self.backtest_settings.initial_cash * (1.0 + net_returns).cumprod()
        )
        equity_curve.name = "equity"
        net_returns.name = "returns"

        trades = _extract_trades(position, close)
        # validate_ohlcv_frame already confirmed this is a DatetimeIndex.
        periods_per_year = _infer_periods_per_year(cast(pd.DatetimeIndex, frame.index))
        metrics: PerformanceMetrics = compute_metrics(
            equity_curve, net_returns, trades, periods_per_year
        )

        logger.info(
            "Backtest complete: total_return=%.4f sharpe=%.4f "
            "max_drawdown=%.4f trades=%d",
            metrics.total_return,
            metrics.sharpe_ratio,
            metrics.max_drawdown,
            metrics.num_trades,
        )

        return BacktestResult(
            equity_curve=equity_curve,
            returns=net_returns,
            positions=position.rename("position"),
            trades=trades,
            metrics=metrics,
        )
