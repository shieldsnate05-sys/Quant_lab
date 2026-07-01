"""
Quant-Lab CLI Entry Point.

Fetches OHLCV bars for a symbol, runs the EMA crossover strategy
through the vectorized backtest engine, and prints a performance
summary.

Usage
-----
python main.py --symbol QQQ --timeframe 1Day --lookback-days 365
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta

from backtesting.engine import BacktestEngine
from backtesting.results import BacktestResult
from config.logging_config import get_logger
from config.settings import settings
from core.enums import TimeFrame
from core.exceptions import QuantLabError
from data.downloader import AlpacaDownloader
from data.loader import CachedDataLoader
from strategies.ema_cross import EMACrossStrategy

logger = get_logger(__name__)

_TIMEFRAME_CHOICES: dict[str, TimeFrame] = {tf.value: tf for tf in TimeFrame}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments for the Quant-Lab CLI.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    argparse.Namespace
        Parsed ``symbol``, ``timeframe``, and ``lookback_days`` values.
    """
    parser = argparse.ArgumentParser(
        description="Run the Quant-Lab EMA crossover backtest."
    )

    parser.add_argument(
        "--symbol", default=settings.data.symbol, help="Ticker symbol to backtest."
    )
    parser.add_argument(
        "--timeframe",
        default=TimeFrame.DAILY.value,
        choices=sorted(_TIMEFRAME_CHOICES),
        help="Bar timeframe to fetch.",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=365,
        help="Number of calendar days of history to fetch.",
    )

    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> int:
    """
    Run the Quant-Lab CLI: fetch data, backtest, and report results.

    Parameters
    ----------
    argv : list[str] | None, optional
        Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``1`` if a
        :class:`~core.exceptions.QuantLabError` was raised.
    """
    args = parse_args(argv)
    timeframe = _TIMEFRAME_CHOICES[args.timeframe]

    end = datetime.now(UTC)
    start = end - timedelta(days=args.lookback_days)

    try:
        loader = CachedDataLoader(AlpacaDownloader())
        frame = loader.fetch_ohlcv(args.symbol, timeframe, start, end)

        strategy = EMACrossStrategy(allow_shorting=settings.backtest.allow_shorting)
        engine = BacktestEngine()
        result: BacktestResult = engine.run(frame, strategy)
    except QuantLabError:
        logger.exception("Quant-Lab run failed for symbol=%s.", args.symbol)
        return 1

    metrics = result.metrics
    print(f"\nQuant-Lab EMA Crossover Backtest: {args.symbol} ({args.timeframe})")
    print("-" * 60)
    print(f"Total return:          {metrics.total_return:+.2%}")
    print(f"CAGR:                  {metrics.cagr:+.2%}")
    print(f"Annualized volatility: {metrics.annualized_volatility:.2%}")
    print(f"Sharpe ratio:          {metrics.sharpe_ratio:.2f}")
    print(f"Max drawdown:          {metrics.max_drawdown:.2%}")
    print(f"Win rate:              {metrics.win_rate:.2%}")
    print(f"Profit factor:         {metrics.profit_factor:.2f}")
    print(f"Number of trades:      {metrics.num_trades}")

    return 0


if __name__ == "__main__":
    sys.exit(run())
