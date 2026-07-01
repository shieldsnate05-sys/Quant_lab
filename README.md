# Quant-Lab

A professional quantitative research platform for market data acquisition,
indicator/feature engineering, strategy development, and backtesting.

## Architecture

```
config/        Centralized settings (config.settings) and filesystem paths (config.paths)
core/          Shared exceptions, type aliases, enums, and constants used by every package
data/          OHLCV data loading (Alpaca), Parquet caching, and schema validation
indicators/    Vectorized technical indicators (trend, momentum)
features/      Feature engineering (planned)
strategies/    Trading strategies implementing strategies.base.Strategy
backtesting/   Vectorized backtest engine, performance metrics, and results
optimization/  Parameter optimization (planned)
ml/            Machine-learning models (planned)
reports/       Report generation (planned)
visualization/ Charting (planned)
utils/         Shared utilities
tests/         Test suite (pytest)
```

Every package depends only on `core` and the packages listed above it; `core`
depends on nothing else in the platform. Filesystem paths always come from
`config.paths`, logging always comes from `config.logging_config.get_logger`,
and custom exceptions always derive from `core.exceptions.QuantLabError`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -e ".[dev]"
```

Alpaca API credentials are read from the environment:

```bash
export ALPACA_API_KEY=your_key
export ALPACA_SECRET_KEY=your_secret
```

## Quickstart

Run the EMA crossover strategy against a symbol's daily bars:

```bash
python main.py --symbol QQQ --timeframe 1Day --lookback-days 365
```

This fetches OHLCV bars via `data.AlpacaDataLoader` (cached to Parquet under
`data/parquet/` by `data.CachedDataLoader`), runs
`strategies.EMACrossStrategy` through `backtesting.BacktestEngine`, and
prints a performance summary (total return, CAGR, Sharpe ratio, max
drawdown, win rate, profit factor, and trade count).

## Development

```bash
black .
ruff check .
mypy .
pytest
```

## Bootstrapping a fresh checkout

`bootstrap.py` idempotently creates the directory layout and package
`__init__.py` files this platform expects:

```bash
python bootstrap.py
```
