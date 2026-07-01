"""
Global configuration for Quant-Lab.

This module defines strongly typed configuration objects used throughout the
application. A single Settings instance is created and imported by the rest of
the codebase.

Example
-------
from config.settings import settings

print(settings.data.symbol)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from config.paths import PROJECT_ROOT

# =============================================================================
# Data Settings
# =============================================================================


@dataclass(slots=True)
class DataSettings:
    """Configuration related to market data."""

    symbol: str = "QQQ"

    primary_timeframe: str = "1Min"

    secondary_timeframe: str = "5Min"

    timezone: str = "America/New_York"

    use_adjusted_data: bool = True

    cache_data: bool = True

    max_download_days: int = 3650


# =============================================================================
# Logging Settings
# =============================================================================


@dataclass(slots=True)
class LoggingSettings:
    """Application logging configuration."""

    level: str = "INFO"

    log_directory: str = "logs"

    filename: str = "quant_lab.log"

    max_file_size_mb: int = 25

    backup_count: int = 10

    console_logging: bool = True


# =============================================================================
# Backtest Settings
# =============================================================================


@dataclass(slots=True)
class BacktestSettings:
    """Backtesting defaults."""

    initial_cash: float = 100_000.0

    commission_per_trade: float = 0.00

    slippage: float = 0.00

    allow_shorting: bool = True

    fractional_shares: bool = False


# =============================================================================
# Machine Learning Settings
# =============================================================================


@dataclass(slots=True)
class MLSettings:
    """Machine-learning defaults."""

    random_state: int = 42

    train_fraction: float = 0.70

    validation_fraction: float = 0.15

    test_fraction: float = 0.15

    n_jobs: int = -1


# =============================================================================
# Feature Engineering Settings
# =============================================================================


@dataclass(slots=True)
class FeatureSettings:
    """Feature engineering configuration."""

    generate_lag_features: bool = True

    generate_interaction_features: bool = True

    generate_volatility_features: bool = True

    max_lag: int = 20

    rolling_windows: list[int] = field(
        default_factory=lambda: [5, 10, 20, 50, 100, 200]
    )


# =============================================================================
# Alpaca Configuration
# =============================================================================


@dataclass(slots=True)
class AlpacaSettings:
    """Alpaca API configuration, sourced from environment variables."""

    api_key: str = field(default_factory=lambda: os.environ.get("ALPACA_API_KEY", ""))

    secret_key: str = field(
        default_factory=lambda: os.environ.get("ALPACA_SECRET_KEY", "")
    )

    paper_trading: bool = True

    data_feed: str = "iex"


# =============================================================================
# Global Settings
# =============================================================================


@dataclass(slots=True)
class Settings:
    """Top-level application settings."""

    project_name: str = "Quant-Lab"

    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)

    data: DataSettings = field(default_factory=DataSettings)

    logging: LoggingSettings = field(default_factory=LoggingSettings)

    backtest: BacktestSettings = field(default_factory=BacktestSettings)

    ml: MLSettings = field(default_factory=MLSettings)

    features: FeatureSettings = field(default_factory=FeatureSettings)

    alpaca: AlpacaSettings = field(default_factory=AlpacaSettings)


settings = Settings()
