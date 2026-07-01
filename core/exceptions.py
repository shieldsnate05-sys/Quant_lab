"""
Quant-Lab Core Exceptions.

Every custom exception raised anywhere in the platform derives from
:class:`QuantLabError`. Modules must not define their own exception
hierarchies; add a new subclass here instead so callers can catch
platform errors with a single ``except QuantLabError``.
"""

from __future__ import annotations


class QuantLabError(Exception):
    """Base exception for every error raised by the Quant-Lab platform."""


class ConfigurationError(QuantLabError):
    """Raised when application configuration is missing or invalid."""


class DataError(QuantLabError):
    """Raised when market data cannot be retrieved, cached, or parsed."""


class ValidationError(QuantLabError):
    """Raised when input data fails a validation check."""


class IndicatorError(QuantLabError):
    """Raised when an indicator cannot be computed."""


class FeatureError(QuantLabError):
    """Raised when feature engineering fails."""


class StrategyError(QuantLabError):
    """Raised when a trading strategy fails to generate signals."""


class BacktestError(QuantLabError):
    """Raised when a backtest cannot be run or produces invalid results."""


class OptimizationError(QuantLabError):
    """Raised when a parameter optimization run fails."""


class ModelError(QuantLabError):
    """Raised when a machine-learning model fails to train or predict."""


class ReportError(QuantLabError):
    """Raised when report generation fails."""
