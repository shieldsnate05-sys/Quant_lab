"""
config.paths
============

Centralized filesystem paths for Quant-Lab.

This module automatically detects the project root, creates all required
directories, and exposes strongly typed pathlib.Path objects that are used
throughout the application.

Never hardcode filesystem paths anywhere else in the project.
"""

from __future__ import annotations

from pathlib import Path

###############################################################################
# Project Root
###############################################################################

# config/
CONFIG_DIR = Path(__file__).resolve().parent

# Quant_lab/
PROJECT_ROOT = CONFIG_DIR.parent

###############################################################################
# Configuration
###############################################################################

CONFIG = PROJECT_ROOT / "config"

###############################################################################
# Data
###############################################################################

DATA = PROJECT_ROOT / "data"

RAW_DATA = DATA / "raw"

PROCESSED_DATA = DATA / "processed"

CACHE = DATA / "cache"

PARQUET = DATA / "parquet"

EXPORTS = DATA / "exports"

###############################################################################
# Features
###############################################################################

FEATURES = PROJECT_ROOT / "features"

FEATURE_CACHE = FEATURES / "cache"

###############################################################################
# Machine Learning
###############################################################################

ML = PROJECT_ROOT / "ml"

MODELS = ML / "models"

TRAINING = ML / "training"

PREDICTIONS = ML / "predictions"

###############################################################################
# Reports
###############################################################################

REPORTS = PROJECT_ROOT / "reports"

HTML_REPORTS = REPORTS / "html"

PDF_REPORTS = REPORTS / "pdf"

REPORT_TEMPLATES = REPORTS / "templates"

###############################################################################
# Visualization
###############################################################################

VISUALIZATION = PROJECT_ROOT / "visualization"

FIGURES = VISUALIZATION / "figures"

CHARTS = VISUALIZATION / "charts"

###############################################################################
# Backtesting
###############################################################################

BACKTESTING = PROJECT_ROOT / "backtesting"

BACKTEST_RESULTS = BACKTESTING / "results"

TRADES = BACKTESTING / "trades"

###############################################################################
# Optimization
###############################################################################

OPTIMIZATION = PROJECT_ROOT / "optimization"

OPTIMIZATION_RESULTS = OPTIMIZATION / "results"

###############################################################################
# Indicators
###############################################################################

INDICATORS = PROJECT_ROOT / "indicators"

###############################################################################
# Strategies
###############################################################################

STRATEGIES = PROJECT_ROOT / "strategies"

###############################################################################
# Logging
###############################################################################

LOGS = PROJECT_ROOT / "logs"

###############################################################################
# Tests
###############################################################################

TESTS = PROJECT_ROOT / "tests"

TEST_DATA = TESTS / "data"

###############################################################################
# Notebooks
###############################################################################

NOTEBOOKS = PROJECT_ROOT / "notebooks"

###############################################################################
# Temporary Files
###############################################################################

TEMP = PROJECT_ROOT / "temp"

###############################################################################
# Directory Registry
###############################################################################

ALL_DIRECTORIES = (
    DATA,
    RAW_DATA,
    PROCESSED_DATA,
    CACHE,
    PARQUET,
    EXPORTS,
    FEATURES,
    FEATURE_CACHE,
    ML,
    MODELS,
    TRAINING,
    PREDICTIONS,
    REPORTS,
    HTML_REPORTS,
    PDF_REPORTS,
    REPORT_TEMPLATES,
    VISUALIZATION,
    FIGURES,
    CHARTS,
    BACKTESTING,
    BACKTEST_RESULTS,
    TRADES,
    OPTIMIZATION,
    OPTIMIZATION_RESULTS,
    INDICATORS,
    STRATEGIES,
    LOGS,
    TESTS,
    TEST_DATA,
    NOTEBOOKS,
    TEMP,
)

###############################################################################
# Helpers
###############################################################################


def create_directories() -> None:
    """
    Create every required Quant-Lab directory.

    Safe to call multiple times.
    """

    for directory in ALL_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


def project_path(*parts: str) -> Path:
    """
    Construct a path relative to the project root.

    Examples
    --------
    >>> project_path("data", "raw", "qqq.parquet")
    """

    return PROJECT_ROOT.joinpath(*parts)


###############################################################################
# Initialize
###############################################################################

create_directories()
