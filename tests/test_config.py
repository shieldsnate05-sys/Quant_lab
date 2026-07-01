"""Tests for the config package: paths, settings, logging."""

from __future__ import annotations

import logging

from config.logging_config import get_logger
from config.paths import ALL_DIRECTORIES, LOGS, PROJECT_ROOT
from config.settings import Settings, settings


def test_project_root_is_a_directory() -> None:
    assert PROJECT_ROOT.is_dir()


def test_all_directories_exist() -> None:
    for directory in ALL_DIRECTORIES:
        assert directory.is_dir(), f"{directory} was not created."


def test_settings_project_root_matches_config_paths() -> None:
    assert settings.project_root == PROJECT_ROOT


def test_settings_defaults() -> None:
    fresh = Settings()
    assert fresh.data.symbol == "QQQ"
    assert fresh.backtest.initial_cash > 0
    assert 0 < fresh.ml.train_fraction < 1


def test_get_logger_returns_configured_logger() -> None:
    logger = get_logger("quant_lab.tests")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "quant_lab.tests"
    assert logging.getLogger().handlers, "Root logger should have handlers configured."


def test_logs_directory_exists() -> None:
    assert LOGS.is_dir()
