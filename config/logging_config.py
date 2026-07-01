"""
Quant-Lab Logging Configuration.

Centralized logging for the Quant-Lab platform.

Features
--------
- Console logging
- Rotating file logging
- Automatic log directory creation
- Consistent formatting
- Named loggers
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.paths import LOGS

###############################################################################
# Logging Configuration
###############################################################################

LOG_FILE = LOGS / "quant_lab.log"

LOG_LEVEL = logging.INFO

LOG_FORMAT = "%(asctime)s | " "%(levelname)-8s | " "%(name)-25s | " "%(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

MAX_LOG_SIZE = 25 * 1024 * 1024  # 25 MB

BACKUP_COUNT = 10

###############################################################################
# Logger Configuration
###############################################################################


def configure_logging() -> None:
    """
    Configure the application's root logger.

    Safe to call multiple times.
    """

    LOGS.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    root_logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    ###########################################################################
    # Console Handler
    ###########################################################################

    console_handler = logging.StreamHandler()

    console_handler.setLevel(LOG_LEVEL)

    console_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)

    ###########################################################################
    # Rotating File Handler
    ###########################################################################

    file_handler = RotatingFileHandler(
        filename=LOG_FILE,
        mode="a",
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )

    file_handler.setLevel(LOG_LEVEL)

    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)


###############################################################################
# Logger Factory
###############################################################################


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    configure_logging()

    return logging.getLogger(name)


###############################################################################
# Default Logger
###############################################################################

logger = get_logger("Quant-Lab")
