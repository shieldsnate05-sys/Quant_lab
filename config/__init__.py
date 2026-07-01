"""
Quant-Lab Configuration Package.

This package contains the centralized configuration used throughout
the Quant-Lab platform.
"""

from .logging_config import get_logger
from .paths import PROJECT_ROOT
from .settings import settings

__all__ = [
    "PROJECT_ROOT",
    "get_logger",
    "settings",
]
