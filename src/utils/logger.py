"""Shared loguru logger setup."""
from __future__ import annotations

import sys

from loguru import logger

from src.config import LOG_DIR

LOG_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)
logger.add(
    LOG_DIR / "legaldoc.log",
    level="DEBUG",
    rotation="10 MB",
    retention=5,
    backtrace=False,
    diagnose=False,
)

__all__ = ["logger"]
