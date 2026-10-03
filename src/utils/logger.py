"""Shared loguru logger setup.

File logging is best-effort: serverless environments (Vercel, etc.) ship a
read-only filesystem outside /tmp, so a failure to create the log directory
or file sink must not crash the whole app at import time — stderr logging
(which every platform captures) always works regardless.
"""
from __future__ import annotations

import sys

from loguru import logger

from src.config import LOG_DIR

logger.remove()
logger.add(sys.stderr, level="INFO", backtrace=False, diagnose=False)

try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(
        LOG_DIR / "legaldoc.log",
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        backtrace=False,
        diagnose=False,
    )
except OSError:
    logger.warning(f"Could not set up file logging at {LOG_DIR} (read-only filesystem?) — stderr only.")

__all__ = ["logger"]
