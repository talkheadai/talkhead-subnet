"""Executor logging configuration.

PM2 routes stdout -> *-out.log and stderr -> *-error.log. Loguru defaults to
stderr, but we configure it explicitly so executor logs stay out of out.log
alongside bt.logging (which writes to stdout).
"""

from __future__ import annotations

import os
import sys

from loguru import logger

_CONFIGURED = False


def configure_executor_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = os.getenv("EXECUTOR_LOG_LEVEL", "INFO").upper()
    logger.remove()
    logger.add(sys.stderr, level=level)
    _CONFIGURED = True
