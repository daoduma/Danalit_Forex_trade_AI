"""Logging: rotating file handler per component under logs/, plus console.

All timestamps are UTC (ISO-8601 with 'Z' suffix) — matching the project-wide
convention that every stored time is timezone-aware UTC.
"""

from __future__ import annotations

import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from danalit.config import REPO_ROOT

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


class UTCFormatter(logging.Formatter):
    converter = time.gmtime

    def formatTime(self, record, datefmt=None):  # noqa: N802 (stdlib API)
        ct = self.converter(record.created)
        return time.strftime("%Y-%m-%dT%H:%M:%S", ct) + f".{int(record.msecs):03d}Z"


def setup_logging(
    component: str,
    level: int = logging.INFO,
    log_dir: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """Return a configured logger for a component; idempotent per component."""
    logger = logging.getLogger(component)
    if getattr(logger, "_danalit_configured", False):
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = UTCFormatter(_FORMAT)
    log_dir = log_dir or (REPO_ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    fh = RotatingFileHandler(
        log_dir / f"{component}.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger._danalit_configured = True  # type: ignore[attr-defined]
    return logger
