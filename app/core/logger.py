"""Logging subsystem for J.A.R.V.I.S.

Provides a thread-safe, centralized logging architecture with formatted console
output and daily rotating file handlers, initialized from core settings.
"""

from __future__ import annotations

import logging
import sys
import threading
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional

from app.core.config import settings

# Thread-safe initialization lock
_LOCK = threading.Lock()
_INITIALIZED = False

# Standard J.A.R.V.I.S Log Format
# Example: 2026-09-06 11:15:21 | INFO | VOICE | Listening...
LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def _setup_root_logging(
    log_dir: Optional[Path] = None,
    log_level_name: Optional[str] = None,
) -> None:
    """Initialize root logger with console and daily rotating file handlers.

    Args:
        log_dir: Directory path where log files will be written.
            Defaults to settings.log_directory.
        log_level_name: Minimum severity level name (e.g., 'INFO', 'DEBUG').
            Defaults to settings.log_level.
    """
    global _INITIALIZED

    with _LOCK:
        if _INITIALIZED:
            return

        target_dir = (
            log_dir.resolve() if log_dir else settings.log_directory.resolve()
        )
        target_dir.mkdir(parents=True, exist_ok=True)

        level_str = (
            log_level_name.upper()
            if log_level_name
            else settings.log_level.upper()
        )
        numeric_level = getattr(logging, level_str, logging.INFO)

        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

        root_logger = logging.getLogger()
        root_logger.setLevel(numeric_level)

        # Remove existing handlers to avoid duplicates
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)

        # 1. Console Handler (stdout with UTF-8 support)
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # 2. Daily Timed Rotating File Handler
        log_file_path = target_dir / "jarvis.log"
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file_path),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.suffix = "%Y-%m-%d"
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Retrieve or create a thread-safe configured logger for a given module.

    Args:
        name: Name of the subsystem or module (e.g., 'VOICE', 'BRAIN', 'SYSTEM').

    Returns:
        logging.Logger instance configured with the J.A.R.V.I.S formatting and handlers.

    Example:
        >>> logger = get_logger("VOICE")
        >>> logger.info("Listening...")
        2026-09-06 11:15:21 | INFO | VOICE | Listening...
    """
    if not _INITIALIZED:
        _setup_root_logging()

    return logging.getLogger(name)
