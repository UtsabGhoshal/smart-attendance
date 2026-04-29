"""
Structured logging for the Smart Attendance System.

Replaces print() statements with proper Python logging.
Logs to both console AND file (data/logs/app.log).
"""

import logging
import sys
from pathlib import Path

from app.config import settings


def setup_logger(name: str = "attendance") -> logging.Logger:
    """
    Create a configured logger instance.

    Args:
        name: Logger name (used as prefix in log messages).

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(level)

    # Log format
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(fmt)
    logger.addHandler(console)

    # File handler (data/logs/app.log)
    log_dir = settings.DATA_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


# Singleton logger used across the app
logger = setup_logger()
