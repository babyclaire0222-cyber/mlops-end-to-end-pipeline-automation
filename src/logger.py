"""
Centralized structured logging configuration for the pipeline.

Every module in `src/` should obtain its logger via `get_logger(__name__)`
rather than instantiating `logging.getLogger` directly, so that formatting,
log levels, and handlers stay consistent across the whole codebase.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


def _configure_root_logging(log_dir: str = "logs", log_level: int = logging.INFO) -> None:
    """
    Configure the root logger once per process with both a console handler
    and a rotating-friendly file handler.

    Args:
        log_dir: Directory where the log file will be written.
        log_level: Minimum severity level to emit.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "pipeline.log"

    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(filename=str(log_file), encoding="utf-8")
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, log_dir: str = "logs", log_level: int = logging.INFO) -> logging.Logger:
    """
    Retrieve a module-level logger, configuring root handlers on first call.

    Args:
        name: Typically `__name__` of the calling module.
        log_dir: Directory for the persisted log file.
        log_level: Minimum severity level to emit.

    Returns:
        A configured `logging.Logger` instance.
    """
    _configure_root_logging(log_dir=log_dir, log_level=log_level)
    return logging.getLogger(name)
