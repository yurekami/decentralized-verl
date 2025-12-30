"""Logging utilities for decentralized veRL."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level
        log_file: Optional file to log to
        format_string: Optional custom format

    Returns:
        Root logger
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Create formatter
    formatter = logging.Formatter(format_string)

    # Setup root logger
    root_logger = logging.getLogger("decentralized_verl")
    root_logger.setLevel(getattr(logging, level.upper()))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
