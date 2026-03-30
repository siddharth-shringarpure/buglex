"""Logging configuration helper."""

import logging


def configure_logging(
    level: int = logging.INFO,
    log_format: str = "%(asctime)s - %(levelname)s: %(message)s",
    datefmt: str = "%H:%M:%S",
) -> None:
    """Configure root logging for command-line entry points.

    Args:
        level: Logging level (default: logging.INFO)
        log_format: Logging format string
        datefmt: Timestamp format
    """
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=datefmt,
    )
