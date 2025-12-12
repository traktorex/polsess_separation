"""Python logging utility with colored console output."""

import logging
from pathlib import Path
from typing import Optional

try:
    import coloredlogs

    COLOREDLOGS_AVAILABLE = True
except ImportError:
    COLOREDLOGS_AVAILABLE = False


def setup_logger(
    name: str = "polsess", log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """Setup Python logger with colored console and optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    logger.handlers.clear()

    log_format = "[%(asctime)s] %(levelname)s: %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    if COLOREDLOGS_AVAILABLE:
        coloredlogs.install(
            level=log_level,
            logger=logger,
            fmt=log_format,
            datefmt=date_format,
            level_styles={
                "debug": {"color": "cyan"},
                "info": {"color": "green"},
                "warning": {"color": "yellow", "bold": True},
                "error": {"color": "red", "bold": True},
                "critical": {"color": "red", "bold": True, "background": "white"},
            },
            field_styles={
                "asctime": {"color": "white"},
                "levelname": {"bold": True},
            },
        )
    else:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(log_format, datefmt=date_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        plain_formatter = logging.Formatter(log_format, datefmt=date_format)
        file_handler.setFormatter(plain_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    return logger
