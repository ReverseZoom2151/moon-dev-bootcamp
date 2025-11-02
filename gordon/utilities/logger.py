"""
Logging Utilities
=================
Centralized logging configuration for the orchestrator.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from termcolor import colored


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for console output."""

    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red'
    }

    ATTRIBUTES = {
        'CRITICAL': ['bold']
    }

    def format(self, record):
        """Format log record with colors."""
        # Get base format
        log_message = super().format(record)

        # Add color if outputting to terminal
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            color = self.COLORS.get(record.levelname, 'white')
            attrs = self.ATTRIBUTES.get(record.levelname, [])
            log_message = colored(log_message, color, attrs=attrs)

        return log_message


def setup_logger(name: str = None, level: str = "INFO",
                log_file: str = None, max_bytes: int = 10485760,
                backup_count: int = 5) -> logging.Logger:
    """
    Setup a logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        max_bytes: Max size for log file rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name or __name__)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        # Create log directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


class TradingLogger:
    """Specialized logger for trading operations."""

    def __init__(self, name: str = "TradingLogger", log_dir: str = "logs"):
        """Initialize trading logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Main logger
        self.logger = setup_logger(
            name,
            log_file=self.log_dir / "trading.log"
        )

        # Separate loggers for different aspects
        self.trade_logger = setup_logger(
            f"{name}.trades",
            log_file=self.log_dir / "trades.log"
        )

        self.error_logger = setup_logger(
            f"{name}.errors",
            level="ERROR",
            log_file=self.log_dir / "errors.log"
        )

        self.performance_logger = setup_logger(
            f"{name}.performance",
            log_file=self.log_dir / "performance.log"
        )

    def log_trade(self, trade_data: dict):
        """Log trade execution."""
        self.trade_logger.info(f"TRADE: {trade_data}")

    def log_error(self, error_msg: str, exc_info=None):
        """Log error with optional exception info."""
        self.error_logger.error(error_msg, exc_info=exc_info)

    def log_performance(self, metrics: dict):
        """Log performance metrics."""
        self.performance_logger.info(f"METRICS: {metrics}")

    def log_liquidation(self, liq_data: dict):
        """Log liquidation event."""
        self.logger.warning(f"LIQUIDATION: {liq_data}")