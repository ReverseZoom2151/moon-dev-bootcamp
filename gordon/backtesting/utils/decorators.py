"""Decorators for backtesting utilities"""

import functools
import time
import logging

logger = logging.getLogger(__name__)


def timeit_backtest(func):
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"{func.__name__} executed in {elapsed:.2f}s")
        return result
    return wrapper
