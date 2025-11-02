"""
Core Interfaces
===============
Abstract interfaces for dependency injection and testing.
"""

from .data_fetcher import DataFetcherInterface
from .signal_executor import SignalExecutorInterface
from .alert_handler import AlertHandlerInterface
from .order_executor import OrderExecutorInterface
from .market_analyzer import MarketAnalyzerInterface

__all__ = [
    'DataFetcherInterface',
    'SignalExecutorInterface',
    'AlertHandlerInterface',
    'OrderExecutorInterface',
    'MarketAnalyzerInterface'
]