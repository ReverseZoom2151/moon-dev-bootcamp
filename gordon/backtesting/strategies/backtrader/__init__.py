"""
Backtrader Strategies Module
============================
Contains strategies implemented using the Backtrader framework.
"""

try:
    from .sma_cross_strategy import SmaCrossStrategy
    __all__ = ['SmaCrossStrategy']
except ImportError:
    # Backtrader not available - disable strategies
    SmaCrossStrategy = None  # type: ignore
    __all__ = []
