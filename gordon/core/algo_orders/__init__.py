"""
Algorithmic Orders Module
==========================
Modular implementation of algorithmic trading orders.

This module provides exchange-agnostic implementations of various
algorithmic order types, from Day 4's manual_loop and scheduled_bot
to advanced strategies like TWAP, VWAP, and grid trading.

Available Order Types:
    - ManualLoopOrder: Test order placement/cancellation
    - ScheduledOrder: Periodic automated trading
    - TWAPOrder: Time-Weighted Average Price execution
    - VWAPOrder: Volume-Weighted Average Price execution
    - IcebergOrder: Hide large order sizes
    - GridOrder: Grid trading strategy
    - DCAOrder: Dollar Cost Averaging

Usage:
    from gordon.core.algo_orders import ManualLoopOrder, AlgoType

    order = ManualLoopOrder(orchestrator, event_bus, "binance", "BTC/USDT", 0.001)
    await order.start()
"""

from .base import BaseAlgoOrder, AlgoType
from .manual_loop import ManualLoopOrder
from .scheduled import ScheduledOrder
from .twap import TWAPOrder
from .vwap import VWAPOrder
from .iceberg import IcebergOrder
from .grid import GridOrder
from .dca import DCAOrder

__all__ = [
    'BaseAlgoOrder',
    'AlgoType',
    'ManualLoopOrder',
    'ScheduledOrder',
    'TWAPOrder',
    'VWAPOrder',
    'IcebergOrder',
    'GridOrder',
    'DCAOrder',
]

# Version info
__version__ = '1.0.0'
__author__ = 'ATC Bootcamp'
