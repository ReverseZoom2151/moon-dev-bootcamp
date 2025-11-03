"""
Database Module
===============
Database layer for persistent storage of trades, positions, and performance metrics.
Similar structure to conversation_memory.py but using SQLAlchemy for structured data.
"""

from .models import (
    Trade,
    Position,
    StrategyMetric,
    RiskMetric,
    PerformanceSnapshot,
    Base
)
from .manager import DatabaseManager
from .event_listener import DatabaseEventListener

__all__ = [
    'Trade',
    'Position',
    'StrategyMetric',
    'RiskMetric',
    'PerformanceSnapshot',
    'Base',
    'DatabaseManager',
    'DatabaseEventListener',
]

