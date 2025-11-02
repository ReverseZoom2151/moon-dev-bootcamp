"""
Exchange Orchestrator for ATC Bootcamp Projects (Days 2-56)
=============================================================

A unified trading system that consolidates all functionality from Days 2-56
of the ATC Bootcamp, providing a single interface for multi-exchange trading,
strategy management, and operational features.

Author: ATC Bootcamp
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "ATC Bootcamp"

from .core.orchestrator import ExchangeOrchestrator
from .core.event_bus import EventBus
from .exchanges.base import BaseExchange

__all__ = [
    "ExchangeOrchestrator",
    "EventBus",
    "BaseExchange"
]