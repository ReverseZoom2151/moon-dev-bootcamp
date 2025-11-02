"""
Base Stream Module
==================
Base class for all market data streams.
Provides common functionality and interface for stream handlers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime
import logging


class BaseStream(ABC):
    """
    Abstract base class for all market data streams.

    All stream types inherit from this class and implement
    the process() method for their specific data type.
    """

    def __init__(self, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize the base stream.

        Args:
            event_bus: Event bus for publishing events
            config: Optional configuration dictionary
        """
        self.event_bus = event_bus
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)

        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'start_time': datetime.now()
        }

    @abstractmethod
    async def process(self, exchange: str, data: Dict) -> None:
        """
        Process incoming stream data.

        Args:
            exchange: Exchange name (e.g., 'binance', 'hyperliquid')
            data: Raw data from the exchange

        This method must be implemented by all subclasses.
        """
        pass

    async def emit_event(self, event_type: str, data: Dict) -> None:
        """
        Emit an event to the event bus with error handling.

        Args:
            event_type: Type of event to emit
            data: Event data
        """
        try:
            await self.event_bus.emit(event_type, data)
        except Exception as e:
            self.logger.error(f"Error emitting event {event_type}: {e}")
            self.stats['total_errors'] += 1

    def get_statistics(self) -> Dict:
        """
        Get stream statistics.

        Returns:
            Dictionary containing stream statistics
        """
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'processing_rate': self.stats['total_processed'] / max(uptime, 1)
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'total_processed': 0,
            'total_errors': 0,
            'start_time': datetime.now()
        }

    def _handle_error(self, error: Exception, context: str = "") -> None:
        """
        Handle errors with consistent logging.

        Args:
            error: The exception that occurred
            context: Additional context about the error
        """
        self.stats['total_errors'] += 1
        error_msg = f"Error in {self.__class__.__name__}"
        if context:
            error_msg += f" ({context})"
        error_msg += f": {error}"
        self.logger.error(error_msg)
