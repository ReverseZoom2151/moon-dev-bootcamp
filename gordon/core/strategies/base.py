"""
Base Strategy Class
====================
Foundation for all trading strategies.
Provides common interface and utilities for strategy implementations.
"""

import logging
from typing import Dict, Optional, Any
from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All strategies must inherit from this class and implement the abstract methods.
    """

    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize base strategy.

        Args:
            name: Strategy name for logging and identification
        """
        self.name = name
        self.logger = logging.getLogger(f"Strategy.{name}")
        self.is_paused = False
        self.interval = 60  # Default execution interval in seconds
        self.config = {}
        self.exchange_connections = {}

    async def initialize(self, config: Dict):
        """
        Initialize strategy with configuration.

        Args:
            config: Strategy configuration dictionary
        """
        self.config = config
        self.logger.info(f"{self.name} initialized with config: {config}")

    @abstractmethod
    async def execute(self) -> Optional[Dict]:
        """
        Execute strategy logic and generate signal.
        Must be implemented by each strategy.

        Returns:
            Trading signal dictionary or None if no action needed
            Signal format: {
                'action': 'BUY' | 'SELL' | 'HOLD',
                'symbol': str,
                'size': float,
                'confidence': float (0-1),
                'metadata': dict
            }
        """
        pass

    async def on_market_update(self, event: Dict):
        """
        Handle market update event.
        Can be overridden by strategies that need real-time data.

        Args:
            event: Market update event
        """
        pass

    async def on_position_update(self, event: Dict):
        """
        Handle position update event.
        Can be overridden by strategies that track positions.

        Args:
            event: Position update event
        """
        pass

    async def pause(self):
        """Pause strategy execution."""
        self.is_paused = True
        self.logger.info(f"{self.name} paused")

    async def resume(self):
        """Resume strategy execution."""
        self.is_paused = False
        self.logger.info(f"{self.name} resumed")

    async def cleanup(self):
        """
        Cleanup strategy resources.
        Should be called when strategy is being stopped.
        """
        self.logger.info(f"{self.name} cleanup completed")

    def set_exchange_connection(self, exchange: str, connection: Any):
        """
        Set exchange connection for direct API calls.

        Args:
            exchange: Exchange name
            connection: Exchange connection object
        """
        self.exchange_connections[exchange] = connection
        self.logger.debug(f"Exchange connection set for {exchange}")

    def get_status(self) -> Dict:
        """
        Get current strategy status.

        Returns:
            Status dictionary with strategy information
        """
        return {
            "name": self.name,
            "is_paused": self.is_paused,
            "interval": self.interval,
            "config": self.config,
            "has_connections": len(self.exchange_connections) > 0
        }

    def validate_signal(self, signal: Dict) -> bool:
        """
        Validate trading signal format.

        Args:
            signal: Trading signal to validate

        Returns:
            True if valid, False otherwise
        """
        required_fields = ['action', 'symbol']
        valid_actions = ['BUY', 'SELL', 'HOLD']

        if not signal:
            return False

        # Check required fields
        for field in required_fields:
            if field not in signal:
                self.logger.error(f"Signal missing required field: {field}")
                return False

        # Validate action
        if signal['action'] not in valid_actions:
            self.logger.error(f"Invalid signal action: {signal['action']}")
            return False

        return True