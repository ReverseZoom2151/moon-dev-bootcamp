"""
Base Algorithmic Order Module
==============================
Base classes and enums for algorithmic trading orders.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, Any
from enum import Enum


class AlgoType(Enum):
    """Types of algorithmic trading strategies."""
    MANUAL_LOOP = "manual_loop"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    GRID = "grid"
    DCA = "dca"  # Dollar Cost Averaging


class BaseAlgoOrder:
    """
    Base class for all algorithmic order types.

    Provides common functionality for algorithm execution, monitoring,
    and lifecycle management.
    """

    def __init__(self, orchestrator: Any, event_bus: Any,
                 exchange: str, symbol: str, size: float,
                 params: Optional[Dict] = None):
        """
        Initialize base algorithmic order.

        Args:
            orchestrator: Main orchestrator instance (ExchangeOrchestrator)
            event_bus: Event bus for communication
            exchange: Exchange to trade on
            symbol: Trading symbol
            size: Position size
            params: Algorithm-specific parameters
        """
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.exchange = exchange
        self.symbol = symbol
        self.size = size
        self.params = params or {}

        self.logger = logging.getLogger(self.__class__.__name__)

        # Algorithm state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Statistics
        self.stats = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_volume": 0,
            "pnl": 0.0
        }

        # Configuration
        self.dry_run = self.params.get('dry_run', False)
        self.dynamic_pricing = self.params.get('dynamic_pricing', False)

    async def execute(self):
        """
        Execute the algorithmic order.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")

    async def start(self):
        """Start the algorithm execution."""
        self.is_running = True
        self.start_time = datetime.now()
        self.logger.info(f"Starting {self.__class__.__name__} for {self.symbol} on {self.exchange}")

        await self.event_bus.emit("algorithm_started", {
            "type": self.__class__.__name__,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "size": self.size
        })

        try:
            await self.execute()
        except asyncio.CancelledError:
            self.logger.info(f"{self.__class__.__name__} cancelled")
        except Exception as e:
            self.logger.error(f"Error in {self.__class__.__name__}: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the algorithm execution."""
        self.is_running = False
        self.end_time = datetime.now()

        await self.event_bus.emit("algorithm_stopped", {
            "type": self.__class__.__name__,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "stats": self.stats
        })

        self.logger.info(f"Stopped {self.__class__.__name__}")

    async def get_current_price(self) -> float:
        """Get current market price for the symbol."""
        try:
            ticker = await self.orchestrator.get_market_data(self.exchange, self.symbol)
            if ticker:
                return ticker[-1][4]  # Close price
            return 0.0
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return 0.0

    async def get_market_info(self) -> Dict:
        """Get market information including precision and limits."""
        try:
            exchange_instance = self.orchestrator.exchanges[self.exchange]
            market = await exchange_instance.get_market_info(self.symbol)

            return {
                'min_amount': market.get('limits', {}).get('amount', {}).get('min', 'Unknown'),
                'price_precision': market.get('precision', {}).get('price', 0.1),
                'market_id': market.get('id')
            }
        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            return {}

    async def place_order(self, side: str, amount: float, order_type: str = 'market',
                         price: Optional[float] = None, **kwargs) -> Optional[Dict]:
        """
        Place an order on the exchange.

        Args:
            side: 'buy' or 'sell'
            amount: Order amount
            order_type: 'market' or 'limit'
            price: Order price (required for limit orders)
            **kwargs: Additional order parameters

        Returns:
            Order dict if successful, None otherwise
        """
        if self.dry_run:
            self.logger.info(f"DRY RUN: Would place {side} {order_type} order: {amount} @ {price}")
            return None

        try:
            order = await self.orchestrator.execute_trade(
                exchange=self.exchange,
                symbol=self.symbol,
                side=side,
                amount=amount,
                order_type=order_type,
                price=price,
                **kwargs
            )

            if order:
                self.stats["orders_placed"] += 1
                self.logger.info(f"Placed {side} order: {order.get('id')}")

            return order
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            exchange_instance = self.orchestrator.exchanges[self.exchange]
            cancelled = await exchange_instance.cancel_order(order_id, self.symbol)

            if cancelled:
                self.stats["orders_cancelled"] += 1
                self.logger.info(f"Cancelled order: {order_id}")

            return cancelled
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return False

    def get_stats(self) -> Dict:
        """Get algorithm statistics."""
        stats = self.stats.copy()
        stats["start_time"] = self.start_time
        stats["end_time"] = self.end_time

        if self.start_time and self.end_time:
            stats["duration"] = self.end_time - self.start_time
        elif self.start_time:
            stats["duration"] = datetime.now() - self.start_time

        return stats
