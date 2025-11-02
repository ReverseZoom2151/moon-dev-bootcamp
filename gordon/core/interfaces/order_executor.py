"""
Order Executor Interface
=========================
Abstract interface for order execution and management.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from enum import Enum


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    OPEN = "open"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderExecutorInterface(ABC):
    """
    Abstract interface for order execution.
    Provides consistent order management across exchanges.
    """

    @abstractmethod
    async def place_order(self, symbol: str, side: OrderSide,
                         order_type: OrderType, size: float,
                         price: Optional[float] = None,
                         params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            order_type: Type of order
            size: Order size
            price: Order price (for limit orders)
            params: Additional parameters

        Returns:
            Order result with id, status, etc.
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, symbol: str,
                          size: Optional[float] = None,
                          price: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            size: New size (optional)
            price: New price (optional)

        Returns:
            Modified order details
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get order details.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Order details
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get open orders.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_order_history(self, symbol: Optional[str] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history.

        Args:
            symbol: Filter by symbol
            limit: Number of orders

        Returns:
            List of historical orders
        """
        pass

    @abstractmethod
    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            Number of orders cancelled
        """
        pass