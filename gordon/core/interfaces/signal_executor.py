"""
Signal Executor Interface
==========================
Abstract interface for executing trading signals across different environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum


class ExecutionMode(Enum):
    """Execution environment modes."""
    LIVE = "live"
    PAPER = "paper"
    BACKTEST = "backtest"
    DRY_RUN = "dry_run"


class SignalExecutorInterface(ABC):
    """
    Abstract interface for executing trading signals.
    Enables consistent execution across live, paper, and backtest modes.
    """

    @abstractmethod
    async def execute_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a trading signal.

        Args:
            signal: Trading signal with keys:
                - action: 'BUY', 'SELL', 'CLOSE'
                - symbol: Trading symbol
                - size: Position size
                - type: 'MARKET', 'LIMIT'
                - price: Limit price (optional)
                - stop_loss: Stop loss price (optional)
                - take_profit: Take profit price (optional)
                - metadata: Additional signal metadata

        Returns:
            Execution result with keys:
                - success: bool
                - order_id: str (if successful)
                - executed_price: float
                - executed_size: float
                - timestamp: datetime
                - error: str (if failed)
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, symbol: str,
                          updates: Dict[str, Any]) -> bool:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            symbol: Trading symbol
            updates: Dictionary of updates (price, size, etc.)

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def close_position(self, symbol: str, size: Optional[float] = None) -> Dict[str, Any]:
        """
        Close an open position.

        Args:
            symbol: Trading symbol
            size: Partial size to close (None for full position)

        Returns:
            Execution result
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get list of open orders.

        Args:
            symbol: Filter by symbol (None for all)

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    def get_execution_mode(self) -> ExecutionMode:
        """
        Get current execution mode.

        Returns:
            Current execution mode
        """
        pass

    @abstractmethod
    def set_execution_mode(self, mode: ExecutionMode):
        """
        Set execution mode.

        Args:
            mode: Execution mode to set
        """
        pass

    @abstractmethod
    async def validate_signal(self, signal: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a trading signal before execution.

        Args:
            signal: Trading signal to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass