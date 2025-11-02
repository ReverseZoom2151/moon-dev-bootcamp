"""
Base Exchange Adapter
=====================
Abstract base class for all exchange implementations.
Defines the common interface for exchange operations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import logging


class BaseExchange(ABC):
    """
    Abstract base class for exchange adapters.

    This class defines the interface that all exchange implementations must follow.
    It consolidates common functionality from Days 2-56 of the ATC Bootcamp.
    """

    def __init__(self, credentials: Dict, event_bus: Any):
        """
        Initialize the base exchange.

        Args:
            credentials: Exchange API credentials
            event_bus: Event bus for publishing events
        """
        self.credentials = credentials
        self.event_bus = event_bus
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_connected = False
        self.client = None
        self.positions = {}
        self.orders = {}
        self.balances = {}
        self.symbols_info = {}

    @abstractmethod
    async def initialize(self):
        """Initialize the exchange connection."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the exchange."""
        pass

    @abstractmethod
    async def get_balance(self, asset: Optional[str] = None) -> Dict:
        """
        Get account balance.

        Args:
            asset: Optional specific asset to query

        Returns:
            Balance information
        """
        pass

    @abstractmethod
    async def place_order(self, symbol: str, side: str, amount: float,
                         order_type: str = "market", price: Optional[float] = None,
                         **kwargs) -> Dict:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            amount: Order amount
            order_type: Market, limit, etc.
            price: Price for limit orders
            **kwargs: Additional parameters

        Returns:
            Order information
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel
            symbol: Optional symbol

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Dict:
        """
        Get order information.

        Args:
            order_id: Order ID
            symbol: Optional symbol

        Returns:
            Order details
        """
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get open orders.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of open orders
        """
        pass

    @abstractmethod
    async def get_position(self, symbol: str) -> Dict:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position information
        """
        pass

    @abstractmethod
    async def get_all_positions(self) -> List[Dict]:
        """
        Get all open positions.

        Returns:
            List of positions
        """
        pass

    @abstractmethod
    async def close_position(self, symbol: str, amount: Optional[float] = None) -> Dict:
        """
        Close a position.

        Args:
            symbol: Trading symbol
            amount: Optional partial close amount

        Returns:
            Close order result
        """
        pass

    @abstractmethod
    async def get_ohlcv(self, symbol: str, timeframe: str = "1h",
                       limit: int = 100) -> List[List]:
        """
        Get OHLCV candle data.

        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            limit: Number of candles

        Returns:
            List of OHLCV data
        """
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """
        Get ticker information.

        Args:
            symbol: Trading symbol

        Returns:
            Ticker data
        """
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        """
        Get orderbook.

        Args:
            symbol: Trading symbol
            limit: Depth limit

        Returns:
            Orderbook data
        """
        pass

    # Common utility methods (shared across exchanges)

    async def get_ask_bid(self, symbol: str) -> Tuple[float, float]:
        """
        Get current ask and bid prices.

        Args:
            symbol: Trading symbol

        Returns:
            Tuple of (ask_price, bid_price)
        """
        orderbook = await self.get_orderbook(symbol, limit=1)
        if orderbook:
            ask = orderbook.get("asks", [[0]])[0][0]
            bid = orderbook.get("bids", [[0]])[0][0]
            return float(ask), float(bid)
        return 0.0, 0.0

    async def market_order(self, symbol: str, side: str, amount: float) -> Dict:
        """
        Place a market order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            amount: Order amount

        Returns:
            Order result
        """
        return await self.place_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type="market"
        )

    async def limit_order(self, symbol: str, side: str, amount: float,
                         price: float) -> Dict:
        """
        Place a limit order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            amount: Order amount
            price: Limit price

        Returns:
            Order result
        """
        return await self.place_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type="limit",
            price=price
        )

    async def stop_order(self, symbol: str, side: str, amount: float,
                        stop_price: float) -> Dict:
        """
        Place a stop order.

        Args:
            symbol: Trading symbol
            side: Buy or sell
            amount: Order amount
            stop_price: Stop trigger price

        Returns:
            Order result
        """
        return await self.place_order(
            symbol=symbol,
            side=side,
            amount=amount,
            order_type="stop",
            stop_price=stop_price
        )

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """
        Cancel all open orders.

        Args:
            symbol: Optional filter by symbol

        Returns:
            Number of cancelled orders
        """
        orders = await self.get_open_orders(symbol)
        cancelled = 0

        for order in orders:
            try:
                if await self.cancel_order(order["id"], symbol):
                    cancelled += 1
            except Exception as e:
                self.logger.error(f"Failed to cancel order {order['id']}: {e}")

        return cancelled

    async def close_all_positions(self) -> int:
        """
        Close all open positions.

        Returns:
            Number of closed positions
        """
        positions = await self.get_all_positions()
        closed = 0

        for position in positions:
            try:
                await self.close_position(position["symbol"])
                closed += 1
            except Exception as e:
                self.logger.error(f"Failed to close position {position['symbol']}: {e}")

        return closed

    def calculate_position_size(self, balance: float, risk_percent: float,
                              stop_loss_percent: float, leverage: int = 1) -> float:
        """
        Calculate position size based on risk management.

        Args:
            balance: Account balance
            risk_percent: Risk per trade (%)
            stop_loss_percent: Stop loss distance (%)
            leverage: Leverage to use

        Returns:
            Position size
        """
        risk_amount = balance * (risk_percent / 100)
        position_size = (risk_amount / (stop_loss_percent / 100)) * leverage
        return position_size

    async def get_funding_rate(self, symbol: str) -> float:
        """
        Get funding rate for perpetual contracts.

        Args:
            symbol: Trading symbol

        Returns:
            Current funding rate
        """
        # Default implementation - override in specific exchanges
        return 0.0

    async def get_liquidations(self, symbol: Optional[str] = None) -> List[Dict]:
        """
        Get recent liquidations.

        Args:
            symbol: Optional filter by symbol

        Returns:
            List of liquidations
        """
        # Default implementation - override in specific exchanges
        return []

    async def get_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """
        Get recent trades.

        Args:
            symbol: Trading symbol
            limit: Number of trades

        Returns:
            List of trades
        """
        # Default implementation - override in specific exchanges
        return []

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading symbol
            leverage: Leverage value

        Returns:
            Success status
        """
        # Default implementation - override in specific exchanges
        return False

    async def get_pnl(self, symbol: Optional[str] = None) -> float:
        """
        Get PnL for positions.

        Args:
            symbol: Optional filter by symbol

        Returns:
            Total PnL
        """
        positions = await self.get_all_positions()

        if symbol:
            positions = [p for p in positions if p["symbol"] == symbol]

        total_pnl = sum(p.get("pnl", 0) for p in positions)
        return total_pnl

    async def emit_event(self, event_type: str, data: Any):
        """
        Emit an event to the event bus.

        Args:
            event_type: Type of event
            data: Event data
        """
        if self.event_bus:
            await self.event_bus.emit(event_type, {
                "exchange": self.__class__.__name__,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })

    async def health_check(self) -> bool:
        """
        Check if exchange connection is healthy.

        Returns:
            Health status
        """
        try:
            # Try to get ticker for a common symbol
            await self.get_ticker("BTC/USDT")
            return True
        except Exception:
            return False

    def __str__(self):
        """String representation."""
        return f"{self.__class__.__name__}(connected={self.is_connected})"

    def __repr__(self):
        """Detailed representation."""
        return f"{self.__class__.__name__}(connected={self.is_connected}, positions={len(self.positions)}, orders={len(self.orders)})"