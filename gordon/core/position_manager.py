"""
Position Manager (Enhanced)
============================
Consolidated position management from multiple sources.
Combines basic position tracking with advanced features from Day 20.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any


class PositionManager:
    """
    Centralized position management across all exchanges.

    Consolidates position tracking functionality from:
    - Original position_manager.py
    - strategy_manager.py position methods
    - strategy_manager_enhancement.py AdvancedPositionManager
    """

    def __init__(self, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize position manager.

        Args:
            event_bus: Event bus for communication
            config: Configuration dictionary
        """
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Position tracking
        self.positions: Dict[str, Dict[str, Any]] = {}  # exchange -> symbol -> position
        self.position_history: List[Dict] = []
        self.is_running = False

        # Position limits
        self.max_positions = self.config.get('max_positions', 5)
        self.max_position_size = self.config.get('max_position_size', 10000)  # USD
        self.default_leverage = self.config.get('default_leverage', 1)

        # Risk parameters
        self.default_stop_loss = self.config.get('default_stop_loss', -5.0)  # %
        self.default_take_profit = self.config.get('default_take_profit', 10.0)  # %

        # Exchange connections
        self.exchange_connections = {}

        # Subscribe to events
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers."""
        self.event_bus.subscribe("order_filled", self._on_order_filled)
        self.event_bus.subscribe("position_updated", self._on_position_updated)
        self.event_bus.subscribe("position_closed", self._on_position_closed)
        self.event_bus.subscribe("stop_loss_triggered", self._on_stop_loss)
        self.event_bus.subscribe("take_profit_triggered", self._on_take_profit)

    def set_exchange_connection(self, exchange: str, connection: Any):
        """
        Set exchange connection for direct API calls.

        Args:
            exchange: Exchange name
            connection: Exchange connection object
        """
        self.exchange_connections[exchange] = connection
        self.logger.info(f"Exchange connection set for {exchange}")

    async def start(self):
        """Start position manager."""
        self.is_running = True
        self.logger.info("Position manager started")

        # Start position monitoring
        asyncio.create_task(self._monitor_positions())

    async def stop(self):
        """Stop position manager."""
        self.is_running = False

        # Close all positions if configured
        if self.config.get('close_on_stop', False):
            await self.close_all_positions()

        self.logger.info("Position manager stopped")

    # ========== Basic Position Management ==========

    async def add_position(self, exchange: str, symbol: str, position: Dict):
        """Add or update a position."""
        if exchange not in self.positions:
            self.positions[exchange] = {}

        self.positions[exchange][symbol] = {
            **position,
            "last_updated": datetime.now().isoformat()
        }

        # Emit event
        await self.event_bus.emit("position_added", {
            "exchange": exchange,
            "symbol": symbol,
            "position": position
        })

        # Add to history
        self.position_history.append({
            "timestamp": datetime.now().isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "action": "opened",
            "position": position
        })

    async def remove_position(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Remove a position."""
        if exchange in self.positions and symbol in self.positions[exchange]:
            position = self.positions[exchange][symbol]
            del self.positions[exchange][symbol]

            # Add to history
            self.position_history.append({
                "timestamp": datetime.now().isoformat(),
                "exchange": exchange,
                "symbol": symbol,
                "action": "closed",
                "position": position
            })

            # Emit event
            await self.event_bus.emit("position_removed", {
                "exchange": exchange,
                "symbol": symbol,
                "position": position
            })

            return position
        return None

    async def get_positions(self, exchange: Optional[str] = None) -> List[Dict]:
        """Get positions for an exchange or all exchanges."""
        positions = []

        if exchange:
            if exchange in self.positions:
                for symbol, position in self.positions[exchange].items():
                    positions.append({
                        "exchange": exchange,
                        "symbol": symbol,
                        **position
                    })
        else:
            for ex, symbols in self.positions.items():
                for symbol, position in symbols.items():
                    positions.append({
                        "exchange": ex,
                        "symbol": symbol,
                        **position
                    })

        return positions

    async def get_position(self, exchange: str, symbol: str) -> Optional[Dict]:
        """Get specific position."""
        if exchange in self.positions and symbol in self.positions[exchange]:
            return self.positions[exchange][symbol]
        return None

    # ========== Advanced Position Management (from Day 20) ==========

    async def get_position_info(self, exchange: str, symbol: str) -> Dict:
        """
        Get detailed position information (exchange-agnostic).

        Returns:
            Dict with position information:
            - in_position: bool
            - size: float (positive for long, negative for short)
            - entry_price: float
            - pnl_percent: float
            - is_long: bool or None
        """
        try:
            if exchange not in self.exchange_connections:
                self.logger.error(f"No exchange connection for {exchange}")
                return self._empty_position_info()

            conn = self.exchange_connections[exchange]
            positions = await conn.fetch_positions([symbol])

            if not positions or positions[0]['contracts'] == 0:
                return self._empty_position_info()

            position = positions[0]

            return {
                'in_position': True,
                'size': position['contracts'],
                'entry_price': position.get('entryPrice', 0),
                'pnl_percent': position.get('percentage', 0),
                'is_long': position['side'] == 'long',
                'unrealized_pnl': position.get('unrealizedPnl', 0),
                'margin_used': position.get('collateral', 0)
            }

        except Exception as e:
            self.logger.error(f"Error getting position info for {symbol}: {e}")
            return self._empty_position_info()

    def _empty_position_info(self) -> Dict:
        """Return empty position info structure."""
        return {
            'in_position': False,
            'size': 0,
            'entry_price': 0,
            'pnl_percent': 0,
            'is_long': None,
            'unrealized_pnl': 0,
            'margin_used': 0
        }

    async def adjust_leverage(self, exchange: str, symbol: str, leverage: int) -> bool:
        """
        Adjust leverage for a trading symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            leverage: Leverage value to set

        Returns:
            Success status
        """
        try:
            if exchange not in self.exchange_connections:
                return False

            conn = self.exchange_connections[exchange]
            await conn.set_leverage(leverage, symbol)

            self.logger.info(f"Set leverage to {leverage}x for {symbol} on {exchange}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to adjust leverage: {e}")
            return False

    async def calculate_position_size(self, exchange: str, symbol: str,
                                     usd_size: float, leverage: Optional[int] = None) -> float:
        """
        Calculate position size based on USD amount and leverage.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            usd_size: Position size in USD
            leverage: Leverage to apply (uses default if None)

        Returns:
            Position size in base currency
        """
        try:
            if exchange not in self.exchange_connections:
                return 0

            conn = self.exchange_connections[exchange]
            ticker = await conn.fetch_ticker(symbol)
            price = ticker['last']

            # Use provided leverage or default
            leverage = leverage or self.default_leverage

            # Calculate size
            size = (usd_size / price) * leverage

            # Get symbol precision
            market = await conn.fetch_market(symbol)
            precision = market.get('precision', {}).get('amount', 8)
            size = round(size, precision)

            self.logger.debug(f"Calculated size: {size} {symbol} at {leverage}x")
            return size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    async def kill_switch(self, exchange: str, symbol: str) -> bool:
        """
        Emergency position closure - closes position immediately at market.

        Args:
            exchange: Exchange name
            symbol: Symbol to close position for

        Returns:
            Success status
        """
        try:
            position_info = await self.get_position_info(exchange, symbol)

            if not position_info['in_position']:
                self.logger.info(f"No position to close for {symbol}")
                return True

            self.logger.warning(f"KILL SWITCH activated for {symbol} on {exchange}")

            conn = self.exchange_connections[exchange]

            # Cancel all open orders for this symbol
            await self.cancel_symbol_orders(exchange, symbol)

            # Close position at market
            size = abs(position_info['size'])
            side = 'sell' if position_info['is_long'] else 'buy'

            order = await conn.create_market_order(
                symbol=symbol,
                side=side,
                amount=size,
                params={'reduceOnly': True}
            )

            self.logger.info(f"Kill switch order placed: {order['id']}")

            # Remove from tracking
            await self.remove_position(exchange, symbol)

            return True

        except Exception as e:
            self.logger.error(f"Error in kill switch: {e}")
            return False

    async def pnl_close(self, exchange: str, symbol: str,
                       target: Optional[float] = None,
                       max_loss: Optional[float] = None) -> bool:
        """
        Close positions based on profit/loss targets.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            target: Target profit percentage (uses default if None)
            max_loss: Maximum loss percentage (uses default if None)

        Returns:
            True if position was closed, False otherwise
        """
        try:
            position_info = await self.get_position_info(exchange, symbol)

            if not position_info['in_position']:
                return False

            pnl_percent = position_info['pnl_percent']
            target = target or self.default_take_profit
            max_loss = max_loss or self.default_stop_loss

            if pnl_percent >= target:
                self.logger.info(f"Target reached: {pnl_percent:.2f}% >= {target}% for {symbol}")
                await self.kill_switch(exchange, symbol)
                return True
            elif pnl_percent <= max_loss:
                self.logger.warning(f"Stop loss hit: {pnl_percent:.2f}% <= {max_loss}% for {symbol}")
                await self.kill_switch(exchange, symbol)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error in pnl_close: {e}")
            return False

    async def cancel_symbol_orders(self, exchange: str, symbol: str) -> int:
        """
        Cancel all open orders for a symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Number of orders cancelled
        """
        try:
            if exchange not in self.exchange_connections:
                return 0

            conn = self.exchange_connections[exchange]
            open_orders = await conn.fetch_open_orders(symbol)

            cancelled = 0
            for order in open_orders:
                await conn.cancel_order(order['id'], symbol)
                cancelled += 1
                self.logger.debug(f"Cancelled order {order['id']}")

            if cancelled > 0:
                self.logger.info(f"Cancelled {cancelled} orders for {symbol}")

            return cancelled

        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
            return 0

    async def cancel_all_orders(self, exchange: Optional[str] = None) -> int:
        """
        Cancel all open orders across all symbols.

        Args:
            exchange: Specific exchange or None for all

        Returns:
            Total number of orders cancelled
        """
        total_cancelled = 0

        exchanges = [exchange] if exchange else list(self.exchange_connections.keys())

        for ex in exchanges:
            try:
                conn = self.exchange_connections[ex]
                open_orders = await conn.fetch_open_orders()

                for order in open_orders:
                    await conn.cancel_order(order['id'], order['symbol'])
                    total_cancelled += 1

            except Exception as e:
                self.logger.error(f"Error cancelling orders on {ex}: {e}")

        if total_cancelled > 0:
            self.logger.info(f"Cancelled {total_cancelled} total orders")

        return total_cancelled

    async def close_all_positions(self, exchange: Optional[str] = None):
        """Close all positions for an exchange or all exchanges."""
        positions = await self.get_positions(exchange)

        for position in positions:
            await self.kill_switch(position["exchange"], position["symbol"])

    # ========== Position Monitoring ==========

    async def _monitor_positions(self):
        """Monitor positions for risk management."""
        while self.is_running:
            try:
                # Check all positions
                for exchange, symbols in self.positions.items():
                    for symbol, position in symbols.items():
                        # Check stop loss / take profit
                        await self._check_position_limits(exchange, symbol, position)

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(10)

    async def _check_position_limits(self, exchange: str, symbol: str, position: Dict):
        """Check if position hits any limits."""
        # Check using pnl_close method
        # Note: position param reserved for future use (e.g., cached position data)
        await self.pnl_close(exchange, symbol)

    # ========== Event Handlers ==========

    async def _on_order_filled(self, event: Dict):
        """Handle order filled event."""
        data = event.get("data", {})
        exchange = data.get("exchange")
        symbol = data.get("symbol")
        # Order details available in data.get("order") for future use

        if not exchange or not symbol:
            return

        # Update position based on filled order
        position_info = await self.get_position_info(exchange, symbol)
        if position_info['in_position']:
            await self.add_position(exchange, symbol, position_info)

    async def _on_position_updated(self, event: Dict):
        """Handle position update event."""
        data = event.get("data", {})
        exchange = data.get("exchange")
        symbol = data.get("symbol")
        position = data.get("position")

        if exchange and symbol and position:
            await self.add_position(exchange, symbol, position)

    async def _on_position_closed(self, event: Dict):
        """Handle position closed event."""
        data = event.get("data", {})
        exchange = data.get("exchange")
        symbol = data.get("symbol")

        if exchange and symbol:
            await self.remove_position(exchange, symbol)

    async def _on_stop_loss(self, event: Dict):
        """Handle stop loss triggered event."""
        data = event.get("data", {})
        exchange = data.get("exchange")
        symbol = data.get("symbol")

        self.logger.warning(f"Stop loss triggered for {symbol} on {exchange}")
        await self.kill_switch(exchange, symbol)

    async def _on_take_profit(self, event: Dict):
        """Handle take profit triggered event."""
        data = event.get("data", {})
        exchange = data.get("exchange")
        symbol = data.get("symbol")

        self.logger.info(f"Take profit triggered for {symbol} on {exchange}")
        await self.kill_switch(exchange, symbol)

    # ========== Statistics ==========

    def get_statistics(self) -> Dict:
        """Get position statistics."""
        total_positions = sum(len(symbols) for symbols in self.positions.values())
        total_pnl = 0
        winning_positions = 0
        losing_positions = 0
        total_margin = 0

        for exchange, symbols in self.positions.items():
            for symbol, position in symbols.items():
                pnl = position.get("unrealized_pnl", 0) or position.get("pnl", 0)
                total_pnl += pnl
                total_margin += position.get("margin_used", 0)

                if pnl > 0:
                    winning_positions += 1
                elif pnl < 0:
                    losing_positions += 1

        return {
            "total_positions": total_positions,
            "total_pnl": round(total_pnl, 2),
            "total_margin_used": round(total_margin, 2),
            "winning_positions": winning_positions,
            "losing_positions": losing_positions,
            "win_rate": round(winning_positions / max(total_positions, 1) * 100, 1),
            "position_limit": self.max_positions,
            "positions_available": self.max_positions - total_positions
        }

    def __repr__(self):
        """String representation."""
        stats = self.get_statistics()
        return (f"PositionManager(positions={stats['total_positions']}/{self.max_positions}, "
                f"pnl={stats['total_pnl']}, win_rate={stats['win_rate']}%)")