"""
Order Manager Mixin
===================
Provides common order management operations and patterns.

Common patterns identified across exchanges:
- Bracket orders (entry + stop loss + take profit)
- Trailing stop orders
- TWAP (Time-Weighted Average Price) execution
- Order status formatting
- Position closing logic

This mixin provides reusable order management functionality.
"""

import asyncio
from typing import Dict, List, Optional, Union
from datetime import datetime


class OrderManagerMixin:
    """
    Mixin class that provides common order management functionality.

    This mixin adds high-level order management methods that build on
    the basic order placement functionality.

    Usage:
    ------
        class MyExchange(OrderManagerMixin, BaseExchange):
            async def complex_strategy(self):
                # Place a bracket order
                orders = await self.place_bracket_order(
                    'BTC/USDT', 'buy', 0.1, stop_loss=45000, take_profit=55000
                )

                # Execute TWAP
                twap_orders = await self.execute_twap(
                    'ETH/USDT', 'buy', 10, duration_minutes=60, num_slices=12
                )

    Methods provide building blocks for complex trading strategies.
    """

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        stop_loss: float,
        take_profit: float,
        entry_price: Optional[float] = None
    ) -> Dict:
        """
        Place a bracket order: entry order with stop loss and take profit.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            amount: Order amount
            stop_loss: Stop loss price
            take_profit: Take profit price
            entry_price: Optional limit price for entry (None for market order)

        Returns:
            Dictionary containing:
            - main_order: Entry order details
            - stop_loss: Stop loss order details
            - take_profit: Take profit order details

        Examples:
        ---------
            # Buy BTC with stop loss and take profit
            orders = await place_bracket_order(
                'BTC/USDT', 'buy', 0.1,
                stop_loss=45000,
                take_profit=55000
            )
        """
        result = {
            "main_order": {},
            "stop_loss": {},
            "take_profit": {},
            "status": "failed"
        }

        try:
            # Place main entry order
            if entry_price:
                main_order = await self.place_order(
                    symbol, side, amount, order_type="limit", price=entry_price
                )
            else:
                main_order = await self.place_order(
                    symbol, side, amount, order_type="market"
                )

            if not main_order:
                return result

            result["main_order"] = main_order

            # Determine exit side (opposite of entry)
            exit_side = "sell" if side.lower() == "buy" else "buy"

            # Place stop loss order
            try:
                stop_loss_order = await self.place_order(
                    symbol, exit_side, amount,
                    order_type="stop",
                    price=stop_loss
                )
                result["stop_loss"] = stop_loss_order
            except Exception as e:
                self.logger.warning(f"Failed to place stop loss: {e}")

            # Place take profit order
            try:
                take_profit_order = await self.place_order(
                    symbol, exit_side, amount,
                    order_type="limit",
                    price=take_profit
                )
                result["take_profit"] = take_profit_order
            except Exception as e:
                self.logger.warning(f"Failed to place take profit: {e}")

            result["status"] = "success"
            return result

        except Exception as e:
            self.logger.error(f"Failed to place bracket order: {e}")
            return result

    async def place_scaled_entry(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        num_orders: int,
        price_range: tuple
    ) -> List[Dict]:
        """
        Place multiple orders scaled across a price range.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_amount: Total amount to distribute across orders
            num_orders: Number of orders to place
            price_range: Tuple of (min_price, max_price)

        Returns:
            List of placed orders

        Examples:
        ---------
            # Scale into a BTC position across price range
            orders = await place_scaled_entry(
                'BTC/USDT', 'buy', 1.0, 5, (48000, 52000)
            )
        """
        orders = []
        amount_per_order = total_amount / num_orders
        min_price, max_price = price_range
        price_step = (max_price - min_price) / (num_orders - 1) if num_orders > 1 else 0

        for i in range(num_orders):
            price = min_price + (price_step * i)
            try:
                order = await self.place_order(
                    symbol, side, amount_per_order,
                    order_type="limit",
                    price=price
                )
                if order:
                    orders.append(order)
            except Exception as e:
                self.logger.error(f"Failed to place scaled order {i+1}: {e}")

        return orders

    async def execute_twap(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        duration_minutes: int,
        num_slices: int
    ) -> List[Dict]:
        """
        Execute Time-Weighted Average Price (TWAP) order.

        Splits a large order into smaller chunks executed over time.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_amount: Total amount to trade
            duration_minutes: Duration over which to execute
            num_slices: Number of order slices

        Returns:
            List of executed orders

        Examples:
        ---------
            # Buy 10 ETH over 60 minutes in 12 slices (every 5 minutes)
            orders = await execute_twap('ETH/USDT', 'buy', 10, 60, 12)
        """
        slice_amount = total_amount / num_slices
        interval_seconds = (duration_minutes * 60) / num_slices
        orders = []

        self.logger.info(
            f"Starting TWAP execution: {total_amount} {symbol} over {duration_minutes}m "
            f"in {num_slices} slices (every {interval_seconds:.1f}s)"
        )

        for i in range(num_slices):
            try:
                order = await self.place_order(
                    symbol, side, slice_amount, order_type="market"
                )
                if order:
                    orders.append(order)
                    self.logger.info(f"TWAP slice {i+1}/{num_slices} executed")
            except Exception as e:
                self.logger.error(f"Failed to execute TWAP slice {i+1}: {e}")

            # Wait before next slice (unless it's the last one)
            if i < num_slices - 1:
                await asyncio.sleep(interval_seconds)

        self.logger.info(f"TWAP execution completed: {len(orders)}/{num_slices} slices executed")
        return orders

    async def execute_vwap(
        self,
        symbol: str,
        side: str,
        total_amount: float,
        duration_minutes: int,
        volume_profile: Optional[List[float]] = None
    ) -> List[Dict]:
        """
        Execute Volume-Weighted Average Price (VWAP) order.

        Similar to TWAP but adjusts order sizes based on expected volume profile.

        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            total_amount: Total amount to trade
            duration_minutes: Duration over which to execute
            volume_profile: Optional list of volume weights (default: uniform)

        Returns:
            List of executed orders

        Examples:
        ---------
            # Use U-shaped volume profile (higher at open/close)
            profile = [0.15, 0.10, 0.08, 0.08, 0.08, 0.10, 0.12, 0.14, 0.15]
            orders = await execute_vwap('BTC/USDT', 'buy', 1.0, 90, profile)
        """
        # Use uniform profile if none provided
        if volume_profile is None:
            num_slices = max(int(duration_minutes / 5), 1)  # Default: 1 slice per 5 minutes
            volume_profile = [1.0 / num_slices] * num_slices

        # Normalize volume profile to sum to 1.0
        total_weight = sum(volume_profile)
        normalized_profile = [w / total_weight for w in volume_profile]

        num_slices = len(normalized_profile)
        interval_seconds = (duration_minutes * 60) / num_slices
        orders = []

        self.logger.info(f"Starting VWAP execution: {total_amount} {symbol} in {num_slices} slices")

        for i, weight in enumerate(normalized_profile):
            slice_amount = total_amount * weight

            try:
                order = await self.place_order(
                    symbol, side, slice_amount, order_type="market"
                )
                if order:
                    orders.append(order)
                    self.logger.info(f"VWAP slice {i+1}/{num_slices} executed ({weight*100:.1f}% weight)")
            except Exception as e:
                self.logger.error(f"Failed to execute VWAP slice {i+1}: {e}")

            if i < num_slices - 1:
                await asyncio.sleep(interval_seconds)

        return orders

    async def smart_close_position(
        self,
        symbol: str,
        partial_percent: Optional[float] = None,
        use_limit: bool = False,
        limit_offset_percent: float = 0.1
    ) -> Dict:
        """
        Intelligently close a position with optional partial close and limit orders.

        Args:
            symbol: Trading symbol
            partial_percent: Percentage of position to close (None = 100%)
            use_limit: Whether to use limit order instead of market
            limit_offset_percent: Price offset for limit order (% from current price)

        Returns:
            Order result

        Examples:
        ---------
            # Close 50% of position with limit order
            order = await smart_close_position('BTC/USDT', partial_percent=50, use_limit=True)
        """
        try:
            # Get current position
            position = await self.get_position(symbol)
            if not position or position.get("amount", 0) == 0:
                self.logger.warning(f"No position found for {symbol}")
                return {}

            # Calculate close amount
            position_amount = abs(position["amount"])
            if partial_percent:
                close_amount = position_amount * (partial_percent / 100)
            else:
                close_amount = position_amount

            # Determine close side
            position_side = position.get("side", "long")
            close_side = "sell" if position_side == "long" else "buy"

            # Place close order
            if use_limit:
                # Get current price and apply offset
                ticker = await self.get_ticker(symbol)
                current_price = ticker.get("last", 0)

                if close_side == "sell":
                    # Sell slightly above current price
                    limit_price = current_price * (1 + limit_offset_percent / 100)
                else:
                    # Buy slightly below current price
                    limit_price = current_price * (1 - limit_offset_percent / 100)

                return await self.place_order(
                    symbol, close_side, close_amount,
                    order_type="limit",
                    price=limit_price,
                    reduce_only=True
                )
            else:
                # Market order
                return await self.place_order(
                    symbol, close_side, close_amount,
                    order_type="market",
                    reduce_only=True
                )

        except Exception as e:
            self.logger.error(f"Failed to smart close position: {e}")
            return {}

    def format_order_status(self, order: Dict) -> str:
        """
        Format order information as a readable string.

        Args:
            order: Order dictionary

        Returns:
            Formatted string representation

        Examples:
        ---------
            format_order_status(order) ->
            "Order #12345: BUY 0.1 BTC/USDT @ 50000 [FILLED]"
        """
        order_id = order.get("id", "N/A")
        symbol = order.get("symbol", "N/A")
        side = order.get("side", "N/A").upper()
        amount = order.get("amount", 0)
        price = order.get("price", 0)
        status = order.get("status", "N/A").upper()

        return f"Order #{order_id}: {side} {amount} {symbol} @ {price} [{status}]"

    def format_position_status(self, position: Dict) -> str:
        """
        Format position information as a readable string.

        Args:
            position: Position dictionary

        Returns:
            Formatted string representation

        Examples:
        ---------
            format_position_status(position) ->
            "Position: LONG 0.5 BTC/USDT @ 50000 | PnL: +500.00 (+1.00%)"
        """
        symbol = position.get("symbol", "N/A")
        amount = position.get("amount", 0)
        side = "LONG" if amount > 0 else "SHORT"
        entry_price = position.get("entry_price", 0)
        pnl = position.get("pnl", 0)
        pnl_percent = position.get("pnl_percent", 0)

        pnl_sign = "+" if pnl >= 0 else ""

        return (
            f"Position: {side} {abs(amount)} {symbol} @ {entry_price} | "
            f"PnL: {pnl_sign}{pnl:.2f} ({pnl_sign}{pnl_percent:.2f}%)"
        )

    async def get_order_fill_price(self, order_id: str, symbol: str) -> Optional[float]:
        """
        Get the average fill price for an order.

        Args:
            order_id: Order ID
            symbol: Trading symbol

        Returns:
            Average fill price or None if order not filled
        """
        try:
            order = await self.get_order(order_id, symbol)
            if order and order.get("status") in ["closed", "filled"]:
                return order.get("price")
            return None
        except Exception as e:
            self.logger.error(f"Failed to get order fill price: {e}")
            return None

    async def wait_for_order_fill(
        self,
        order_id: str,
        symbol: str,
        timeout: int = 60,
        poll_interval: float = 1.0
    ) -> bool:
        """
        Wait for an order to be filled.

        Args:
            order_id: Order ID
            symbol: Trading symbol
            timeout: Maximum time to wait in seconds
            poll_interval: How often to check order status

        Returns:
            True if order filled, False if timeout

        Examples:
        ---------
            order = await place_order(...)
            filled = await wait_for_order_fill(order['id'], 'BTC/USDT', timeout=30)
            if filled:
                print("Order executed!")
        """
        start_time = asyncio.get_event_loop().time()

        while (asyncio.get_event_loop().time() - start_time) < timeout:
            try:
                order = await self.get_order(order_id, symbol)
                if order and order.get("status") in ["closed", "filled"]:
                    return True
            except Exception as e:
                self.logger.warning(f"Error checking order status: {e}")

            await asyncio.sleep(poll_interval)

        return False

    async def cancel_and_replace_order(
        self,
        order_id: str,
        symbol: str,
        new_price: float
    ) -> Optional[Dict]:
        """
        Cancel an existing order and replace it with a new one at a different price.

        Args:
            order_id: Existing order ID to cancel
            symbol: Trading symbol
            new_price: New price for the replacement order

        Returns:
            New order details or None if failed

        Examples:
        ---------
            # Adjust limit order price
            new_order = await cancel_and_replace_order(
                old_order_id, 'BTC/USDT', new_price=51000
            )
        """
        try:
            # Get original order details
            original_order = await self.get_order(order_id, symbol)
            if not original_order:
                self.logger.error(f"Original order {order_id} not found")
                return None

            # Cancel original order
            cancelled = await self.cancel_order(order_id, symbol)
            if not cancelled:
                self.logger.error(f"Failed to cancel order {order_id}")
                return None

            # Place new order with same parameters but different price
            new_order = await self.place_order(
                symbol=original_order["symbol"],
                side=original_order["side"],
                amount=original_order["amount"],
                order_type="limit",
                price=new_price
            )

            return new_order

        except Exception as e:
            self.logger.error(f"Failed to cancel and replace order: {e}")
            return None
