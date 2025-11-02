"""
Account Monitor
===============
Monitors account balance, exposure, and minimum balance requirements.
"""

import asyncio
from typing import Dict, Optional, Tuple
from .base_manager import BaseRiskManager


class AccountMonitor(BaseRiskManager):
    """
    Handles account monitoring:
    - Balance tracking
    - Minimum balance enforcement
    - Total exposure calculation
    - Account value monitoring
    - Kill switches for account breaches
    """

    def __init__(self, event_bus, config_manager, demo_mode: bool = False):
        """
        Initialize account monitor.

        Args:
            event_bus: Event bus for communication
            config_manager: Configuration manager
            demo_mode: Whether to run in demo mode
        """
        super().__init__(event_bus, config_manager, demo_mode)

        # Account parameters
        self.account_minimum_balance = self.risk_config.get("account_minimum_balance", 7)
        self.monitoring_interval = self.risk_config.get("monitoring_interval", 60)

        # Tracking
        self.current_balances = {}  # exchange -> balance
        self.total_exposure = {}  # exchange -> exposure

    async def get_account_value(self, exchange: str) -> float:
        """
        Get total account value for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Account value in USD
        """
        if self.demo_mode:
            return 10000.0  # Demo balance

        try:
            if exchange not in self.exchange_connections:
                self.logger.warning(f"No exchange connection for {exchange}")
                return 0.0

            conn = self.exchange_connections[exchange]
            balance = await conn.fetch_balance()

            # Try different balance fields (Day 5 style)
            if 'info' in balance and 'availableBalance' in balance['info']:
                value = float(balance['info']['availableBalance'])
            elif 'free' in balance and 'USDT' in balance['free']:
                value = float(balance['free']['USDT'])
            elif 'USD' in balance:
                value = float(balance['USD']['free'])
            else:
                # Sum all free balances
                value = sum(
                    float(v.get('free', 0)) for v in balance.values()
                    if isinstance(v, dict) and 'free' in v
                )

            self.current_balances[exchange] = value
            self.logger.info(f"Account value for {exchange}: ${value:.2f}")

            return value

        except Exception as e:
            self.logger.error(f"Error getting account value for {exchange}: {e}")
            return 0.0

    async def check_account_minimum(self, exchange: str) -> bool:
        """
        Check if account balance is below minimum threshold.

        Args:
            exchange: Exchange name

        Returns:
            True if below minimum, False otherwise
        """
        if self.demo_mode:
            return False

        try:
            balance = await self.get_account_value(exchange)

            if balance < self.account_minimum_balance:
                self.logger.warning(
                    f"Account balance ${balance:.2f} below minimum ${self.account_minimum_balance}"
                )

                # Emit emergency stop event
                await self.emit_event("account_minimum_breach", {
                    "exchange": exchange,
                    "balance": balance,
                    "minimum": self.account_minimum_balance
                })

                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking account minimum for {exchange}: {e}")
            return False

    async def calculate_total_exposure(self, exchange: str) -> float:
        """
        Calculate total account exposure across all positions.

        Args:
            exchange: Exchange name

        Returns:
            Total exposure in USD
        """
        if self.demo_mode:
            return 1000.0  # Demo exposure

        try:
            if exchange not in self.exchange_connections:
                return 0.0

            conn = self.exchange_connections[exchange]
            positions = await conn.fetch_positions()

            total_exposure = sum(abs(pos.get('notional', 0)) for pos in positions)

            self.total_exposure[exchange] = total_exposure
            self.logger.info(f"Total exposure for {exchange}: ${total_exposure:.2f}")

            return total_exposure

        except Exception as e:
            self.logger.error(f"Error calculating exposure for {exchange}: {e}")
            return 0.0

    async def get_account_metrics(self, exchange: str) -> Dict:
        """
        Get comprehensive account metrics.

        Args:
            exchange: Exchange name

        Returns:
            Dictionary of account metrics
        """
        balance = await self.get_account_value(exchange)
        exposure = await self.calculate_total_exposure(exchange)

        exposure_ratio = (exposure / balance * 100) if balance > 0 else 0

        return {
            "exchange": exchange,
            "balance": balance,
            "exposure": exposure,
            "exposure_ratio_percent": exposure_ratio,
            "minimum_balance": self.account_minimum_balance,
            "above_minimum": balance >= self.account_minimum_balance,
            "free_margin": balance - exposure
        }

    async def get_ask_bid(self, exchange: str, symbol: str) -> Tuple[float, float, Dict]:
        """
        Get current ask and bid prices.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (ask_price, bid_price, order_book)
        """
        if self.demo_mode:
            return 42010, 42000, {}

        try:
            if exchange not in self.exchange_connections:
                self.logger.error(f"No exchange connection for {exchange}")
                return 0, 0, {}

            conn = self.exchange_connections[exchange]
            order_book = await conn.fetch_order_book(symbol)

            if not order_book or 'bids' not in order_book or 'asks' not in order_book:
                self.logger.error(f"Invalid order book for {symbol}")
                return 0, 0, {}

            bid = order_book['bids'][0][0] if order_book['bids'] else 0
            ask = order_book['asks'][0][0] if order_book['asks'] else 0

            self.logger.info(f"Prices for {symbol} - Ask: {ask}, Bid: {bid}")
            return ask, bid, order_book

        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return 0, 0, {}

    async def get_precision_info(self, exchange: str, symbol: str) -> Tuple[int, int]:
        """
        Get size and price decimal precision for a symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Tuple of (size_decimals, price_decimals)
        """
        if self.demo_mode:
            return 8, 2

        try:
            if exchange not in self.exchange_connections:
                return 8, 2

            conn = self.exchange_connections[exchange]
            market = await conn.market(symbol)

            if not market:
                self.logger.warning(f"No market info for {symbol}, using defaults")
                return 8, 2

            # Extract precision info
            size_decimals = market.get('precision', {}).get('amount', 8)
            price_decimals = market.get('precision', {}).get('price', 2)

            self.logger.info(f"{symbol} precision - Size: {size_decimals}, Price: {price_decimals}")
            return size_decimals, price_decimals

        except Exception as e:
            self.logger.error(f"Error getting precision for {symbol}: {e}")
            return 8, 2

    async def place_limit_order_with_retry(
        self,
        exchange: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        reduce_only: bool = False,
        max_retries: int = 3
    ) -> Optional[Dict]:
        """
        Place limit order with retry logic.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            side: 'buy' or 'sell'
            size: Order size
            price: Limit price
            reduce_only: Whether order reduces position only
            max_retries: Maximum retry attempts

        Returns:
            Order result or None if failed
        """
        if self.demo_mode:
            self.logger.info(f"[DEMO] Would place {side} order for {symbol}: {size} @ {price}")
            return {'id': 'demo_order', 'status': 'demo'}

        for attempt in range(max_retries):
            try:
                if exchange not in self.exchange_connections:
                    self.logger.error(f"No exchange connection for {exchange}")
                    return None

                conn = self.exchange_connections[exchange]
                params = {'reduceOnly': reduce_only} if reduce_only else {}

                order = await conn.create_limit_order(
                    symbol=symbol,
                    side=side,
                    amount=size,
                    price=price,
                    params=params
                )

                self.logger.info(
                    f"Limit {side.upper()} order placed for {symbol}: "
                    f"{size} @ {price} (reduce_only={reduce_only})"
                )
                return order

            except Exception as e:
                self.logger.error(f"Order attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    self.logger.error(f"Failed to place order after {max_retries} attempts")
                    return None

    async def cancel_all_orders_for_symbol(self, exchange: str, symbol: str) -> bool:
        """
        Cancel all open orders for a symbol.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Success status
        """
        if self.demo_mode:
            self.logger.info(f"[DEMO] All orders cancelled for {symbol}")
            return True

        try:
            if exchange not in self.exchange_connections:
                return False

            conn = self.exchange_connections[exchange]
            await conn.cancel_all_orders(symbol)
            self.logger.info(f"All orders cancelled for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error cancelling orders for {symbol}: {e}")
            return False

    def update_minimum_balance(self, new_minimum: float):
        """
        Update minimum balance requirement.

        Args:
            new_minimum: New minimum balance
        """
        old_minimum = self.account_minimum_balance
        self.account_minimum_balance = new_minimum

        self.logger.info(
            f"Minimum balance updated: ${new_minimum:.2f} (previous: ${old_minimum:.2f})"
        )

    def get_balance_status(self, exchange: str) -> Dict:
        """
        Get current balance status for an exchange.

        Args:
            exchange: Exchange name

        Returns:
            Dictionary with balance status
        """
        balance = self.current_balances.get(exchange, 0)
        exposure = self.total_exposure.get(exchange, 0)

        return {
            "exchange": exchange,
            "current_balance": balance,
            "minimum_balance": self.account_minimum_balance,
            "above_minimum": balance >= self.account_minimum_balance,
            "total_exposure": exposure,
            "free_margin": balance - exposure
        }
