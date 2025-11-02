"""
Iceberg Order Module
====================
Iceberg order execution strategy.

Shows only a portion of the total order to the market to avoid
revealing the full order size.
"""

import asyncio
from typing import Optional, Dict
from .base import BaseAlgoOrder


class IcebergOrder(BaseAlgoOrder):
    """
    Iceberg order execution.

    Shows only a small portion (the "tip of the iceberg") of the total order
    to the market at any given time, hiding the full order size to minimize
    market impact and prevent information leakage.
    """

    async def execute(self):
        """Execute the iceberg order algorithm."""
        visible_size = self.params.get('visible_size', self.size / 10)
        side = self.params.get('side', 'buy')
        price = self.params.get('price')

        self.logger.info(f"Starting iceberg order: {self.size} {self.symbol}, visible: {visible_size}")

        remaining = self.size

        try:
            while remaining > 0 and self.is_running:
                # Calculate current slice
                current_size = min(visible_size, remaining)

                # Place order
                order = await self.place_order(
                    side=side,
                    amount=current_size,
                    order_type='limit' if price else 'market',
                    price=price
                )

                if order:
                    # Wait for fill or timeout
                    filled = await self._wait_for_fill(order['id'], timeout=30)

                    if filled > 0:
                        remaining -= filled
                        self.stats["orders_filled"] += 1
                        self.stats["total_volume"] += filled
                        self.logger.info(f"Iceberg: {filled} filled, {remaining:.8f} remaining")
                    else:
                        # Cancel unfilled order
                        await self.cancel_order(order['id'])
                        self.logger.info("Order not filled, will retry with new order")
                        await asyncio.sleep(5)
                else:
                    self.logger.warning("Failed to place order, retrying...")
                    await asyncio.sleep(5)

            if remaining <= 0:
                self.logger.info(f"Iceberg order completed: {self.size} {self.symbol} fully executed")
            else:
                self.logger.info(f"Iceberg order stopped with {remaining:.8f} {self.symbol} remaining")

        except asyncio.CancelledError:
            self.logger.info(f"Iceberg order cancelled with {remaining:.8f} {self.symbol} remaining")
            raise
        except Exception as e:
            self.logger.error(f"Error in iceberg order: {e}")
            raise

    async def _wait_for_fill(self, order_id: str, timeout: int = 30) -> float:
        """
        Wait for an order to be filled or timeout.

        Args:
            order_id: Order ID to monitor
            timeout: Maximum time to wait in seconds

        Returns:
            Amount filled
        """
        elapsed = 0
        check_interval = 5

        try:
            exchange_instance = self.orchestrator.exchanges[self.exchange]

            while elapsed < timeout and self.is_running:
                await asyncio.sleep(check_interval)
                elapsed += check_interval

                # Check order status
                order_status = await exchange_instance.get_order(order_id, self.symbol)

                if order_status:
                    filled = order_status.get('filled', 0)
                    status = order_status.get('status', 'unknown')

                    if status in ['closed', 'filled']:
                        return filled
                    elif status == 'cancelled':
                        return filled  # Return partial fill if any
                    elif filled > 0:
                        # Partial fill - continue waiting
                        self.logger.info(f"Order {order_id} partially filled: {filled}")

            return 0.0

        except Exception as e:
            self.logger.error(f"Error checking order status: {e}")
            return 0.0
