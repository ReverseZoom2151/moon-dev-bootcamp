"""
Manual Loop Order Module
=========================
Day 4's manual_loop() function implementation.

Places limit orders below market price, waits, then cancels them.
Used for testing order placement/cancellation without fills.
"""

import asyncio
import ccxt
from typing import Optional, Dict
from .base import BaseAlgoOrder


class ManualLoopOrder(BaseAlgoOrder):
    """
    Day 4's manual_loop() algorithmic order.

    Original from Day_4_Projects/binance_algo_orders.py:
    Places limit orders below market price, waits, then cancels them.
    Used for testing order placement/cancellation without fills.
    """

    async def execute(self):
        """Execute the manual loop algorithm."""
        wait_time = self.params.get('wait_time', 10)
        bid_offset = self.params.get('bid_offset', 0.01)  # 1% below market

        loop_count = 0

        try:
            while self.is_running:
                loop_count += 1

                # Get current market price
                current_price = await self.get_current_price()
                if not current_price:
                    self.logger.error("Failed to get market data")
                    await asyncio.sleep(5)
                    continue

                # Calculate bid price
                if self.dynamic_pricing:
                    bid_price = current_price * (1 - bid_offset)
                else:
                    bid_price = self.params.get('fixed_bid', current_price * 0.99)

                self.logger.info(f"Loop {loop_count}: Current price: {current_price}, Bid: {bid_price}")

                # Place order
                if not self.dry_run:
                    order = await self.place_order(
                        side='buy',
                        amount=self.size,
                        order_type='limit',
                        price=bid_price,
                        post_only=True  # GTX equivalent from Day 4 (params = {'timeInForce': 'GTX'})
                    )

                    if order:
                        # Day 4 style countdown before cancelling
                        self.logger.info(f"Waiting {wait_time} seconds before cancellation...")
                        for i in range(wait_time, 0, -1):
                            self.logger.info(f"Cancelling in {i} seconds...")
                            await asyncio.sleep(1)

                        # Cancel order
                        await self.cancel_order(order['id'])
                else:
                    self.logger.info(f"DRY RUN: Would place buy order: {self.size} @ {bid_price}")
                    await asyncio.sleep(wait_time)

                # Brief pause between iterations
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            self.logger.info("Manual loop cancelled")
            raise
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds error: {e}")
            self.logger.info("Pausing for 30 seconds due to insufficient funds...")
            await asyncio.sleep(30)
        except Exception as e:
            self.logger.error(f"Error in manual loop: {e}")
            raise
