"""
TWAP (Time-Weighted Average Price) Module
==========================================
Time-Weighted Average Price execution strategy.

Splits large orders over time to minimize market impact.
"""

import asyncio
from typing import Optional, Dict
from .base import BaseAlgoOrder


class TWAPOrder(BaseAlgoOrder):
    """
    Time-Weighted Average Price execution.

    Splits large orders into smaller slices executed at regular intervals
    to minimize market impact and achieve average price over time period.
    """

    async def execute(self):
        """Execute the TWAP algorithm."""
        duration_minutes = self.params.get('duration', 60)
        num_slices = self.params.get('slices', 20)
        side = self.params.get('side', 'buy')

        slice_size = self.size / num_slices
        interval = (duration_minutes * 60) / num_slices

        self.logger.info(f"Starting TWAP: {self.size} {self.symbol} over {duration_minutes} minutes")
        self.logger.info(f"Slice size: {slice_size}, Interval: {interval}s, Total slices: {num_slices}")

        try:
            for i in range(num_slices):
                if not self.is_running:
                    self.logger.info("TWAP execution stopped")
                    break

                # Execute slice
                order = await self.place_order(
                    side=side,
                    amount=slice_size,
                    order_type='market'
                )

                if order:
                    self.stats["orders_filled"] += 1
                    self.stats["total_volume"] += slice_size
                    self.logger.info(f"TWAP slice {i+1}/{num_slices} executed: {slice_size} {self.symbol}")

                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(interval)

            self.logger.info(f"TWAP execution completed: {self.stats['orders_filled']}/{num_slices} slices filled")

        except asyncio.CancelledError:
            self.logger.info(f"TWAP execution cancelled after {self.stats['orders_filled']} slices")
            raise
        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {e}")
            raise
