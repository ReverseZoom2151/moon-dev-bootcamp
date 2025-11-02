"""
VWAP (Volume-Weighted Average Price) Module
============================================
Volume-Weighted Average Price execution strategy.

Splits orders based on historical volume patterns to minimize market impact.
"""

import asyncio
from typing import Optional, Dict
from .base import BaseAlgoOrder


class VWAPOrder(BaseAlgoOrder):
    """
    Volume-Weighted Average Price execution.

    Distributes order execution across time periods weighted by historical
    volume patterns to minimize market impact and achieve better average prices.
    """

    async def execute(self):
        """Execute the VWAP algorithm."""
        duration_minutes = self.params.get('duration', 60)
        num_slices = self.params.get('slices', 20)
        side = self.params.get('side', 'buy')

        self.logger.info(f"Starting VWAP: {self.size} {self.symbol} over {duration_minutes} minutes")

        try:
            # Get historical volume data
            volume_weights = await self._calculate_volume_weights(num_slices)

            interval = (duration_minutes * 60) / num_slices

            for i in range(num_slices):
                if not self.is_running:
                    self.logger.info("VWAP execution stopped")
                    break

                # Calculate slice size based on volume weight
                slice_size = self.size * volume_weights[i]

                # Execute slice
                order = await self.place_order(
                    side=side,
                    amount=slice_size,
                    order_type='market'
                )

                if order:
                    self.stats["orders_filled"] += 1
                    self.stats["total_volume"] += slice_size
                    self.logger.info(f"VWAP slice {i+1}/{num_slices} executed: {slice_size} {self.symbol} (weight: {volume_weights[i]:.3f})")

                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(interval)

            self.logger.info(f"VWAP execution completed: {self.stats['orders_filled']}/{num_slices} slices filled")

        except asyncio.CancelledError:
            self.logger.info(f"VWAP execution cancelled after {self.stats['orders_filled']} slices")
            raise
        except Exception as e:
            self.logger.error(f"Error in VWAP execution: {e}")
            raise

    async def _calculate_volume_weights(self, num_slices: int) -> list:
        """
        Calculate volume weights based on historical data.

        In production, this would analyze historical volume patterns.
        For now, we use a simplified distribution.

        Args:
            num_slices: Number of slices to create

        Returns:
            List of volume weights (normalized to sum to 1.0)
        """
        try:
            # Get historical OHLCV data
            ohlcv = await self.orchestrator.get_market_data(
                self.exchange,
                self.symbol,
                timeframe='1h',
                limit=num_slices * 2
            )

            if ohlcv and len(ohlcv) >= num_slices:
                # Extract volumes from the most recent periods
                volumes = [candle[5] for candle in ohlcv[-num_slices:]]  # Volume is index 5

                # Normalize to sum to 1.0
                total_volume = sum(volumes)
                if total_volume > 0:
                    weights = [v / total_volume for v in volumes]
                    self.logger.info(f"Calculated volume weights from historical data")
                    return weights

        except Exception as e:
            self.logger.warning(f"Could not calculate volume weights from historical data: {e}")

        # Fallback: use uniform distribution
        self.logger.info(f"Using uniform volume distribution (no historical data)")
        weight = 1.0 / num_slices
        return [weight] * num_slices
