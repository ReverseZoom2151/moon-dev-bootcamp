"""
DCA (Dollar Cost Averaging) Module
===================================
Dollar Cost Averaging strategy implementation.

Buys fixed amounts at regular intervals regardless of price to
average out market volatility.
"""

import asyncio
from typing import Optional, Dict
from .base import BaseAlgoOrder


class DCAOrder(BaseAlgoOrder):
    """
    Dollar Cost Averaging strategy.

    Executes fixed-size buy orders at regular intervals to average out
    the purchase price over time, reducing the impact of market volatility.
    """

    async def execute(self):
        """Execute the DCA algorithm."""
        interval_hours = self.params.get('interval_hours', 24)
        total_buys = self.params.get('total_buys', 30)

        self.logger.info(f"Starting DCA: {self.size} {self.symbol} every {interval_hours} hours")
        self.logger.info(f"Total scheduled buys: {total_buys}")

        try:
            for i in range(total_buys):
                if not self.is_running:
                    self.logger.info(f"DCA stopped after {i} buys")
                    break

                # Execute buy
                current_price = await self.get_current_price()
                self.logger.info(f"DCA buy {i+1}/{total_buys}: {self.size} {self.symbol} at {current_price}")

                order = await self.place_order(
                    side='buy',
                    amount=self.size,
                    order_type='market'
                )

                if order:
                    self.stats["orders_filled"] += 1
                    self.stats["total_volume"] += self.size
                    self.logger.info(f"DCA buy {i+1}/{total_buys} executed successfully")

                    # Track average purchase price
                    if 'total_spent' not in self.stats:
                        self.stats['total_spent'] = 0
                        self.stats['average_price'] = 0

                    self.stats['total_spent'] += current_price * self.size
                    self.stats['average_price'] = self.stats['total_spent'] / self.stats['total_volume']

                    self.logger.info(f"Average purchase price: {self.stats['average_price']:.2f}")
                else:
                    self.logger.warning(f"DCA buy {i+1}/{total_buys} failed")

                # Wait for next interval
                if i < total_buys - 1:
                    self.logger.info(f"Waiting {interval_hours} hours until next buy...")
                    await asyncio.sleep(interval_hours * 3600)

            # Final summary
            self.logger.info("===== DCA SUMMARY =====")
            self.logger.info(f"Total buys executed: {self.stats['orders_filled']}/{total_buys}")
            self.logger.info(f"Total volume: {self.stats['total_volume']} {self.symbol}")
            if 'average_price' in self.stats:
                self.logger.info(f"Average purchase price: {self.stats['average_price']:.2f}")
                self.logger.info(f"Total spent: {self.stats['total_spent']:.2f}")

        except asyncio.CancelledError:
            self.logger.info(f"DCA cancelled after {self.stats['orders_filled']} buys")
            raise
        except Exception as e:
            self.logger.error(f"Error in DCA: {e}")
            raise
