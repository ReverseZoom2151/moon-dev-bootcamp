"""
Grid Trading Module
===================
Grid trading strategy implementation.

Places multiple orders at different price levels to profit from
price oscillations.
"""

import asyncio
from typing import Optional, Dict, List
from .base import BaseAlgoOrder


class GridOrder(BaseAlgoOrder):
    """
    Grid trading strategy.

    Places multiple buy and sell orders at predetermined price levels
    around a center price, creating a "grid" of orders that profit from
    price oscillations within a range.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_orders: List[Dict] = []

    async def execute(self):
        """Execute the grid trading algorithm."""
        grid_levels = self.params.get('levels', 10)
        grid_spacing = self.params.get('spacing', 0.005)  # 0.5% spacing

        # Get current price
        center_price = await self.get_current_price()
        if not center_price:
            self.logger.error("Failed to get market data for grid trading")
            return

        self.logger.info(f"Starting grid trading: {grid_levels} levels around {center_price}")
        self.logger.info(f"Grid spacing: {grid_spacing * 100}%")

        try:
            # Place grid orders
            await self._place_grid_orders(center_price, grid_levels, grid_spacing)

            self.logger.info(f"Placed {len(self.active_orders)} grid orders")

            # Monitor grid
            await self._monitor_grid(center_price, grid_levels, grid_spacing)

        except asyncio.CancelledError:
            self.logger.info("Grid trading cancelled")
            await self._cancel_all_orders()
            raise
        except Exception as e:
            self.logger.error(f"Error in grid trading: {e}")
            await self._cancel_all_orders()
            raise

    async def _place_grid_orders(self, center_price: float, levels: int, spacing: float):
        """Place all grid orders."""
        self.active_orders = []

        for i in range(levels // 2):
            if not self.is_running:
                break

            # Buy orders below center
            buy_price = center_price * (1 - spacing * (i + 1))
            buy_order = await self.place_order(
                side='buy',
                amount=self.size,
                order_type='limit',
                price=buy_price
            )
            if buy_order:
                self.active_orders.append({
                    'order': buy_order,
                    'side': 'buy',
                    'price': buy_price,
                    'level': -(i + 1)
                })
                self.logger.info(f"Grid buy order placed at {buy_price} (level {-(i+1)})")

            # Sell orders above center
            sell_price = center_price * (1 + spacing * (i + 1))
            sell_order = await self.place_order(
                side='sell',
                amount=self.size,
                order_type='limit',
                price=sell_price
            )
            if sell_order:
                self.active_orders.append({
                    'order': sell_order,
                    'side': 'sell',
                    'price': sell_price,
                    'level': (i + 1)
                })
                self.logger.info(f"Grid sell order placed at {sell_price} (level {i+1})")

            # Small delay between order placements
            await asyncio.sleep(0.5)

    async def _monitor_grid(self, center_price: float, levels: int, spacing: float):
        """Monitor grid and replace filled orders."""
        check_interval = self.params.get('check_interval', 60)  # Check every minute

        while self.is_running:
            await asyncio.sleep(check_interval)

            if not self.is_running:
                break

            try:
                # Check status of all grid orders
                filled_orders = await self._check_filled_orders()

                if filled_orders:
                    self.logger.info(f"Grid: {len(filled_orders)} orders filled")

                    # Replace filled orders
                    for filled in filled_orders:
                        await self._replace_filled_order(filled, center_price, spacing)

                # Optional: Adjust grid if price moved significantly
                current_price = await self.get_current_price()
                if current_price and abs(current_price - center_price) / center_price > 0.1:
                    self.logger.info(f"Price moved significantly from {center_price} to {current_price}")
                    # Could reposition grid here if desired

            except Exception as e:
                self.logger.error(f"Error monitoring grid: {e}")

    async def _check_filled_orders(self) -> List[Dict]:
        """Check which orders have been filled."""
        filled_orders = []
        exchange_instance = self.orchestrator.exchanges[self.exchange]

        for grid_order in self.active_orders[:]:  # Copy list to allow modifications
            try:
                order_status = await exchange_instance.get_order(
                    grid_order['order']['id'],
                    self.symbol
                )

                if order_status and order_status.get('status') in ['closed', 'filled']:
                    filled_orders.append(grid_order)
                    self.active_orders.remove(grid_order)
                    self.stats["orders_filled"] += 1
                    self.stats["total_volume"] += order_status.get('filled', 0)

            except Exception as e:
                self.logger.error(f"Error checking order {grid_order['order']['id']}: {e}")

        return filled_orders

    async def _replace_filled_order(self, filled_order: Dict, center_price: float, spacing: float):
        """Replace a filled order with a new one on the opposite side."""
        side = filled_order['side']
        level = filled_order['level']

        # Place opposite order
        if side == 'buy':
            # Filled buy, place sell above
            new_price = filled_order['price'] * (1 + spacing)
            new_side = 'sell'
        else:
            # Filled sell, place buy below
            new_price = filled_order['price'] * (1 - spacing)
            new_side = 'buy'

        new_order = await self.place_order(
            side=new_side,
            amount=self.size,
            order_type='limit',
            price=new_price
        )

        if new_order:
            self.active_orders.append({
                'order': new_order,
                'side': new_side,
                'price': new_price,
                'level': -level  # Opposite level
            })
            self.logger.info(f"Replaced filled {side} order with {new_side} at {new_price}")

    async def _cancel_all_orders(self):
        """Cancel all active grid orders."""
        self.logger.info(f"Cancelling {len(self.active_orders)} grid orders")

        for grid_order in self.active_orders:
            try:
                await self.cancel_order(grid_order['order']['id'])
            except Exception as e:
                self.logger.error(f"Error cancelling order {grid_order['order']['id']}: {e}")

        self.active_orders = []
