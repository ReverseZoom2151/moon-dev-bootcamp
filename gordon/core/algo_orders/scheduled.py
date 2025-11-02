"""
Scheduled Order Module
======================
Day 4's scheduled_bot() function implementation.

Executes trades at fixed intervals (default 28 seconds).
Used for automated periodic order placement.
"""

import asyncio
import ccxt
from datetime import datetime
from typing import Optional, Dict
from .base import BaseAlgoOrder


class ScheduledOrder(BaseAlgoOrder):
    """
    Day 4's scheduled_bot() algorithmic order.

    Original from Day_4_Projects/binance_algo_orders.py:
    Executes trades at fixed intervals (default 28 seconds).
    Used for automated periodic order placement.
    """

    async def execute(self):
        """Execute the scheduled bot algorithm."""
        interval_seconds = self.params.get('interval', 28)

        self.logger.info(f"Bot scheduled to run every {interval_seconds} seconds")

        # Day 4 style tracking
        successful_runs = 0
        failed_runs = 0

        try:
            while self.is_running:
                # Execute trade
                success = await self._execute_scheduled_trade()

                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1

                # Day 4 style status update every 10 executions
                total_runs = successful_runs + failed_runs
                if total_runs % 10 == 0:
                    await self._print_status_update(successful_runs, failed_runs)

                # Wait for next execution
                await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("Scheduled bot cancelled")
            raise
        except Exception as e:
            self.logger.error(f"Error in scheduled bot: {e}")
            raise
        finally:
            # Day 4 style final summary
            await self._print_final_summary(successful_runs, failed_runs)

    async def _execute_scheduled_trade(self) -> bool:
        """Execute a scheduled trade. Returns True if successful."""
        try:
            # Get current price
            current_price = await self.get_current_price()
            if not current_price:
                return False

            # Calculate order price
            if self.dynamic_pricing:
                bid_price = current_price * 0.99
            else:
                bid_price = self.params.get('fixed_bid', current_price * 0.99)

            # Place order
            if not self.dry_run:
                order = await self.place_order(
                    side='buy',
                    amount=self.size,
                    order_type='limit',
                    price=bid_price
                )

                if order:
                    # Wait and cancel
                    await asyncio.sleep(5)
                    await self.cancel_order(order['id'])
                    return True
                return False
            else:
                self.logger.info(f"DRY RUN: Scheduled trade at {bid_price}")
                return True

        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds in scheduled trade: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Error in scheduled trade: {e}")
            return False

    async def _print_status_update(self, successful: int, failed: int):
        """Print Day 4 style status update."""
        self.logger.info("----- BOT STATUS UPDATE -----")
        self.logger.info(f"Total executions: {successful + failed}")
        self.logger.info(f"Successful: {successful}, Failed: {failed}")

        # Calculate run time
        if self.start_time:
            current_time = datetime.now()
            duration = current_time - self.start_time
            self.logger.info(f"Running for: {duration}")

        # Get current balance
        btc_balance = await self._get_balance('BTC')
        self.logger.info(f"Current BTC balance: {btc_balance}")

    async def _print_final_summary(self, successful: int, failed: int):
        """Print Day 4 style final summary."""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else None

        self.logger.info("===== BOT SUMMARY =====")
        if self.start_time:
            self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        if duration:
            self.logger.info(f"Total run time: {duration}")
        self.logger.info(f"Total executions: {successful + failed}")
        self.logger.info(f"Successful executions: {successful}")
        self.logger.info(f"Failed executions: {failed}")

        # Final balance check
        final_balance = await self._get_balance('BTC')
        self.logger.info(f"Final BTC balance: {final_balance}")

        self.logger.info("Bot execution completed.")

    async def _get_balance(self, currency: str = 'BTC') -> float:
        """Get account balance for specified currency."""
        try:
            exchange_instance = self.orchestrator.exchanges[self.exchange]
            balance = await exchange_instance.fetch_balance()

            currency_balance = balance.get(currency, {}).get('free', 0)
            return currency_balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0
