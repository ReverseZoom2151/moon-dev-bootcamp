"""
Day 4 Algorithmic Orders Module - Refactored
=============================================
Exchange-agnostic implementation of Day 4's algorithmic order execution.

This module now acts as a facade over the modular algo_orders package,
maintaining backward compatibility while providing a cleaner architecture.

Consolidates functionality specifically from:
- Day_4_Projects/binance_algo_orders.py (manual_loop, scheduled_bot, algorithmic_trading)
- Day_4_Projects/binance_bot.py (RSI/Bollinger/MACD strategies with order execution)

This module preserves the exact algorithmic order execution logic from Day 4
while making it exchange-agnostic through the orchestrator interface.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
import logging
from math import log10
import pandas as pd
import pandas_ta as ta
import ccxt  # For InsufficientFunds exception

# Import modular order types
from .algo_orders import (
    AlgoType,
    BaseAlgoOrder,
    ManualLoopOrder,
    ScheduledOrder,
    TWAPOrder,
    VWAPOrder,
    IcebergOrder,
    GridOrder,
    DCAOrder
)


class AlgorithmicTrader:
    """
    Day 4 Algorithmic Order Execution System - Refactored.

    This class maintains the original interface from Day 4's trading algorithms
    but now delegates to modular order type implementations for better
    maintainability and testability.

    Original algorithms from:
    - binance_algo_orders.py: manual_loop(), scheduled_bot(), algorithmic_trading()
    - binance_bot.py: RSI, Bollinger Bands, and MACD strategy implementations

    Preserves the original Day 4 logic while abstracting exchange-specific calls
    through the orchestrator interface.
    """

    def __init__(self, exchange_orchestrator: Any, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize algorithmic trader.

        Args:
            exchange_orchestrator: Main orchestrator instance
            event_bus: Event bus for communication
            config: Configuration dictionary
        """
        self.orchestrator = exchange_orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Algorithm state
        self.active_algorithms: Dict[str, asyncio.Task] = {}
        self.active_orders: Dict[str, BaseAlgoOrder] = {}
        self.algo_stats: Dict[str, Dict] = {}

        # Configuration
        self.max_iterations = self.config.get('max_iterations', None)
        self.max_orders = self.config.get('max_orders', 50)
        self.dry_run = self.config.get('dry_run', False)
        self.dynamic_pricing = self.config.get('dynamic_pricing', False)

        # Performance tracking
        self.order_count = 0
        self.pnl = 0.0
        self.successful_orders = 0
        self.failed_orders = 0

        # Setup event handlers
        self._setup_event_handlers()

    def _setup_event_handlers(self):
        """Setup event handlers."""
        self.event_bus.subscribe("order_filled", self._on_order_filled)
        self.event_bus.subscribe("order_failed", self._on_order_failed)

    async def start_algorithm(self, algo_type: AlgoType, exchange: str, symbol: str,
                             size: float, params: Optional[Dict] = None):
        """
        Start an algorithmic trading strategy.

        Args:
            algo_type: Type of algorithm to run
            exchange: Exchange to trade on
            symbol: Trading symbol
            size: Position size
            params: Algorithm-specific parameters
        """
        algo_id = f"{exchange}_{symbol}_{algo_type.value}"

        if algo_id in self.active_algorithms:
            self.logger.warning(f"Algorithm {algo_id} is already running")
            return

        # Initialize stats
        self.algo_stats[algo_id] = {
            "start_time": datetime.now(),
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_cancelled": 0,
            "total_volume": 0,
            "pnl": 0.0
        }

        # Merge global config with params
        merged_params = {
            'dry_run': self.dry_run,
            'dynamic_pricing': self.dynamic_pricing,
            **(params or {})
        }

        # Create appropriate order type instance
        order_instance = None

        if algo_type == AlgoType.MANUAL_LOOP:
            order_instance = ManualLoopOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.SCHEDULED:
            order_instance = ScheduledOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.CONTINUOUS:
            # Use legacy implementation for continuous trading
            task = asyncio.create_task(
                self.algorithmic_trading_loop(exchange, symbol, size, params)
            )
            self.active_algorithms[algo_id] = task
        elif algo_type == AlgoType.TWAP:
            order_instance = TWAPOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.VWAP:
            order_instance = VWAPOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.ICEBERG:
            order_instance = IcebergOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.GRID:
            order_instance = GridOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        elif algo_type == AlgoType.DCA:
            order_instance = DCAOrder(
                self.orchestrator, self.event_bus, exchange, symbol, size, merged_params
            )
        else:
            self.logger.error(f"Unknown algorithm type: {algo_type}")
            return

        # Start the order instance if created
        if order_instance:
            self.active_orders[algo_id] = order_instance
            task = asyncio.create_task(order_instance.start())
            self.active_algorithms[algo_id] = task

        await self.event_bus.emit("algorithm_started", {
            "algo_id": algo_id,
            "type": algo_type.value,
            "exchange": exchange,
            "symbol": symbol,
            "size": size
        })

        self.logger.info(f"Started algorithm: {algo_id}")

    async def stop_algorithm(self, algo_id: str):
        """Stop a running algorithm."""
        if algo_id not in self.active_algorithms:
            self.logger.warning(f"Algorithm {algo_id} not found")
            return

        # Stop the order instance if it exists
        if algo_id in self.active_orders:
            order_instance = self.active_orders[algo_id]
            order_instance.is_running = False

        task = self.active_algorithms[algo_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.active_algorithms[algo_id]
        if algo_id in self.active_orders:
            del self.active_orders[algo_id]

        # Final stats
        stats = self.algo_stats.get(algo_id, {})
        stats["end_time"] = datetime.now()

        await self.event_bus.emit("algorithm_stopped", {
            "algo_id": algo_id,
            "stats": stats
        })

        self.logger.info(f"Stopped algorithm: {algo_id}")

    # ============================================
    # LEGACY METHODS FOR BACKWARD COMPATIBILITY
    # These delegate to the modular implementations
    # ============================================

    async def manual_loop(self, exchange: str, symbol: str, size: float,
                         params: Optional[Dict] = None):
        """
        Day 4's manual_loop() function - Now delegates to ManualLoopOrder.

        Backward compatibility wrapper.
        """
        params = params or {}
        params['dry_run'] = self.dry_run
        params['dynamic_pricing'] = self.dynamic_pricing

        order = ManualLoopOrder(self.orchestrator, self.event_bus, exchange, symbol, size, params)
        await order.start()

    async def scheduled_bot(self, exchange: str, symbol: str, size: float,
                          params: Optional[Dict] = None):
        """
        Day 4's scheduled_bot() function - Now delegates to ScheduledOrder.

        Backward compatibility wrapper.
        """
        params = params or {}
        params['dry_run'] = self.dry_run
        params['dynamic_pricing'] = self.dynamic_pricing

        order = ScheduledOrder(self.orchestrator, self.event_bus, exchange, symbol, size, params)
        await order.start()

    async def algorithmic_trading_loop(self, exchange: str, symbol: str, size: float,
                                      params: Optional[Dict] = None):
        """
        Day 4's algorithmic_trading() function - Direct implementation.

        Original from Day_4_Projects/binance_algo_orders.py:
        Main algorithmic trading loop with iteration/order limits,
        dynamic pricing, and continuous order placement/cancellation.
        """
        params = params or {}
        wait_time = params.get('wait_time', 5)
        bid_offset = params.get('bid_offset', 0.01)

        self.logger.info(f"Starting algorithmic trading loop for {symbol} on {exchange}")

        iteration = 0
        start_time = datetime.now()

        try:
            while True:
                iteration += 1

                # Check max iterations
                if self.max_iterations and iteration > self.max_iterations:
                    self.logger.info(f"Reached max iterations: {self.max_iterations}")
                    break

                # Check max orders
                if self.order_count >= self.max_orders:
                    self.logger.info(f"Reached max orders: {self.max_orders}")
                    break

                self.logger.info(f"Iteration {iteration}")

                # Get market data
                ticker = await self.orchestrator.get_market_data(exchange, symbol)
                if not ticker:
                    await asyncio.sleep(5)
                    continue

                current_price = ticker[-1][4]  # Close price

                # Get market info for precision
                market_info = await self.get_market_info(exchange, symbol)
                price_precision = market_info.get('price_precision', 0.1)

                # Dynamic pricing with Day 4 style precision rounding
                if self.dynamic_pricing:
                    bid_price = round(current_price * (1 - bid_offset), int(-1 * log10(price_precision)))
                    ask_price = round(current_price * (1 + bid_offset), int(-1 * log10(price_precision)))
                else:
                    bid_price = params.get('fixed_bid', current_price * 0.99)
                    ask_price = params.get('fixed_ask', current_price * 1.01)

                # Place and cancel orders
                if not self.dry_run:
                    # Place buy order
                    buy_order = await self.orchestrator.execute_trade(
                        exchange=exchange,
                        symbol=symbol,
                        side='buy',
                        amount=size,
                        order_type='limit',
                        price=bid_price
                    )

                    if buy_order:
                        self.order_count += 1
                        await asyncio.sleep(wait_time)

                        # Cancel order
                        exchange_instance = self.orchestrator.exchanges[exchange]
                        await exchange_instance.cancel_order(buy_order['id'], symbol)
                else:
                    self.logger.info(f"DRY RUN: Would trade at bid={bid_price}, ask={ask_price}")

                await asyncio.sleep(2)

        except asyncio.CancelledError:
            self.logger.info("Algorithmic trading loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in algorithmic trading loop: {e}")
        finally:
            # Print summary
            duration = datetime.now() - start_time
            self.logger.info(f"Algorithm summary: {iteration} iterations, {self.order_count} orders, Duration: {duration}")

    async def twap_execution(self, exchange: str, symbol: str, total_size: float,
                           params: Optional[Dict] = None):
        """TWAP execution - Delegates to TWAPOrder."""
        params = params or {}
        order = TWAPOrder(self.orchestrator, self.event_bus, exchange, symbol, total_size, params)
        await order.start()

    async def iceberg_order(self, exchange: str, symbol: str, total_size: float,
                          params: Optional[Dict] = None):
        """Iceberg order - Delegates to IcebergOrder."""
        params = params or {}
        order = IcebergOrder(self.orchestrator, self.event_bus, exchange, symbol, total_size, params)
        await order.start()

    async def grid_trading(self, exchange: str, symbol: str, size: float,
                         params: Optional[Dict] = None):
        """Grid trading - Delegates to GridOrder."""
        params = params or {}
        order = GridOrder(self.orchestrator, self.event_bus, exchange, symbol, size, params)
        await order.start()

    async def dollar_cost_averaging(self, exchange: str, symbol: str, amount_per_buy: float,
                                  params: Optional[Dict] = None):
        """DCA - Delegates to DCAOrder."""
        params = params or {}
        order = DCAOrder(self.orchestrator, self.event_bus, exchange, symbol, amount_per_buy, params)
        await order.start()

    # ============================================
    # DAY 4 STRATEGY INTEGRATION (binance_bot.py)
    # ============================================

    async def run_strategy(self, exchange: str, symbol: str, size: float,
                         strategy: str, params: Optional[Dict] = None):
        """
        Day 4's strategy runner from binance_bot.py.

        Executes the three main strategies from Day 4:
        - RSI (Relative Strength Index)
        - Bollinger Bands
        - MACD (Moving Average Convergence Divergence)
        """
        params = params or {}

        if strategy == 'rsi':
            await self.rsi_strategy(exchange, symbol, size, params)
        elif strategy == 'bollinger':
            await self.bollinger_strategy(exchange, symbol, size, params)
        elif strategy == 'macd':
            await self.macd_strategy(exchange, symbol, size, params)
        else:
            self.logger.warning(f"Unknown strategy: {strategy}")

    async def rsi_strategy(self, exchange: str, symbol: str, size: float,
                         params: Optional[Dict] = None):
        """
        Day 4's RSI strategy from binance_bot.py.

        Uses RSI indicator with default thresholds:
        - Buy when RSI < 30 (oversold)
        - Sell when RSI > 70 (overbought)
        """
        params = params or {}
        rsi_period = params.get('rsi_period', 14)
        oversold = params.get('oversold', 30)
        overbought = params.get('overbought', 70)

        # Get OHLCV data
        ohlcv = await self.orchestrator.get_market_data(exchange, symbol, '1h', 100)
        if not ohlcv:
            self.logger.error("Failed to get OHLCV data for RSI")
            return

        # Calculate RSI
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['rsi'] = ta.rsi(df['close'], length=rsi_period)
        current_rsi = df['rsi'].iloc[-1]

        self.logger.info(f"Current RSI: {current_rsi}")

        # Generate signals
        if current_rsi < oversold:
            # Buy signal
            self.logger.info(f"RSI {current_rsi} < {oversold}: BUY signal")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='buy',
                amount=size,
                order_type='market'
            )
        elif current_rsi > overbought:
            # Sell signal
            self.logger.info(f"RSI {current_rsi} > {overbought}: SELL signal")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='sell',
                amount=size,
                order_type='market',
                reduce_only=True
            )

    async def bollinger_strategy(self, exchange: str, symbol: str, size: float,
                                params: Optional[Dict] = None):
        """
        Day 4's Bollinger Bands strategy from binance_bot.py.

        Trades based on price position relative to bands:
        - Buy when price < lower band
        - Sell when price > upper band
        """
        params = params or {}
        period = params.get('period', 20)
        std_dev = params.get('std_dev', 2)

        # Get OHLCV data
        ohlcv = await self.orchestrator.get_market_data(exchange, symbol, '1h', 100)
        if not ohlcv:
            return

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Calculate Bollinger Bands
        bb = ta.bbands(df['close'], length=period, std=std_dev)
        current_price = df['close'].iloc[-1]
        lower_band = bb['BBL_20_2.0'].iloc[-1]
        upper_band = bb['BBU_20_2.0'].iloc[-1]

        # Generate signals
        if current_price < lower_band:
            # Buy signal
            self.logger.info(f"Price {current_price} < Lower BB {lower_band}: BUY")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='buy',
                amount=size,
                order_type='market'
            )
        elif current_price > upper_band:
            # Sell signal
            self.logger.info(f"Price {current_price} > Upper BB {upper_band}: SELL")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='sell',
                amount=size,
                order_type='market',
                reduce_only=True
            )

    async def macd_strategy(self, exchange: str, symbol: str, size: float,
                          params: Optional[Dict] = None):
        """
        Day 4's MACD strategy (mentioned but not fully implemented in binance_bot.py).

        MACD (Moving Average Convergence Divergence):
        - Buy when MACD crosses above signal line
        - Sell when MACD crosses below signal line
        """
        params = params or {}
        fast_period = params.get('fast_period', 12)
        slow_period = params.get('slow_period', 26)
        signal_period = params.get('signal_period', 9)

        # Get OHLCV data
        ohlcv = await self.orchestrator.get_market_data(exchange, symbol, '1h', 100)
        if not ohlcv:
            self.logger.error("Failed to get OHLCV data for MACD")
            return

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # Calculate MACD
        macd = ta.macd(df['close'], fast=fast_period, slow=slow_period, signal=signal_period)
        current_macd = macd['MACD_12_26_9'].iloc[-1]
        current_signal = macd['MACDs_12_26_9'].iloc[-1]
        prev_macd = macd['MACD_12_26_9'].iloc[-2]
        prev_signal = macd['MACDs_12_26_9'].iloc[-2]

        self.logger.info(f"Current MACD: {current_macd}, Signal: {current_signal}")

        # Check for crossovers
        if prev_macd <= prev_signal and current_macd > current_signal:
            # MACD crossed above signal - BUY
            self.logger.info("MACD crossed above signal: BUY signal")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='buy',
                amount=size,
                order_type='market'
            )
        elif prev_macd >= prev_signal and current_macd < current_signal:
            # MACD crossed below signal - SELL
            self.logger.info("MACD crossed below signal: SELL signal")
            await self.orchestrator.execute_trade(
                exchange=exchange,
                symbol=symbol,
                side='sell',
                amount=size,
                order_type='market',
                reduce_only=True
            )

    # ============================================
    # DAY 4 HELPER METHODS
    # ============================================

    async def get_market_info(self, exchange: str, symbol: str) -> Dict:
        """
        Day 4's get_market_info() from binance_algo_orders.py.

        Gets and displays market information including precision and limits.
        """
        try:
            exchange_instance = self.orchestrator.exchanges[exchange]
            market = await exchange_instance.get_market_info(symbol)

            min_amount = market.get('limits', {}).get('amount', {}).get('min', 'Unknown')
            price_precision = market.get('precision', {}).get('price', 0.1)

            self.logger.info(f"Market ID: {market.get('id')}")
            self.logger.info(f"Minimum contract amount: {min_amount}")
            self.logger.info(f"Price precision: {price_precision}")

            return {
                'min_amount': min_amount,
                'price_precision': price_precision,
                'market_id': market.get('id')
            }
        except Exception as e:
            self.logger.error(f"Error fetching market info: {e}")
            return {}

    async def get_current_price(self, exchange: str, symbol: str) -> float:
        """
        Day 4's get_current_price() from binance_algo_orders.py.

        Fetches and returns the current market price.
        """
        try:
            ticker = await self.orchestrator.get_market_data(exchange, symbol)
            if ticker:
                current_price = ticker[-1][4]  # Close price
                self.logger.info(f"Current market price: ${current_price}")
                return current_price
            return 0.0
        except Exception as e:
            self.logger.error(f"Error fetching current price: {e}")
            return 0.0

    async def get_balance(self, exchange: str, currency: str = 'BTC') -> float:
        """
        Day 4's get_balance() from binance_algo_orders.py.

        Fetches and returns account balance for specified currency.
        """
        try:
            exchange_instance = self.orchestrator.exchanges[exchange]
            balance = await exchange_instance.fetch_balance()

            currency_balance = balance.get(currency, {}).get('free', 0)
            self.logger.info(f"Current {currency} balance: {currency_balance}")

            return currency_balance
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return 0.0

    async def ask_bid(self, exchange: str, symbol: str) -> tuple:
        """
        Day 4's ask_bid() function from binance_bot.py.

        Fetches order book and returns (ask, bid) prices.
        Used for getting current market prices for order placement.
        """
        try:
            exchange_instance = self.orchestrator.exchanges[exchange]
            orderbook = await exchange_instance.fetch_order_book(symbol)

            bid = orderbook['bids'][0][0] if orderbook['bids'] else 0.0
            ask = orderbook['asks'][0][0] if orderbook['asks'] else 0.0

            self.logger.info(f'Fetched prices for {symbol}: Ask={ask}, Bid={bid}')
            return ask, bid
        except Exception as e:
            self.logger.error(f'Error fetching order book: {e}')
            return 0.0, 0.0

    # ============================================
    # EVENT HANDLERS
    # ============================================

    async def _on_order_filled(self, event: Dict):
        """Handle order filled event."""
        self.successful_orders += 1
        # Update PnL if needed

    async def _on_order_failed(self, event: Dict):
        """Handle order failed event."""
        self.failed_orders += 1

    # ============================================
    # STATISTICS
    # ============================================

    def get_statistics(self) -> Dict:
        """Get algorithm statistics."""
        return {
            "active_algorithms": list(self.active_algorithms.keys()),
            "total_algorithms": len(self.algo_stats),
            "order_count": self.order_count,
            "successful_orders": self.successful_orders,
            "failed_orders": self.failed_orders,
            "pnl": self.pnl,
            "algo_stats": self.algo_stats
        }

    async def stop_all(self):
        """Stop all running algorithms."""
        algo_ids = list(self.active_algorithms.keys())
        for algo_id in algo_ids:
            await self.stop_algorithm(algo_id)
