"""
Day 4 Algorithmic Orders Module
================================
Exchange-agnostic implementation of Day 4's algorithmic order execution.

Consolidates functionality specifically from:
- Day_4_Projects/binance_algo_orders.py (manual_loop, scheduled_bot, algorithmic_trading)
- Day_4_Projects/binance_bot.py (RSI/Bollinger/MACD strategies with order execution)

This module preserves the exact algorithmic order execution logic from Day 4
while making it exchange-agnostic through the orchestrator interface.
"""

import asyncio
from datetime import datetime
from typing import Dict, Optional, Any
from enum import Enum
import logging
from math import log10
import pandas as pd
import pandas_ta as ta
import ccxt  # For InsufficientFunds exception


class AlgoType(Enum):
    """Types of algorithmic trading strategies."""
    MANUAL_LOOP = "manual_loop"
    SCHEDULED = "scheduled"
    CONTINUOUS = "continuous"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price
    ICEBERG = "iceberg"
    GRID = "grid"
    DCA = "dca"  # Dollar Cost Averaging


class AlgorithmicTrader:
    """
    Day 4 Algorithmic Order Execution System.

    This class is a direct port of Day 4's trading algorithms:
    - binance_algo_orders.py: manual_loop(), scheduled_bot(), algorithmic_trading()
    - binance_bot.py: RSI, Bollinger Bands, and MACD strategy implementations

    Preserves the original Day 4 logic while abstracting exchange-specific calls
    through the orchestrator interface.
    """

    def __init__(self, orchestrator: Any, event_bus: Any, config: Optional[Dict] = None):
        """
        Initialize algorithmic trader.

        Args:
            orchestrator: Main orchestrator instance (ExchangeOrchestrator)
            event_bus: Event bus for communication
            config: Configuration dictionary
        """
        self.orchestrator = orchestrator
        self.event_bus = event_bus
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Algorithm state
        self.active_algorithms: Dict[str, asyncio.Task] = {}
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

        # Start the appropriate algorithm
        if algo_type == AlgoType.MANUAL_LOOP:
            task = asyncio.create_task(
                self.manual_loop(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.SCHEDULED:
            task = asyncio.create_task(
                self.scheduled_bot(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.CONTINUOUS:
            task = asyncio.create_task(
                self.algorithmic_trading_loop(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.TWAP:
            task = asyncio.create_task(
                self.twap_execution(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.ICEBERG:
            task = asyncio.create_task(
                self.iceberg_order(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.GRID:
            task = asyncio.create_task(
                self.grid_trading(exchange, symbol, size, params)
            )
        elif algo_type == AlgoType.DCA:
            task = asyncio.create_task(
                self.dollar_cost_averaging(exchange, symbol, size, params)
            )
        else:
            self.logger.error(f"Unknown algorithm type: {algo_type}")
            return

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

        task = self.active_algorithms[algo_id]
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        del self.active_algorithms[algo_id]

        # Final stats
        stats = self.algo_stats.get(algo_id, {})
        stats["end_time"] = datetime.now()

        await self.event_bus.emit("algorithm_stopped", {
            "algo_id": algo_id,
            "stats": stats
        })

        self.logger.info(f"Stopped algorithm: {algo_id}")

    # ============================================
    # DAY 4 CORE ALGORITHMS (binance_algo_orders.py)
    # ============================================

    async def manual_loop(self, exchange: str, symbol: str, size: float,
                         params: Optional[Dict] = None):
        """
        Day 4's manual_loop() function - Direct implementation.

        Original from Day_4_Projects/binance_algo_orders.py:
        Places limit orders below market price, waits, then cancels them.
        Used for testing order placement/cancellation without fills.
        """
        params = params or {}
        wait_time = params.get('wait_time', 10)
        bid_offset = params.get('bid_offset', 0.01)  # 1% below market

        self.logger.info(f"Starting manual loop for {symbol} on {exchange}")
        loop_count = 0

        try:
            while True:
                loop_count += 1

                # Get current market price
                ticker = await self.orchestrator.get_market_data(exchange, symbol)
                if not ticker:
                    self.logger.error("Failed to get market data")
                    await asyncio.sleep(5)
                    continue

                current_price = ticker[-1][4] if ticker else 0  # Close price

                # Calculate bid price
                if self.dynamic_pricing:
                    bid_price = current_price * (1 - bid_offset)
                else:
                    bid_price = params.get('fixed_bid', current_price * 0.99)

                self.logger.info(f"Loop {loop_count}: Current price: {current_price}, Bid: {bid_price}")

                # Place order
                if not self.dry_run:
                    order = await self.orchestrator.execute_trade(
                        exchange=exchange,
                        symbol=symbol,
                        side='buy',
                        amount=size,
                        order_type='limit',
                        price=bid_price,
                        post_only=True  # GTX equivalent from Day 4 (params = {'timeInForce': 'GTX'})
                    )

                    if order:
                        self.algo_stats[f"{exchange}_{symbol}_manual_loop"]["orders_placed"] += 1

                        # Day 4 style countdown before cancelling
                        self.logger.info(f"Waiting {wait_time} seconds before cancellation...")
                        for i in range(wait_time, 0, -1):
                            self.logger.info(f"Cancelling in {i} seconds...")
                            await asyncio.sleep(1)

                        # Cancel order
                        exchange_instance = self.orchestrator.exchanges[exchange]
                        cancelled = await exchange_instance.cancel_order(order['id'], symbol)

                        if cancelled:
                            self.algo_stats[f"{exchange}_{symbol}_manual_loop"]["orders_cancelled"] += 1
                else:
                    self.logger.info(f"DRY RUN: Would place buy order: {size} @ {bid_price}")
                    await asyncio.sleep(wait_time)

                # Brief pause between iterations
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            self.logger.info("Manual loop cancelled")
        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds error: {e}")
            self.logger.info("Pausing for 30 seconds due to insufficient funds...")
            await asyncio.sleep(30)
        except Exception as e:
            self.logger.error(f"Error in manual loop: {e}")

    async def scheduled_bot(self, exchange: str, symbol: str, size: float,
                          params: Optional[Dict] = None):
        """
        Day 4's scheduled_bot() function - Direct implementation.

        Original from Day_4_Projects/binance_algo_orders.py:
        Executes trades at fixed intervals (default 28 seconds).
        Used for automated periodic order placement.
        """
        params = params or {}
        interval_seconds = params.get('interval', 28)

        self.logger.info(f"Starting scheduled bot for {symbol} on {exchange}")
        self.logger.info(f"Bot scheduled to run every {interval_seconds} seconds")

        # Day 4 style tracking
        successful_runs = 0
        failed_runs = 0
        start_time = datetime.now()

        try:
            while True:
                # Execute trade
                success = await self._execute_scheduled_trade(exchange, symbol, size, params)

                if success:
                    successful_runs += 1
                else:
                    failed_runs += 1

                # Day 4 style status update every 10 executions
                total_runs = successful_runs + failed_runs
                if total_runs % 10 == 0:
                    self.logger.info("----- BOT STATUS UPDATE -----")
                    self.logger.info(f"Total executions: {total_runs}")
                    self.logger.info(f"Successful: {successful_runs}, Failed: {failed_runs}")

                    # Calculate run time
                    current_time = datetime.now()
                    duration = current_time - start_time
                    self.logger.info(f"Running for: {duration}")

                    # Get current balance
                    btc_balance = await self.get_balance(exchange, 'BTC')
                    self.logger.info(f"Current BTC balance: {btc_balance}")

                # Wait for next execution
                await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("Scheduled bot cancelled")
        except Exception as e:
            self.logger.error(f"Error in scheduled bot: {e}")
        finally:
            # Day 4 style final summary
            end_time = datetime.now()
            duration = end_time - start_time

            self.logger.info("===== BOT SUMMARY =====")
            self.logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total run time: {duration}")
            self.logger.info(f"Total executions: {successful_runs + failed_runs}")
            self.logger.info(f"Successful executions: {successful_runs}")
            self.logger.info(f"Failed executions: {failed_runs}")

            # Final balance check
            final_balance = await self.get_balance(exchange, 'BTC')
            self.logger.info(f"Final BTC balance: {final_balance}")

            self.logger.info("Bot execution completed.")

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

    # ============================================
    # EXTENDED ALGORITHMS (Beyond Day 4)
    # These are additional common algo order types
    # ============================================

    async def twap_execution(self, exchange: str, symbol: str, total_size: float,
                           params: Optional[Dict] = None):
        """
        Time-Weighted Average Price execution.
        Splits large orders over time to minimize market impact.
        """
        params = params or {}
        duration_minutes = params.get('duration', 60)
        num_slices = params.get('slices', 20)

        slice_size = total_size / num_slices
        interval = (duration_minutes * 60) / num_slices

        self.logger.info(f"Starting TWAP: {total_size} {symbol} over {duration_minutes} minutes")

        try:
            for i in range(num_slices):
                # Execute slice
                order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side=params.get('side', 'buy'),
                    amount=slice_size,
                    order_type='market'
                )

                if order:
                    self.logger.info(f"TWAP slice {i+1}/{num_slices} executed")

                # Wait for next slice
                if i < num_slices - 1:
                    await asyncio.sleep(interval)

        except asyncio.CancelledError:
            self.logger.info("TWAP execution cancelled")
        except Exception as e:
            self.logger.error(f"Error in TWAP execution: {e}")

    async def iceberg_order(self, exchange: str, symbol: str, total_size: float,
                          params: Optional[Dict] = None):
        """
        Iceberg order execution.
        Shows only a portion of the total order to the market.
        """
        params = params or {}
        visible_size = params.get('visible_size', total_size / 10)
        side = params.get('side', 'buy')
        price = params.get('price')

        self.logger.info(f"Starting iceberg order: {total_size} {symbol}, visible: {visible_size}")

        remaining = total_size

        try:
            while remaining > 0:
                # Calculate current slice
                current_size = min(visible_size, remaining)

                # Place order
                order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side=side,
                    amount=current_size,
                    order_type='limit' if price else 'market',
                    price=price
                )

                if order:
                    # Wait for fill or timeout
                    await asyncio.sleep(30)  # Wait 30 seconds

                    # Check if filled
                    exchange_instance = self.orchestrator.exchanges[exchange]
                    order_status = await exchange_instance.get_order(order['id'], symbol)

                    if order_status and order_status.get('filled', 0) > 0:
                        remaining -= order_status['filled']
                        self.logger.info(f"Iceberg: {order_status['filled']} filled, {remaining} remaining")
                    else:
                        # Cancel unfilled order
                        await exchange_instance.cancel_order(order['id'], symbol)
                        await asyncio.sleep(5)

        except asyncio.CancelledError:
            self.logger.info("Iceberg order cancelled")
        except Exception as e:
            self.logger.error(f"Error in iceberg order: {e}")

    async def grid_trading(self, exchange: str, symbol: str, size: float,
                         params: Optional[Dict] = None):
        """
        Grid trading strategy.
        Places multiple orders at different price levels.
        """
        params = params or {}
        grid_levels = params.get('levels', 10)
        grid_spacing = params.get('spacing', 0.005)  # 0.5% spacing

        # Get current price
        ticker = await self.orchestrator.get_market_data(exchange, symbol)
        if not ticker:
            self.logger.error("Failed to get market data for grid trading")
            return

        center_price = ticker[-1][4]

        self.logger.info(f"Starting grid trading: {grid_levels} levels around {center_price}")

        try:
            # Place grid orders
            orders = []

            for i in range(grid_levels // 2):
                # Buy orders below center
                buy_price = center_price * (1 - grid_spacing * (i + 1))
                buy_order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side='buy',
                    amount=size,
                    order_type='limit',
                    price=buy_price
                )
                if buy_order:
                    orders.append(buy_order)

                # Sell orders above center
                sell_price = center_price * (1 + grid_spacing * (i + 1))
                sell_order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side='sell',
                    amount=size,
                    order_type='limit',
                    price=sell_price
                )
                if sell_order:
                    orders.append(sell_order)

            self.logger.info(f"Placed {len(orders)} grid orders")

            # Monitor grid
            while True:
                await asyncio.sleep(60)  # Check every minute
                # Grid monitoring logic here

        except asyncio.CancelledError:
            self.logger.info("Grid trading cancelled")
            # Cancel all grid orders
            exchange_instance = self.orchestrator.exchanges[exchange]
            for order in orders:
                await exchange_instance.cancel_order(order['id'], symbol)
        except Exception as e:
            self.logger.error(f"Error in grid trading: {e}")

    async def dollar_cost_averaging(self, exchange: str, symbol: str, amount_per_buy: float,
                                  params: Optional[Dict] = None):
        """
        Dollar Cost Averaging strategy.
        Buys fixed dollar amounts at regular intervals.
        """
        params = params or {}
        interval_hours = params.get('interval_hours', 24)
        total_buys = params.get('total_buys', 30)

        self.logger.info(f"Starting DCA: ${amount_per_buy} every {interval_hours} hours, {total_buys} total")

        try:
            for i in range(total_buys):
                # Execute buy
                order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side='buy',
                    amount=amount_per_buy,  # This should be converted to base currency
                    order_type='market'
                )

                if order:
                    self.logger.info(f"DCA buy {i+1}/{total_buys} executed")
                    self.algo_stats[f"{exchange}_{symbol}_dca"]["total_volume"] += amount_per_buy

                # Wait for next interval
                if i < total_buys - 1:
                    await asyncio.sleep(interval_hours * 3600)

        except asyncio.CancelledError:
            self.logger.info("DCA cancelled")
        except Exception as e:
            self.logger.error(f"Error in DCA: {e}")

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
    # HELPER METHODS
    # ============================================

    async def _execute_scheduled_trade(self, exchange: str, symbol: str, size: float,
                                      params: Dict) -> bool:
        """Execute a scheduled trade. Returns True if successful."""
        try:
            # Get current price
            ticker = await self.orchestrator.get_market_data(exchange, symbol)
            if not ticker:
                return False

            current_price = ticker[-1][4]

            # Calculate order price
            if self.dynamic_pricing:
                bid_price = current_price * 0.99
            else:
                bid_price = params.get('fixed_bid', current_price * 0.99)

            # Place order
            if not self.dry_run:
                order = await self.orchestrator.execute_trade(
                    exchange=exchange,
                    symbol=symbol,
                    side='buy',
                    amount=size,
                    order_type='limit',
                    price=bid_price
                )

                if order:
                    self.successful_orders += 1

                    # Wait and cancel
                    await asyncio.sleep(5)
                    exchange_instance = self.orchestrator.exchanges[exchange]
                    await exchange_instance.cancel_order(order['id'], symbol)
                    return True
                return False
            else:
                self.logger.info(f"DRY RUN: Scheduled trade at {bid_price}")
                return True

        except ccxt.InsufficientFunds as e:
            self.logger.error(f"Insufficient funds in scheduled trade: {e}")
            self.failed_orders += 1
            return False
        except Exception as e:
            self.logger.error(f"Error in scheduled trade: {e}")
            self.failed_orders += 1
            return False

    async def _on_order_filled(self, event: Dict):
        """Handle order filled event."""
        self.successful_orders += 1
        # Update PnL if needed

    async def _on_order_failed(self, event: Dict):
        """Handle order failed event."""
        self.failed_orders += 1

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