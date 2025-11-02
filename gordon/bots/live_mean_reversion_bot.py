"""
Live Mean Reversion Trading Bot
================================
Day 20: Mean reversion strategy for ranging markets (binance_mr_bot.py)

This bot implements a mean reversion strategy that:
- Buys when price drops below SMA by a certain percentage
- Sells when price rises above SMA by a certain percentage
- Supports multiple symbols with individual configurations
- Includes optional stop loss and take profit orders
"""

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiveMeanReversionBot:
    """
    Live Mean Reversion Trading Bot (Day 20 - binance_mr_bot.py).

    Implements mean reversion strategy:
    - Monitors price relative to moving average
    - Buys on oversold conditions (price below SMA)
    - Sells on overbought conditions (price above SMA)
    - Manages positions with stop loss and take profit
    """

    # Default symbols configuration (from Day 20 bot)
    DEFAULT_SYMBOLS_DATA = {
        'WIFUSDT': {'sma_period': 14, 'buy_range': (14, 15), 'sell_range': (14, 22)},
        'POPCATUSDT': {'sma_period': 14, 'buy_range': (12, 13), 'sell_range': (14, 18)}
    }

    def __init__(self, exchange_adapter=None, config: Optional[Dict] = None):
        """
        Initialize the mean reversion bot.

        Args:
            exchange_adapter: Exchange connection adapter
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.config = config or self._get_default_config()

        # Trading parameters
        self.order_usd_size = self.config.get('order_usd_size', 10)
        self.leverage = self.config.get('leverage', 3)
        self.timeframe = self.config.get('timeframe', '4h')

        # Symbols configuration
        self.symbols = self.config.get('symbols', ['WIFUSDT', 'POPCATUSDT'])
        self.symbols_data = self.config.get('symbols_data', self.DEFAULT_SYMBOLS_DATA)

        # Position tracking
        self.active_positions = {}

        # Demo mode
        self.demo_mode = self.config.get('demo_mode', False)
        if self.demo_mode:
            logger.info("🔧 Running in DEMO MODE - no real trades will be executed")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'order_usd_size': 10,
            'leverage': 3,
            'timeframe': '4h',
            'demo_mode': False,
            'max_positions': 5,
            'scan_interval_minutes': 1,
            'symbols': ['WIFUSDT', 'POPCATUSDT'],
            'symbols_data': self.DEFAULT_SYMBOLS_DATA
        }

    def calculate_sma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """
        Calculate Simple Moving Average.

        Args:
            data: DataFrame with OHLCV data
            period: SMA period

        Returns:
            SMA series
        """
        return data['close'].rolling(window=period).mean()

    async def fetch_ohlcv(self, symbol: str, limit: int = 20) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of candles to fetch

        Returns:
            DataFrame with OHLCV data
        """
        if self.demo_mode:
            # Generate demo data
            dates = pd.date_range(end=datetime.now(), periods=limit, freq='4H')
            prices = 100 + np.cumsum(np.random.randn(limit) * 2)
            return pd.DataFrame({
                'timestamp': dates,
                'open': prices - np.random.rand(limit) * 0.5,
                'high': prices + np.random.rand(limit) * 1,
                'low': prices - np.random.rand(limit) * 1,
                'close': prices,
                'volume': np.random.rand(limit) * 1000000
            })

        try:
            if not self.exchange_adapter:
                return pd.DataFrame()

            # Fetch OHLCV data
            ohlcv = await self.exchange_adapter.fetch_ohlcv(
                symbol, timeframe=self.timeframe, limit=limit
            )

            if not ohlcv:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            return df

        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return pd.DataFrame()

    def mean_reversion_strategy(self, symbol: str, data: pd.DataFrame,
                               sma_period: int, buy_range: Tuple[float, float],
                               sell_range: Tuple[float, float]) -> Tuple[str, Optional[float], Optional[float], Optional[float]]:
        """
        Calculate mean reversion signals.

        Args:
            symbol: Trading symbol
            data: OHLCV data
            sma_period: SMA period
            buy_range: Buy threshold range (percentage below SMA)
            sell_range: Sell threshold range (percentage above SMA)

        Returns:
            Tuple of (action, buy_threshold, sell_threshold, current_price)
        """
        try:
            # Calculate SMA
            data['SMA'] = self.calculate_sma(data, sma_period)

            if len(data) < sma_period:
                logger.warning(f"Not enough data to calculate SMA for period {sma_period}")
                return "HOLD", None, None, None

            # Get last valid SMA
            last_valid_sma = data['SMA'].dropna().iloc[-1]

            # Calculate thresholds with random variation within ranges
            buy_threshold = last_valid_sma * (1 - np.random.uniform(buy_range[0], buy_range[1]) / 100)
            sell_threshold = last_valid_sma * (1 + np.random.uniform(sell_range[0], sell_range[1]) / 100)

            # Get current price
            current_price = float(data['close'].iloc[-1])
            buy_threshold = float(buy_threshold)
            sell_threshold = float(sell_threshold)

            # Determine action
            if current_price < buy_threshold:
                action = "BUY"
            elif current_price > sell_threshold:
                action = "SELL"
            else:
                action = "HOLD"

            return action, buy_threshold, sell_threshold, current_price

        except Exception as e:
            logger.error(f"Error in mean reversion strategy for {symbol}: {e}")
            return "HOLD", None, None, None

    async def calculate_position_size(self, symbol: str, entry_price: float) -> float:
        """
        Calculate position size based on account balance and leverage.

        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade

        Returns:
            Position size
        """
        if self.demo_mode:
            return self.order_usd_size / entry_price

        try:
            if not self.exchange_adapter:
                return 0

            # Get account balance
            balance = await self.exchange_adapter.fetch_balance()
            available_balance = balance.get('USDT', {}).get('free', 0)

            # Calculate position size
            position_value = min(self.order_usd_size, available_balance * 0.95)
            position_size = (position_value / entry_price) * self.leverage

            # Get symbol precision
            market = await self.exchange_adapter.fetch_market(symbol)
            precision = market.get('precision', {}).get('amount', 8)
            position_size = round(position_size, precision)

            return position_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0

    async def execute_trade(self, symbol: str, action: str, current_price: float,
                          buy_threshold: float, sell_threshold: float) -> bool:
        """
        Execute a trade based on the signal.

        Args:
            symbol: Trading symbol
            action: BUY, SELL, or HOLD
            current_price: Current market price
            buy_threshold: Buy threshold price
            sell_threshold: Sell threshold price

        Returns:
            Success status
        """
        try:
            if action == "BUY":
                logger.info(f"Executing BUY order for {symbol}")

                # Check if already in position
                if symbol in self.active_positions:
                    logger.info(f"Already in position for {symbol}, no new order placed")
                    return False

                # Check max positions
                if len(self.active_positions) >= self.config.get('max_positions', 5):
                    logger.info(f"Max positions reached ({len(self.active_positions)})")
                    return False

                # Calculate position size
                entry_price = buy_threshold
                position_size = await self.calculate_position_size(symbol, entry_price)

                if position_size <= 0:
                    logger.error(f"Invalid position size for {symbol}")
                    return False

                # Calculate stop loss and take profit
                stop_loss = round(entry_price * 0.3, 3)  # 70% stop loss as per Day 20
                take_profit = round(sell_threshold, 3)

                if self.demo_mode:
                    logger.info(f"[DEMO] Would place BUY order for {symbol}:")
                    logger.info(f"  Entry: {entry_price:.4f}")
                    logger.info(f"  Stop Loss: {stop_loss:.4f}")
                    logger.info(f"  Take Profit: {take_profit:.4f}")
                    logger.info(f"  Size: {position_size}")

                    # Track demo position
                    self.active_positions[symbol] = {
                        'entry_price': entry_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'timestamp': datetime.now()
                    }
                    return True

                else:
                    if not self.exchange_adapter:
                        return False

                    # Place limit order
                    order = await self.exchange_adapter.create_limit_buy_order(
                        symbol=symbol,
                        amount=position_size,
                        price=entry_price,
                        params={
                            'stopLoss': {'triggerPrice': stop_loss, 'price': stop_loss},
                            'takeProfit': {'triggerPrice': take_profit, 'price': take_profit}
                        }
                    )

                    if order:
                        logger.info(f"Order placed for {symbol} at {entry_price}")
                        logger.info(f"  Order ID: {order.get('id')}")

                        # Track position
                        self.active_positions[symbol] = {
                            'order_id': order.get('id'),
                            'entry_price': entry_price,
                            'size': position_size,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timestamp': datetime.now()
                        }
                        return True

                    return False

            elif action == "SELL":
                logger.info(f"SELL signal for {symbol} - orders should already be in place via take-profit")
                return False

            else:
                logger.debug(f"No action needed for {symbol}")
                return False

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            return False

    async def run_trading_strategy(self) -> int:
        """
        Run one cycle of the trading strategy for all symbols.

        Returns:
            Number of orders placed
        """
        orders_placed = 0

        for symbol in self.symbols:
            try:
                # Check if symbol has configuration
                if symbol not in self.symbols_data:
                    logger.warning(f"No configuration for {symbol}, skipping")
                    continue

                # Get symbol configuration
                sym_config = self.symbols_data[symbol]
                sma_period = sym_config['sma_period']
                buy_range = sym_config['buy_range']
                sell_range = sym_config['sell_range']

                # Fetch OHLCV data
                df = await self.fetch_ohlcv(symbol, limit=20)

                if df.empty:
                    logger.warning(f"No data available for {symbol}, skipping")
                    continue

                # Calculate signals
                action, buy_threshold, sell_threshold, current_price = self.mean_reversion_strategy(
                    symbol, df, sma_period, buy_range, sell_range
                )

                logger.info(f"{symbol} - Action: {action}, Buy: {buy_threshold:.4f if buy_threshold else 'N/A'}, "
                          f"Sell: {sell_threshold:.4f if sell_threshold else 'N/A'}, "
                          f"Current: {current_price:.4f if current_price else 'N/A'}")

                # Execute trade if needed
                if await self.execute_trade(symbol, action, current_price, buy_threshold, sell_threshold):
                    orders_placed += 1

            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")

        return orders_placed

    async def run_cycle(self):
        """Run one complete bot cycle."""
        try:
            logger.info("=" * 60)
            logger.info("Starting mean reversion bot cycle...")

            # Run trading strategy
            orders = await self.run_trading_strategy()
            logger.info(f"Cycle complete, placed {orders} orders")

            # Display active positions
            if self.active_positions:
                logger.info(f"Active positions: {len(self.active_positions)}")
                for symbol, pos in self.active_positions.items():
                    logger.info(f"  {symbol}: Entry={pos['entry_price']:.4f}, Size={pos['size']}")

            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Error in bot cycle: {e}")

    async def start(self, interval_minutes: int = None):
        """
        Start the mean reversion bot.

        Args:
            interval_minutes: Minutes between scans
        """
        interval = interval_minutes or self.config.get('scan_interval_minutes', 1)

        logger.info(f"Starting Live Mean Reversion Bot (interval: {interval} minutes)")
        logger.info(f"Configuration: {self.config}")
        logger.info(f"Monitoring symbols: {self.symbols}")

        # Run initial cycle
        await self.run_cycle()

        # Continue running on schedule
        while True:
            try:
                await asyncio.sleep(interval * 60)
                await self.run_cycle()

            except KeyboardInterrupt:
                logger.info("Mean reversion bot stopped by user")
                break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    def get_status(self) -> Dict:
        """Get current bot status."""
        return {
            'demo_mode': self.demo_mode,
            'active_positions': len(self.active_positions),
            'monitored_symbols': len(self.symbols),
            'config': self.config,
            'positions': self.active_positions
        }


def main():
    """Main function to run the mean reversion bot standalone."""
    # Initialize bot with demo mode
    config = {
        'demo_mode': True,
        'order_usd_size': 10,
        'leverage': 3,
        'max_positions': 5,
        'scan_interval_minutes': 1,
        'timeframe': '4h',
        'symbols': ['WIFUSDT', 'POPCATUSDT', 'BTCUSDT'],
        'symbols_data': {
            'WIFUSDT': {'sma_period': 14, 'buy_range': (14, 15), 'sell_range': (14, 22)},
            'POPCATUSDT': {'sma_period': 14, 'buy_range': (12, 13), 'sell_range': (14, 18)},
            'BTCUSDT': {'sma_period': 20, 'buy_range': (5, 10), 'sell_range': (10, 15)}
        }
    }

    bot = LiveMeanReversionBot(config=config)

    # Run async event loop
    asyncio.run(bot.start())


if __name__ == "__main__":
    main()