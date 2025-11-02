"""
Day 12 Engulfing Candle Strategy
=================================
Engulfing candle pattern recognition and trading strategy.

Features:
- Bullish engulfing pattern detection
- Bearish engulfing pattern detection
- SMA confirmation
- Order cooldown periods
- Pattern validation
"""

import logging
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class EngulfingStrategy(BaseStrategy):
    """
    Engulfing candle pattern strategy.

    Detects bullish and bearish engulfing patterns and trades them with
    SMA confirmation for higher probability setups.
    """

    def __init__(self, name: str = "Engulfing_Strategy"):
        """Initialize Engulfing strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'position_size': 1,
            'timeframe': '15m',
            'sma_window': 20,
            'order_cooldown': 300,  # 5 minutes between orders
            'target_profit_percentage': 9,
            'max_loss_percentage': -8
        }

        # Track recent orders
        self.recent_orders = {}

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"Engulfing Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute engulfing pattern strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            symbol = self.config['symbol']

            # Check order cooldown
            if symbol in self.recent_orders:
                elapsed = (datetime.now() - self.recent_orders[symbol]).total_seconds()
                if elapsed < self.config['order_cooldown']:
                    self.logger.debug(f'Order cooldown active: {elapsed:.0f}s elapsed')
                    return None

            # Get market data and detect pattern
            df, pattern = await self._get_market_data_and_pattern()

            if df.empty or not pattern:
                return None

            # Get SMA value for confirmation
            sma_value = df[f'sma_{self.config["sma_window"]}'].iloc[-1]

            # Get current price
            current_price = await self._get_current_price()
            if not current_price:
                return None

            # Trading logic with SMA confirmation
            if pattern == 'bullish' and current_price > sma_value:
                order_price = current_price * 0.999  # Slightly below current

                # Record order time
                self.recent_orders[symbol] = datetime.now()

                return {
                    'action': 'BUY',
                    'symbol': symbol,
                    'size': self.config['position_size'],
                    'price': order_price,
                    'confidence': 0.8,
                    'metadata': {
                        'strategy': 'Engulfing',
                        'pattern': 'bullish',
                        'sma_confirmed': True,
                        'sma_value': sma_value,
                        'entry_price': order_price
                    }
                }

            elif pattern == 'bearish' and current_price < sma_value:
                order_price = current_price * 1.001  # Slightly above current

                # Record order time
                self.recent_orders[symbol] = datetime.now()

                return {
                    'action': 'SELL',
                    'symbol': symbol,
                    'size': self.config['position_size'],
                    'price': order_price,
                    'confidence': 0.8,
                    'metadata': {
                        'strategy': 'Engulfing',
                        'pattern': 'bearish',
                        'sma_confirmed': True,
                        'sma_value': sma_value,
                        'entry_price': order_price
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in Engulfing strategy execution: {e}")
            return None

    def detect_engulfing_pattern(self, df: pd.DataFrame) -> Optional[str]:
        """
        Detect engulfing candle patterns.

        An engulfing pattern occurs when the current candle completely engulfs the previous candle:
        - Bullish engulfing: Current bullish candle engulfs previous bearish candle
        - Bearish engulfing: Current bearish candle engulfs previous bullish candle

        Args:
            df: DataFrame with OHLC data

        Returns:
            'bullish', 'bearish', or None
        """
        if len(df) < 3:
            return None

        # Get the last two complete candles (not the current forming one)
        prev_candle = df.iloc[-3]
        curr_candle = df.iloc[-2]

        # Calculate body sizes
        prev_body_size = abs(prev_candle['close'] - prev_candle['open'])
        curr_body_size = abs(curr_candle['close'] - curr_candle['open'])

        # Determine if candles are bullish or bearish
        prev_bullish = prev_candle['close'] > prev_candle['open']
        curr_bullish = curr_candle['close'] > curr_candle['open']

        # Check for bullish engulfing
        if (not prev_bullish and curr_bullish and
            curr_body_size > prev_body_size and
            curr_candle['open'] <= prev_candle['close'] and
            curr_candle['close'] >= prev_candle['open']):
            self.logger.info("Bullish engulfing pattern detected")
            return 'bullish'

        # Check for bearish engulfing
        elif (prev_bullish and not curr_bullish and
              curr_body_size > prev_body_size and
              curr_candle['open'] >= prev_candle['close'] and
              curr_candle['close'] <= prev_candle['open']):
            self.logger.info("Bearish engulfing pattern detected")
            return 'bearish'

        return None

    async def _get_market_data_and_pattern(self, exchange: str = None) -> tuple:
        """
        Get market data and detect pattern.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (DataFrame, pattern)
        """
        try:
            # Get exchange connection
            if not self.exchange_connections:
                return pd.DataFrame(), None

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(
                self.config['symbol'],
                self.config['timeframe'],
                limit=100
            )

            if not bars:
                return pd.DataFrame(), None

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate SMA
            df[f'sma_{self.config["sma_window"]}'] = df['close'].rolling(
                window=self.config['sma_window']
            ).mean()

            # Detect pattern
            pattern = self.detect_engulfing_pattern(df)

            return df, pattern

        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return pd.DataFrame(), None

    async def _get_current_price(self, exchange: str = None) -> Optional[float]:
        """
        Get current market price.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Current price or None
        """
        try:
            if not self.exchange_connections:
                return None

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            ob = await conn.fetch_order_book(self.config['symbol'])
            bid = ob['bids'][0][0] if ob['bids'] else None
            return bid

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
