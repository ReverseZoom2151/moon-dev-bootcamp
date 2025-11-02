"""
Day 11 Breakout Strategy
=========================
Breakout/breakdown detection and trading strategy.

Features:
- Resistance/support level identification
- Breakout above resistance detection
- Breakdown below support detection
- Dynamic entry pricing
- Position management
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class BreakoutStrategy(BaseStrategy):
    """
    Breakout/breakdown trading strategy.

    Identifies when price breaks above resistance (breakout) or below support (breakdown)
    and places trades in the direction of the break.
    """

    def __init__(self, name: str = "Breakout_Strategy"):
        """Initialize Breakout strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'position_size': 1,
            'timeframe': '15m',
            'lookback_bars': 100,
            'breakout_multiplier': 1.001,  # 0.1% above resistance
            'breakdown_multiplier': 0.999,  # 0.1% below support
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'order_params': {'timeInForce': 'GTX'}
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"Breakout Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute breakout strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Detect breakout/breakdown
            breakout, breakdown, breakout_price, breakdown_price = await self.detect_breakout_breakdown()

            if breakout:
                return {
                    'action': 'BUY',
                    'symbol': self.config['symbol'],
                    'size': self.config['position_size'],
                    'price': breakout_price,
                    'confidence': 0.8,
                    'metadata': {
                        'strategy': 'Breakout',
                        'breakout_price': breakout_price,
                        'type': 'resistance_break'
                    }
                }
            elif breakdown:
                return {
                    'action': 'SELL',
                    'symbol': self.config['symbol'],
                    'size': self.config['position_size'],
                    'price': breakdown_price,
                    'confidence': 0.8,
                    'metadata': {
                        'strategy': 'Breakout',
                        'breakdown_price': breakdown_price,
                        'type': 'support_break'
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in Breakout strategy execution: {e}")
            return None

    async def detect_breakout_breakdown(self, exchange: str = None) -> Tuple[bool, bool, float, float]:
        """
        Detect breakout above resistance or breakdown below support.

        Analyzes price action to identify:
        - Breakouts: Current price > resistance (previous high)
        - Breakdowns: Current price < support (previous low)

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (breakout_detected, breakdown_detected, breakout_price, breakdown_price)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        lookback_bars = self.config['lookback_bars']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return False, False, 0, 0

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=lookback_bars)
            if not bars:
                return False, False, 0, 0

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate support and resistance
            resistance = float(df['close'].max())
            support = float(df['close'].min())

            # Get current price
            ticker = await conn.fetch_ticker(symbol)
            current_price = ticker['last']

            self.logger.info(f"Breakout detection for {symbol}: Price={current_price:.2f}, "
                           f"Resistance={resistance:.2f}, Support={support:.2f}")

            buy_breakout = False
            sell_breakdown = False
            breakout_price = 0
            breakdown_price = 0

            if current_price > resistance:
                self.logger.info(f'BREAKOUT DETECTED for {symbol}')
                buy_breakout = True
                breakout_price = resistance * self.config['breakout_multiplier']
            elif current_price < support:
                self.logger.info(f'BREAKDOWN DETECTED for {symbol}')
                sell_breakdown = True
                breakdown_price = support * self.config['breakdown_multiplier']

            return buy_breakout, sell_breakdown, breakout_price, breakdown_price

        except Exception as e:
            self.logger.error(f"Error detecting breakout/breakdown: {e}")
            return False, False, 0, 0

    async def calculate_support_resistance(self, exchange: str = None) -> Tuple[float, float]:
        """
        Calculate current support and resistance levels.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (support, resistance)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        lookback_bars = self.config['lookback_bars']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return 0, 0

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=lookback_bars)
            if not bars:
                return 0, 0

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate levels
            resistance = float(df['close'].max())
            support = float(df['close'].min())

            return support, resistance

        except Exception as e:
            self.logger.error(f"Error calculating support/resistance: {e}")
            return 0, 0
