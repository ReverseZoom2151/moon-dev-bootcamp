"""
Day 6 SMA Strategy
==================
Simple Moving Average trading strategy with scheduled execution and risk management.

Features:
- SMA-based trend following
- Position size management
- Scheduled trading cycles
- PnL threshold monitoring
- Risk exposure limits
"""

import asyncio
import logging
from typing import Dict, Optional, Any, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class SMAStrategy(BaseStrategy):
    """
    Simple Moving Average (SMA) strategy implementation.

    Generates buy signals when price is above SMA and sell signals when below.
    Includes comprehensive risk management and position monitoring.
    """

    def __init__(self, name: str = "SMA_Strategy"):
        """Initialize SMA strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'size': 1,
            'timeframe': '15m',
            'limit': 100,
            'sma_period': 20,
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'max_risk_amount': 1000,
            'sleep_time': 30,
            'emergency_timeout': 300,
            'schedule_interval_minutes': 15
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"SMA Strategy initialized with period={self.config['sma_period']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute SMA strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Check risk exposure first
            if not await self.check_risk_exposure():
                self.logger.warning("Risk exposure exceeded, skipping execution")
                return None

            # Check existing positions
            await self.check_pnl_and_manage_position()

            # Calculate SMA and get signal
            df, signal = await self.calculate_sma()

            if signal:
                return {
                    'action': signal,
                    'symbol': self.config['symbol'],
                    'size': self.config['size'],
                    'confidence': 0.7,
                    'metadata': {
                        'strategy': 'SMA',
                        'sma_period': self.config['sma_period'],
                        'sma_value': float(df['sma'].iloc[-1]) if not df.empty else None
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in SMA strategy execution: {e}")
            return None

    async def calculate_sma(self, exchange: str = None) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Calculate SMA and generate trading signal.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (dataframe, signal)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        limit = self.config['limit']
        sma_period = self.config['sma_period']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                self.logger.error("No exchange connections available")
                return pd.DataFrame(), None

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate SMA
            df['sma'] = df['close'].rolling(sma_period).mean()

            # Get current price
            ticker = await conn.fetch_ticker(symbol)
            current = ticker['last']
            last_sma = df['sma'].iloc[-1]

            # Generate signal
            signal = None
            if current > last_sma:
                signal = 'BUY'
            elif current < last_sma:
                signal = 'SELL'

            self.logger.info(f"SMA Signal for {symbol}: {signal} (Price: {current:.2f}, SMA: {last_sma:.2f})")

            return df, signal

        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return pd.DataFrame(), None

    async def check_pnl_and_manage_position(self, exchange: str = None) -> bool:
        """
        Check PnL and manage position based on thresholds.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            True if position was closed, False otherwise
        """
        symbol = self.config['symbol']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return False

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Get position
            positions = await conn.fetch_positions([symbol])
            if not positions or positions[0]['contracts'] == 0:
                return False

            pos = positions[0]
            ticker = await conn.fetch_ticker(symbol)
            current = ticker['last']
            entry = pos['entryPrice']
            leverage = pos.get('leverage', 1)
            is_long = pos['side'] == 'long'

            # Calculate PnL percentage
            diff = (current - entry) if is_long else (entry - current)
            perc = (diff / entry * leverage) * 100

            # Check thresholds
            if perc >= self.config['target_profit_percentage'] or \
               perc <= self.config['max_loss_percentage']:
                await self.close_position(exchange)
                self.logger.info(f"Position closed due to PnL: {perc:.2f}%")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error checking PnL: {e}")
            return False

    async def close_position(self, exchange: str = None) -> bool:
        """
        Close current position.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Success status
        """
        symbol = self.config['symbol']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return False

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Cancel all orders first
            await conn.cancel_all_orders(symbol)

            # Get position
            positions = await conn.fetch_positions([symbol])
            if not positions or positions[0]['contracts'] == 0:
                return False

            pos = positions[0]
            side = 'sell' if pos['side'] == 'long' else 'buy'
            size = pos['contracts']

            await conn.create_market_order(symbol, side, size, {'reduceOnly': True})
            self.logger.info(f"Position closed for {symbol}")
            return True

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False

    async def check_risk_exposure(self, exchange: str = None) -> bool:
        """
        Check risk exposure across all positions.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            True if within limits, False if positions were closed
        """
        try:
            # Get exchange connection
            if not self.exchange_connections:
                return True

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            positions = await conn.fetch_positions()
            total_notional = sum(abs(p.get('notional', 0)) for p in positions if p.get('notional'))

            if total_notional > self.config['max_risk_amount']:
                self.logger.warning(f"Risk exposure ${total_notional:.2f} exceeds limit ${self.config['max_risk_amount']}")

                # Close all positions
                for p in positions:
                    if p.get('notional', 0) != 0:
                        await self.close_position(exchange)

                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking risk exposure: {e}")
            return False
