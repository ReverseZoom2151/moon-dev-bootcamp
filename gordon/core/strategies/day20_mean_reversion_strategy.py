"""
Day 20 Mean Reversion Strategy
================================
SMA-based mean reversion trading strategy for ranging markets.

Features:
- Buy when price drops below SMA by buy_pct percentage
- Sell when price rises above SMA by sell_pct percentage
- Symbol-specific configuration support
- Optional stop loss and take profit levels
- Position management and risk checks
"""

import asyncio
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np

from .base import BaseStrategy
from .indicators.sma import SMA


class MeanReversionStrategy(BaseStrategy):
    """
    SMA-based Mean Reversion Strategy (Day 20).

    Trading logic:
    - Buy when price drops below SMA by buy_pct percentage
    - Sell when price rises above SMA by sell_pct percentage
    - Designed for ranging markets with reversion to mean behavior
    - Supports symbol-specific configurations
    """

    def __init__(self, name: str = "MeanReversion_Strategy"):
        """Initialize Mean Reversion strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'position_size': 1,
            'timeframe': '4h',
            'limit': 20,
            'sma_period': 14,
            'buy_pct': 10.0,  # Buy when price is X% below SMA
            'sell_pct': 15.0,  # Sell when price is X% above SMA
            'stop_loss': 0.0,  # Optional stop loss percentage (0 = disabled)
            'take_profit': 0.0,  # Optional take profit percentage (0 = disabled)
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'max_risk_amount': 1000,
            'symbols_data': {}  # Symbol-specific configs: {'SYMBOL': {'sma_period': 14, 'buy_pct': 10, 'sell_pct': 15}}
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(
            f"Mean Reversion Strategy initialized for {self.config['symbol']} "
            f"(SMA: {self.config['sma_period']}, Buy: {self.config['buy_pct']}%, "
            f"Sell: {self.config['sell_pct']}%)"
        )

    async def execute(self) -> Optional[Dict]:
        """
        Execute mean reversion strategy logic.

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

            # Calculate mean reversion signal
            signal_data = await self.calculate_mean_reversion_signal()

            if signal_data and signal_data.get('action') != 'HOLD':
                return {
                    'action': signal_data['action'],
                    'symbol': self.config['symbol'],
                    'size': self.config['position_size'],
                    'price': signal_data.get('entry_price'),
                    'confidence': signal_data.get('confidence', 0.7),
                    'metadata': {
                        'strategy': 'MeanReversion',
                        'sma_period': signal_data.get('sma_period'),
                        'sma_value': signal_data.get('sma_value'),
                        'current_price': signal_data.get('current_price'),
                        'buy_threshold': signal_data.get('buy_threshold'),
                        'sell_threshold': signal_data.get('sell_threshold'),
                        'stop_loss': signal_data.get('stop_loss'),
                        'take_profit': signal_data.get('take_profit')
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in Mean Reversion strategy execution: {e}")
            return None

    async def calculate_mean_reversion_signal(self, exchange: str = None) -> Optional[Dict]:
        """
        Calculate mean reversion signal based on SMA and price deviation.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Signal dictionary with action and thresholds
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        limit = self.config['limit']

        # Get symbol-specific config if available
        symbols_data = self.config.get('symbols_data', {})
        if symbol in symbols_data:
            sym_config = symbols_data[symbol]
            sma_period = sym_config.get('sma_period', self.config['sma_period'])
            buy_pct = sym_config.get('buy_pct', self.config['buy_pct'])
            sell_pct = sym_config.get('sell_pct', self.config['sell_pct'])
        else:
            sma_period = self.config['sma_period']
            buy_pct = self.config['buy_pct']
            sell_pct = self.config['sell_pct']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                self.logger.error("No exchange connections available")
                return None

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not bars:
                self.logger.error(f"No data available for {symbol}")
                return None

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Check if we have enough data
            if len(df) < sma_period:
                self.logger.warning(f"Not enough data to calculate SMA for period {sma_period}")
                return None

            # Calculate SMA
            df['sma'] = SMA.calculate(df['close'], sma_period)

            # Get current price and SMA
            current_price = float(df['close'].iloc[-1])
            last_sma = df['sma'].dropna().iloc[-1]

            if np.isnan(last_sma) or not np.isfinite(last_sma):
                self.logger.warning("SMA value is invalid")
                return None

            # Calculate thresholds
            buy_threshold = last_sma * (1 - buy_pct / 100)
            sell_threshold = last_sma * (1 + sell_pct / 100)

            # Determine action
            action = 'HOLD'
            entry_price = None
            stop_loss = None
            take_profit = None

            if current_price < buy_threshold:
                action = 'BUY'
                entry_price = current_price
                
                # Calculate stop loss if enabled
                if self.config['stop_loss'] > 0:
                    stop_loss = entry_price * (1 - self.config['stop_loss'] / 100)
                
                # Calculate take profit if enabled
                if self.config['take_profit'] > 0:
                    take_profit = entry_price * (1 + self.config['take_profit'] / 100)
                elif sell_threshold > entry_price:
                    take_profit = sell_threshold

            elif current_price > sell_threshold:
                action = 'SELL'
                entry_price = current_price

            # Calculate confidence based on deviation
            if action == 'BUY':
                deviation = ((buy_threshold - current_price) / buy_threshold) * 100
                confidence = min(0.95, 0.6 + (deviation / buy_pct) * 0.3)
            elif action == 'SELL':
                deviation = ((current_price - sell_threshold) / sell_threshold) * 100
                confidence = min(0.95, 0.6 + (deviation / sell_pct) * 0.3)
            else:
                confidence = 0.5

            self.logger.info(
                f"Mean Reversion Signal for {symbol}: {action} "
                f"(Price: {current_price:.4f}, SMA: {last_sma:.4f}, "
                f"Buy Threshold: {buy_threshold:.4f}, Sell Threshold: {sell_threshold:.4f})"
            )

            return {
                'action': action,
                'current_price': current_price,
                'sma_value': last_sma,
                'sma_period': sma_period,
                'buy_threshold': buy_threshold,
                'sell_threshold': sell_threshold,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence
            }

        except Exception as e:
            self.logger.error(f"Error calculating mean reversion signal: {e}")
            return None

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
                self.logger.warning(
                    f"Risk exposure ${total_notional:.2f} exceeds limit ${self.config['max_risk_amount']}"
                )

                # Close all positions
                for p in positions:
                    if p.get('notional', 0) != 0:
                        await self.close_position(exchange)

                return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking risk exposure: {e}")
            return False

    async def cleanup(self):
        """Cleanup strategy resources."""
        await super().cleanup()

