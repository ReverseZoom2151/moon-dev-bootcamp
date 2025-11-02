"""
Day 12 VWAP Probability Strategy
=================================
Probability-based trading strategy using VWAP signals.

Features:
- VWAP-based position direction
- Probability-weighted decisions
- Dynamic position sizing
- Account value-based sizing
- Leverage support
"""

import random
import logging
from typing import Dict, Optional
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class VWAPProbabilityStrategy(BaseStrategy):
    """
    VWAP probability-based trading strategy.

    Uses probability distributions to make trading decisions:
    - Above VWAP: 70% chance to go long, 30% short
    - Below VWAP: 30% chance to go long, 70% short
    """

    def __init__(self, name: str = "VWAP_Probability_Strategy"):
        """Initialize VWAP Probability strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'leverage': 1,
            'long_prob_above_vwap': 0.70,  # 70% chance to go long above VWAP
            'long_prob_below_vwap': 0.30,  # 30% chance to go long below VWAP
            'account_allocation': 0.95,  # Use 95% of account
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'timeframe': '15m',
            'limit': 100
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"VWAP Probability Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute VWAP probability strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Calculate position size
            pos_size = await self._calculate_position_size()

            if pos_size <= 0:
                self.logger.warning("Invalid position size, skipping execution")
                return None

            # Get VWAP and current price
            vwap_value, current_price = await self._get_vwap_and_price()

            if not vwap_value or not current_price:
                return None

            # Probability-based decision
            random_chance = random.random()

            if current_price > vwap_value:
                # Price above VWAP
                going_long = random_chance <= self.config['long_prob_above_vwap']
                confidence = self.config['long_prob_above_vwap'] if going_long else (1 - self.config['long_prob_above_vwap'])
                self.logger.info(f'Price above VWAP ({current_price:.2f} > {vwap_value:.2f}), going long: {going_long}')
            else:
                # Price below VWAP
                going_long = random_chance <= self.config['long_prob_below_vwap']
                confidence = self.config['long_prob_below_vwap'] if going_long else (1 - self.config['long_prob_below_vwap'])
                self.logger.info(f'Price below VWAP ({current_price:.2f} < {vwap_value:.2f}), going long: {going_long}')

            action = 'BUY' if going_long else 'SELL'
            entry_price = current_price * 0.999 if going_long else current_price * 1.001

            return {
                'action': action,
                'symbol': self.config['symbol'],
                'size': pos_size,
                'price': entry_price,
                'confidence': confidence,
                'metadata': {
                    'strategy': 'VWAP_Probability',
                    'vwap': vwap_value,
                    'current_price': current_price,
                    'probability': confidence,
                    'above_vwap': current_price > vwap_value
                }
            }

        except Exception as e:
            self.logger.error(f"Error in VWAP Probability strategy execution: {e}")
            return None

    async def _calculate_position_size(self, exchange: str = None) -> float:
        """
        Calculate position size based on account value and leverage.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Calculated position size
        """
        try:
            # Get exchange connection
            if not self.exchange_connections:
                return 0

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Get account value
            balance = await conn.fetch_balance()
            acct_value = balance['total'].get('USDT', 0)

            if acct_value <= 0:
                self.logger.warning("No account value available")
                return 0

            # Get current price
            ticker = await conn.fetch_ticker(self.config['symbol'])
            price = ticker['last']

            if price <= 0:
                return 0

            # Calculate position size based on leverage
            pos_size = (acct_value * self.config['account_allocation'] / price) * self.config['leverage']

            # Round to appropriate decimals
            try:
                market = await conn.fetch_market(self.config['symbol'])
                precision = market.get('precision', {}).get('amount', 8)
                pos_size = round(pos_size, precision)
            except Exception:
                # Default to 3 decimals if market info not available
                pos_size = round(pos_size, 3)

            self.logger.info(f"Calculated position size: {pos_size} (acct: ${acct_value:.2f}, price: ${price:.2f})")

            return max(pos_size, 0.001)  # Ensure minimum size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0

    async def _get_vwap_and_price(self, exchange: str = None) -> tuple:
        """
        Get VWAP value and current price.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (vwap_value, current_price)
        """
        try:
            # Get exchange connection
            if not self.exchange_connections:
                return None, None

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(
                self.config['symbol'],
                self.config['timeframe'],
                limit=self.config['limit']
            )

            if not bars:
                return None, None

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['volume_x_typical'] = df['volume'] * df['typical_price']
            df['cum_volume'] = df['volume'].cumsum()
            df['cum_vol_x_price'] = df['volume_x_typical'].cumsum()
            df['vwap'] = df['cum_vol_x_price'] / df['cum_volume']

            latest_vwap = float(df['vwap'].iloc[-1])

            # Get current price from order book
            ob = await conn.fetch_order_book(self.config['symbol'])
            bid = ob['bids'][0][0] if ob['bids'] else None

            if not bid:
                return None, None

            return latest_vwap, bid

        except Exception as e:
            self.logger.error(f"Error getting VWAP and price: {e}")
            return None, None

    async def on_market_update(self, event: Dict):
        """
        Handle market update event.

        Args:
            event: Market update event
        """
        # Recalculate VWAP-based probabilities on market updates
        if event.get('symbol') == self.config['symbol']:
            try:
                vwap_value, current_price = await self._get_vwap_and_price()
                if vwap_value and current_price:
                    position = "above" if current_price > vwap_value else "below"
                    self.logger.debug(f"Market update: Price is {position} VWAP")
            except Exception as e:
                self.logger.error(f"Error handling market update: {e}")
