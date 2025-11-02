"""
Day 10 Volume Analysis Strategy
=================================
Volume-based trading strategy with circuit breakers and dynamic position sizing.

Features:
- Order book volume analysis
- Circuit breakers for volatility and spread
- Dynamic position sizing based on market conditions
- Post-trade cooldown periods
- Multi-timeframe SMA confirmation
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class VolumeStrategy(BaseStrategy):
    """
    Volume analysis strategy.

    Uses order book volume dynamics to determine market control (bulls vs bears)
    and places trades with comprehensive risk management.
    """

    def __init__(self, name: str = "Volume_Strategy"):
        """Initialize Volume strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'position_size': 10,
            'vol_decimal': 0.35,  # Volume control threshold
            'circuit_breaker_volatility': 5.0,  # Max volatility %
            'circuit_breaker_spread': 0.5,  # Max spread %
            'sleep_after_trade': 5,  # Minutes to wait after trade
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'order_params': {'timeInForce': 'GTX'}
        }

        # Rate limiting
        self.retry_count = {}
        self.max_retries = 3

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"Volume Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute volume analysis strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Check circuit breakers first
            if not await self.check_circuit_breakers():
                self.logger.warning("Circuit breakers active - skipping execution")
                return None

            # Analyze order book volume
            vol_under_threshold, control_decimal = await self.analyze_order_book_volume()

            # Get market direction from daily SMA
            signal = await self._get_market_direction()

            if signal and signal != 'NEUTRAL':
                # Calculate dynamic position size
                pos_size = await self.calculate_dynamic_position_size()

                return {
                    'action': signal,
                    'symbol': self.config['symbol'],
                    'size': pos_size,
                    'confidence': self._calculate_confidence(control_decimal),
                    'metadata': {
                        'strategy': 'Volume_Analysis',
                        'control_decimal': control_decimal,
                        'vol_under_threshold': vol_under_threshold,
                        'dynamic_size': pos_size
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in Volume strategy execution: {e}")
            return None

    async def analyze_order_book_volume(self, exchange: str = None) -> Tuple[bool, float]:
        """
        Analyze order book volume dynamics.

        Samples order book multiple times to determine if bulls or bears are in control.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (vol_under_decimal, control_decimal)
        """
        symbol = self.config['symbol']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return False, 0.5

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Sample order book multiple times (11 samples over 1 minute)
            bid_vols = []
            ask_vols = []

            for x in range(11):
                ob = await conn.fetch_order_book(symbol)
                bid_vol = sum(vol for _, vol in ob['bids'])
                ask_vol = sum(vol for _, vol in ob['asks'])
                bid_vols.append(bid_vol)
                ask_vols.append(ask_vol)
                self.logger.debug(f"Order book sample {x}: bid_vol={bid_vol}, ask_vol={ask_vol}")
                await asyncio.sleep(5)

            total_bidvol = sum(bid_vols)
            total_askvol = sum(ask_vols)

            self.logger.info(f'Last minute volume summary - Bid Vol: {total_bidvol} | Ask Vol: {total_askvol}')

            # Determine market control
            if total_bidvol > total_askvol:
                control_dec = total_askvol / total_bidvol
                self.logger.info(f'Bulls are in control: {control_dec}')
                bullish = True
            else:
                control_dec = total_bidvol / total_askvol
                self.logger.info(f'Bears are in control: {control_dec}')
                bullish = False

            # Check if volume control is strong enough
            vol_under_dec = control_dec < self.config['vol_decimal']

            return vol_under_dec, control_dec

        except Exception as e:
            self.logger.error(f"Error analyzing order book volume: {e}")
            return False, 0.5

    async def check_circuit_breakers(self, exchange: str = None) -> bool:
        """
        Check circuit breakers for safe trading conditions.

        Checks:
        - Recent volatility
        - Bid-ask spread

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            True if safe to trade, False if circuit breaker triggered
        """
        symbol = self.config['symbol']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return False

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Get 15m data for volatility check
            bars = await conn.fetch_ohlcv(symbol, '15m', limit=100)
            if not bars:
                return False

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Check recent volatility
            recent_volatility = df['close'].pct_change().rolling(5).std().iloc[-1] * 100

            if recent_volatility > self.config['circuit_breaker_volatility']:
                self.logger.warning(f"Circuit breaker triggered - high volatility: {recent_volatility:.2f}%")
                return False

            # Check spread
            ob = await conn.fetch_order_book(symbol)
            if ob['asks'] and ob['bids']:
                ask = ob['asks'][0][0]
                bid = ob['bids'][0][0]
                spread_pct = (ask - bid) / bid * 100 if bid != 0 else 0

                if spread_pct > self.config['circuit_breaker_spread']:
                    self.logger.warning(f"Circuit breaker triggered - high spread: {spread_pct:.2f}%")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error checking circuit breakers: {e}")
            return False

    async def calculate_dynamic_position_size(self, exchange: str = None) -> int:
        """
        Calculate position size based on volatility.

        Lower position size in high volatility conditions.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Adjusted position size
        """
        symbol = self.config['symbol']
        base_size = self.config['position_size']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return base_size

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Get 15m data
            bars = await conn.fetch_ohlcv(symbol, '15m', limit=100)
            if not bars:
                return base_size

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate volatility
            volatility = df['close'].pct_change().std() * 100

            # Adjust size based on volatility (lower size in high volatility)
            adjusted_size = base_size * (1 / (1 + volatility))

            return max(int(adjusted_size), 1)

        except Exception as e:
            self.logger.error(f"Error calculating dynamic position size: {e}")
            return base_size

    async def sleep_on_close(self, exchange: str = None):
        """
        Sleep after recent trade to avoid overtrading.

        Args:
            exchange: Exchange name (uses first available if None)
        """
        symbol = self.config['symbol']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            closed_orders = await conn.fetch_closed_orders(symbol)

            for ord in reversed(closed_orders):
                if ord['status'] == 'closed' and ord['filled'] > 0:
                    txttime = ord['timestamp'] // 1000
                    ex_timestamp = int(time.time())
                    time_spread = (ex_timestamp - txttime) / 60

                    if time_spread < self.config['sleep_after_trade']:
                        sleepy = round(self.config['sleep_after_trade'] - time_spread) * 60
                        self.logger.info(f'Time since last trade ({time_spread:.1f} mins) is less than {self.config["sleep_after_trade"]} mins, sleeping for {sleepy} secs')
                        await asyncio.sleep(sleepy)
                    else:
                        self.logger.info(f'It has been {time_spread:.1f} mins since last fill, no sleep needed')
                    break

        except Exception as e:
            self.logger.error(f"Error in sleep_on_close: {e}")

    async def _get_market_direction(self, exchange: str = None) -> str:
        """
        Get market direction from daily SMA.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            'BUY', 'SELL', or 'NEUTRAL'
        """
        try:
            # Import SMA strategy for calculation
            from .day6_sma_strategy import SMAStrategy

            sma_strat = SMAStrategy()
            sma_strat.exchange_connections = self.exchange_connections
            sma_strat.config = {
                'symbol': self.config['symbol'],
                'timeframe': '1d',
                'limit': 100,
                'sma_period': 20
            }

            df, signal = await sma_strat.calculate_sma(exchange)
            return signal or 'NEUTRAL'

        except Exception as e:
            self.logger.error(f"Error getting market direction: {e}")
            return 'NEUTRAL'

    def _calculate_confidence(self, control_decimal: float) -> float:
        """
        Calculate confidence based on volume control strength.

        Args:
            control_decimal: Volume control ratio

        Returns:
            Confidence score (0-1)
        """
        # Lower control_decimal means stronger control
        if control_decimal < 0.2:
            return 0.9
        elif control_decimal < 0.35:
            return 0.75
        elif control_decimal < 0.5:
            return 0.6
        else:
            return 0.4
