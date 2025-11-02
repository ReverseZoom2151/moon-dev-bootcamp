"""
Day 11 Supply/Demand Zones Strategy
====================================
Supply and demand zone identification and trading strategy.

Features:
- Multi-timeframe zone calculation
- Demand zones (support areas)
- Supply zones (resistance areas)
- Caching with retry logic
- Mock data fallback
"""

import asyncio
import time
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

from .base import BaseStrategy


class SupplyDemandStrategy(BaseStrategy):
    """
    Supply/demand zones trading strategy.

    Identifies key price levels where supply and demand are concentrated
    and places trades at these strategic zones.
    """

    def __init__(self, name: str = "SupplyDemand_Strategy"):
        """Initialize Supply/Demand strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'position_size': 1,
            'timeframe': '4h',
            'lookback_days': 30,
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'cache_ttl': 3600,  # 1 hour cache
            'max_retries': 3,
            'retry_delay': 2,
            'enable_mock_fallback': True
        }

        # Caching
        self.data_cache = {}
        self.cache_timestamps = {}

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"Supply/Demand Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute supply/demand strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Calculate supply/demand zones
            sd_df = await self.calculate_supply_demand_zones()

            if sd_df.empty:
                return None

            # Get current market price
            current_price = await self._get_current_price()
            if not current_price:
                return None

            # Calculate average zone levels
            timeframe = self.config['timeframe']
            buy_zone = float(sd_df[f'{timeframe}_dz'].mean())
            sell_zone = float(sd_df[f'{timeframe}_sz'].mean())

            # Determine which zone is closer
            diff_to_buy = abs(current_price - buy_zone)
            diff_to_sell = abs(current_price - sell_zone)

            if diff_to_buy < diff_to_sell:
                return {
                    'action': 'BUY',
                    'symbol': self.config['symbol'],
                    'size': self.config['position_size'],
                    'price': buy_zone,
                    'confidence': 0.75,
                    'metadata': {
                        'strategy': 'SupplyDemand',
                        'zone_type': 'demand',
                        'zone_price': buy_zone,
                        'current_price': current_price,
                        'distance': diff_to_buy
                    }
                }
            else:
                return {
                    'action': 'SELL',
                    'symbol': self.config['symbol'],
                    'size': self.config['position_size'],
                    'price': sell_zone,
                    'confidence': 0.75,
                    'metadata': {
                        'strategy': 'SupplyDemand',
                        'zone_type': 'supply',
                        'zone_price': sell_zone,
                        'current_price': current_price,
                        'distance': diff_to_sell
                    }
                }

        except Exception as e:
            self.logger.error(f"Error in Supply/Demand strategy execution: {e}")
            return None

    async def calculate_supply_demand_zones(self, exchange: str = None) -> pd.DataFrame:
        """
        Calculate supply and demand zones.

        Identifies key price levels:
        - Demand zones: Areas of support where buyers step in
        - Supply zones: Areas of resistance where sellers emerge

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            DataFrame with supply and demand zones
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        lookback_days = self.config['lookback_days']

        # Check cache first
        cache_key = f"sdz_{symbol}_{timeframe}_{lookback_days}"
        if cache_key in self.data_cache:
            cache_age = time.time() - self.cache_timestamps.get(cache_key, 0)
            if cache_age < self.config['cache_ttl']:
                self.logger.debug(f"Using cached supply/demand zones for {symbol}")
                return self.data_cache[cache_key]

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return self._create_mock_zones(timeframe)

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Calculate since timestamp
            since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1000)

            # Fetch OHLCV data with retry logic
            bars = None
            for retry in range(self.config['max_retries']):
                try:
                    bars = await conn.fetch_ohlcv(symbol, timeframe, since=since, limit=500)
                    if bars:
                        break
                except Exception as e:
                    self.logger.warning(f"Retry {retry + 1}/{self.config['max_retries']}: {e}")
                    await asyncio.sleep(self.config['retry_delay'])

            if not bars:
                self.logger.warning(f"Failed to fetch data for {symbol}, using mock data")
                if self.config['enable_mock_fallback']:
                    return self._create_mock_zones(timeframe)
                return pd.DataFrame()

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate zones
            if len(df) > 2:
                support = df[:-2]['close'].min()
                resistance = df[:-2]['close'].max()
                support_low = df[:-2]['low'].min()
                resistance_high = df[:-2]['high'].max()
            else:
                support = df['close'].min()
                resistance = df['close'].max()
                support_low = df['low'].min()
                resistance_high = df['high'].max()

            sd_df = pd.DataFrame({
                f'{timeframe}_dz': [support_low, support],  # Demand zones
                f'{timeframe}_sz': [resistance_high, resistance]  # Supply zones
            })

            # Cache the result
            self.data_cache[cache_key] = sd_df
            self.cache_timestamps[cache_key] = time.time()

            self.logger.info(f'Supply/Demand zones for {symbol}:\n{sd_df}')
            return sd_df

        except Exception as e:
            self.logger.error(f"Error calculating supply/demand zones: {e}")
            if self.config['enable_mock_fallback']:
                return self._create_mock_zones(timeframe)
            return pd.DataFrame()

    def _create_mock_zones(self, timeframe: str) -> pd.DataFrame:
        """
        Create mock supply/demand zones for fallback.

        Args:
            timeframe: Timeframe string

        Returns:
            DataFrame with mock zones
        """
        # Use reasonable mock values
        return pd.DataFrame({
            f'{timeframe}_dz': [95000, 96000],  # Demand zones
            f'{timeframe}_sz': [104000, 105000]  # Supply zones
        })

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

            ticker = await conn.fetch_ticker(self.config['symbol'])
            return ticker['last']

        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None

    async def cleanup(self):
        """Cleanup strategy resources."""
        # Clear caches
        self.data_cache.clear()
        self.cache_timestamps.clear()
        await super().cleanup()
