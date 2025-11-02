"""
Day 9 VWMA Strategy
===================
Volume Weighted Moving Average strategy with multi-timeframe analysis and caching.

Features:
- VWMA calculation for multiple periods (20, 41, 75)
- Comparison with traditional SMAs
- Data caching for performance
- Rate-limited API calls
- Configuration file support
"""

import asyncio
import time
import logging
from typing import Dict, Optional, List
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class VWMAStrategy(BaseStrategy):
    """
    VWMA (Volume Weighted Moving Average) strategy.

    VWMA gives more weight to periods with higher volume, making it more responsive
    to volume changes than traditional moving averages.
    """

    def __init__(self, name: str = "VWMA_Strategy"):
        """Initialize VWMA strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'size': 1,
            'timeframe': '1d',
            'limit': 100,
            'vwma_periods': [20, 41, 75],
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'max_risk_amount': 1000,
            'cache_time': 300,  # 5 minutes cache
            'rate_limit_interval': 0.1,  # Minimum seconds between API calls
            'auto_trade': False
        }

        # Caching
        self._vwma_data_cache = {}
        self._vwma_cache_time = {}
        self._last_api_call_time = 0

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"VWMA Strategy initialized with periods={self.config['vwma_periods']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute VWMA strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Analyze market using VWMA
            analysis = await self.analyze_vwma_market()

            if analysis and analysis['signal'] in ['BUY', 'SELL']:
                return {
                    'action': analysis['signal'],
                    'symbol': self.config['symbol'],
                    'size': self.config['size'],
                    'confidence': 0.8 if analysis['strength'] == 'Strong' else 0.6,
                    'metadata': {
                        'strategy': 'VWMA',
                        'strength': analysis['strength'],
                        'vwma20': analysis['vwma20'],
                        'vwma41': analysis['vwma41'],
                        'vwma75': analysis['vwma75']
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in VWMA strategy execution: {e}")
            return None

    async def rate_limited_api_call(self, func, *args, **kwargs):
        """
        Rate-limited API call wrapper.

        Ensures minimum interval between API calls to prevent rate limiting.

        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        current_time = time.time()
        time_since_last_call = current_time - self._last_api_call_time

        if time_since_last_call < self.config['rate_limit_interval']:
            sleep_time = self.config['rate_limit_interval'] - time_since_last_call
            await asyncio.sleep(sleep_time)

        result = await func(*args, **kwargs)
        self._last_api_call_time = time.time()
        return result

    async def calculate_vwma(self, exchange: str = None, use_cache: bool = True) -> pd.DataFrame:
        """
        Calculate Volume Weighted Moving Average.

        VWMA gives more weight to periods with higher volume, making it more responsive
        to volume changes than traditional moving averages.

        Args:
            exchange: Exchange name (uses first available if None)
            use_cache: Whether to use cached data

        Returns:
            DataFrame with VWMA calculations and signals
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        limit = self.config['limit']
        periods = self.config['vwma_periods']

        # Check cache
        cache_key = f"{exchange}_{symbol}_{timeframe}_vwma"
        current_time = time.time()

        if use_cache and cache_key in self._vwma_data_cache:
            if current_time - self._vwma_cache_time.get(cache_key, 0) < self.config['cache_time']:
                self.logger.debug(f"Using cached VWMA data for {symbol}")
                return self._vwma_data_cache[cache_key]

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return pd.DataFrame()

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data with rate limiting
            bars = await self.rate_limited_api_call(
                conn.fetch_ohlcv, symbol, timeframe, limit=limit
            )

            if not bars:
                return pd.DataFrame()

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate SMAs for comparison
            df['SMA(20)'] = df['close'].rolling(20).mean()
            df['SMA(41)'] = df['close'].rolling(41).mean()
            df['SMA(75)'] = df['close'].rolling(75).mean()

            # Calculate VWMA for each period
            for n in periods:
                # Calculate volume sum for window
                df[f'sum_vol{n}'] = df['volume'].rolling(min_periods=1, window=n).sum()

                # Calculate volume * close
                df['volXclose'] = df['volume'] * df['close']

                # Calculate volume-weighted sum
                df[f'vXc{n}'] = df['volXclose'].rolling(min_periods=1, window=n).sum()

                # Calculate VWMA
                df[f'VWMA({n})'] = df[f'vXc{n}'] / df[f'sum_vol{n}']

                # Generate signals vs SMAs
                df.loc[df[f'VWMA({n})'] > df['SMA(41)'], f'41sig{n}'] = 'BUY'
                df.loc[df[f'VWMA({n})'] > df['SMA(20)'], f'20sig{n}'] = 'BUY'
                df.loc[df[f'VWMA({n})'] > df['SMA(75)'], f'75sig{n}'] = 'BUY'
                df.loc[df[f'VWMA({n})'] < df['SMA(41)'], f'41sig{n}'] = 'SELL'
                df.loc[df[f'VWMA({n})'] < df['SMA(20)'], f'20sig{n}'] = 'SELL'
                df.loc[df[f'VWMA({n})'] < df['SMA(75)'], f'75sig{n}'] = 'SELL'

            # Cache the data
            self._vwma_data_cache[cache_key] = df
            self._vwma_cache_time[cache_key] = current_time

            self.logger.info(f"VWMA calculation completed for {symbol}")
            return df

        except Exception as e:
            self.logger.error(f"Error calculating VWMA: {e}")
            return pd.DataFrame()

    async def analyze_vwma_market(self, exchange: str = None) -> Optional[Dict]:
        """
        Analyze market using VWMA.

        Provides comprehensive market analysis using VWMA vs SMA comparisons.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Market analysis dictionary with signal and strength
        """
        symbol = self.config['symbol']

        try:
            # Calculate VWMA
            df_vwma = await self.calculate_vwma(exchange)

            if df_vwma.empty:
                return None

            # Get last row for current values
            last_row = df_vwma.iloc[-1]
            vwma20 = last_row.get('VWMA(20)', 0)
            vwma41 = last_row.get('VWMA(41)', 0)
            vwma75 = last_row.get('VWMA(75)', 0)

            # Get current price
            if self.exchange_connections:
                exchange = exchange or list(self.exchange_connections.keys())[0]
                conn = self.exchange_connections[exchange]
                ob = await conn.fetch_order_book(symbol)
                current_bid = ob['bids'][0][0] if ob['bids'] else 0
            else:
                current_bid = 0

            # Determine signal and strength
            if vwma20 > vwma41:
                signal = 'BUY'
                strength = 'Strong' if vwma20 > vwma75 else 'Moderate'
            else:
                signal = 'SELL'
                strength = 'Strong' if vwma20 < vwma75 else 'Moderate'

            self.logger.info(f"VWMA Market analysis - Signal: {signal} ({strength})")
            self.logger.info(f"Current: {current_bid}, VWMA(20): {vwma20}, VWMA(41): {vwma41}")

            return {
                'signal': signal,
                'strength': strength,
                'vwma20': vwma20,
                'vwma41': vwma41,
                'vwma75': vwma75,
                'current_price': current_bid,
                'df_vwma': df_vwma
            }

        except Exception as e:
            self.logger.error(f"Error in VWMA market analysis: {e}")
            return None

    async def cleanup(self):
        """Cleanup strategy resources."""
        # Clear caches
        self._vwma_data_cache.clear()
        self._vwma_cache_time.clear()
        await super().cleanup()
