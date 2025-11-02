"""
Day 10 Bollinger Band Strategy
================================
Bollinger Band compression strategy for low volatility breakouts.

Features:
- BB compression detection
- Volatility-based entry signals
- Order placement at optimal levels
- Risk management
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Band compression strategy.

    Identifies low volatility periods (band compression) and places orders
    anticipating breakout moves.
    """

    def __init__(self, name: str = "Bollinger_Strategy"):
        """Initialize Bollinger strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'WIFUSDT',
            'size': 10,
            'timeframe': '1m',
            'sma_window': 20,
            'compression_threshold': 0.02,  # Bandwidth threshold for compression
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

        self.logger.info(f"Bollinger Strategy initialized for {self.config['symbol']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute Bollinger Band strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Check for BB compression
            df, bands_tight = await self.check_bollinger_compression()

            if bands_tight:
                return {
                    'action': 'PLACE_ORDERS',  # Special action for both sides
                    'symbol': self.config['symbol'],
                    'size': self.config['size'],
                    'confidence': 0.7,
                    'metadata': {
                        'strategy': 'Bollinger_Compression',
                        'bandwidth': self._calculate_bandwidth(df) if not df.empty else None,
                        'compression_detected': True
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in Bollinger strategy execution: {e}")
            return None

    async def check_bollinger_compression(self, exchange: str = None) -> Tuple[pd.DataFrame, bool]:
        """
        Check for Bollinger Band compression.

        BB compression indicates low volatility and potential breakout setup.

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (DataFrame with BB, bands_tight boolean)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        window = self.config['sma_window']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return pd.DataFrame(), False

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Use BTC as proxy for market conditions if not BTC itself
            proxy_symbol = 'BTCUSDT' if symbol != 'BTCUSDT' else symbol
            bars = await conn.fetch_ohlcv(proxy_symbol, '1m', limit=500)

            if not bars:
                return pd.DataFrame(), False

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Calculate Bollinger Bands
            sma = df['close'].rolling(window).mean()
            std = df['close'].rolling(window).std()
            df[f'BBU_{window}_2.0'] = sma + (std * 2)
            df[f'BBM_{window}_2.0'] = sma
            df[f'BBL_{window}_2.0'] = sma - (std * 2)

            # Calculate bandwidth (compression metric)
            bandwidth = (df[f'BBU_{window}_2.0'] - df[f'BBL_{window}_2.0']) / df[f'BBM_{window}_2.0']
            bands_tight = bandwidth.iloc[-1] < self.config['compression_threshold']

            self.logger.info(f'Bollinger bands compression detected: {bands_tight} (bandwidth: {bandwidth.iloc[-1]:.4f})')

            return df, bands_tight

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.DataFrame(), False

    async def place_entry_orders(self, exchange: str = None):
        """
        Place entry orders based on Bollinger Band strategy.

        Places limit orders at 10th level of order book on both sides.

        Args:
            exchange: Exchange name (uses first available if None)
        """
        symbol = self.config['symbol']
        pos_size = self.config['size']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Get order book
            ob = await conn.fetch_order_book(symbol)
            bid = ob['bids'][0][0] if ob['bids'] else 0
            ask = ob['asks'][0][0] if ob['asks'] else 0

            # Use 10th level for entry orders
            bid10 = ob['bids'][10][0] if len(ob['bids']) > 10 else bid
            ask10 = ob['asks'][10][0] if len(ob['asks']) > 10 else ask

            # Cancel existing orders
            await conn.cancel_all_orders(symbol)
            self.logger.info('Cancelled existing orders')

            # Place entry orders
            params = self.config['order_params']
            await conn.create_limit_buy_order(symbol, pos_size, bid10, params)
            self.logger.info(f'Placed buy order for {pos_size} at {bid10}')

            await conn.create_limit_sell_order(symbol, pos_size, ask10, params)
            self.logger.info(f'Placed sell order for {pos_size} at {ask10}')

        except Exception as e:
            self.logger.error(f"Error placing Bollinger entry orders: {e}")

    def _calculate_bandwidth(self, df: pd.DataFrame) -> Optional[float]:
        """
        Calculate current Bollinger Band bandwidth.

        Args:
            df: DataFrame with BB calculations

        Returns:
            Current bandwidth or None
        """
        try:
            window = self.config['sma_window']
            if f'BBU_{window}_2.0' in df.columns and f'BBL_{window}_2.0' in df.columns:
                bandwidth = (df[f'BBU_{window}_2.0'] - df[f'BBL_{window}_2.0']) / df[f'BBM_{window}_2.0']
                return float(bandwidth.iloc[-1])
        except Exception:
            pass
        return None
