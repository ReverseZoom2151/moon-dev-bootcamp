"""
Day 7 RSI Strategy
==================
Relative Strength Index (RSI) momentum trading strategy.

Features:
- RSI overbought/oversold detection
- Configurable thresholds (default: 70/30)
- Risk management with PnL targets
- Position monitoring
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) momentum strategy.

    Generates buy signals when RSI < oversold threshold (default 30)
    and sell signals when RSI > overbought threshold (default 70).
    """

    def __init__(self, name: str = "RSI_Strategy"):
        """Initialize RSI strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'size': 1,
            'timeframe': '15m',
            'limit': 100,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'max_risk_amount': 1000
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"RSI Strategy initialized with period={self.config['rsi_period']}, "
                        f"overbought={self.config['rsi_overbought']}, "
                        f"oversold={self.config['rsi_oversold']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute RSI strategy logic.

        Returns:
            Trading signal or None
        """
        try:
            # Calculate RSI and get signal
            df, signal = await self.calculate_rsi()

            if signal and signal != 'NEUTRAL':
                latest_rsi = df['rsi'].iloc[-1] if not df.empty else None

                return {
                    'action': signal,
                    'symbol': self.config['symbol'],
                    'size': self.config['size'],
                    'confidence': self._calculate_confidence(latest_rsi),
                    'metadata': {
                        'strategy': 'RSI',
                        'rsi_period': self.config['rsi_period'],
                        'rsi_value': float(latest_rsi) if latest_rsi else None
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in RSI strategy execution: {e}")
            return None

    async def calculate_rsi(self, exchange: str = None) -> Tuple[pd.DataFrame, str]:
        """
        Calculate RSI and generate trading signal.

        RSI (Relative Strength Index) measures momentum:
        - Above 70: Overbought (potential SELL signal)
        - Below 30: Oversold (potential BUY signal)
        - Between 30-70: Neutral

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (dataframe with RSI, signal)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        limit = self.config['limit']
        rsi_period = self.config['rsi_period']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                self.logger.error("No exchange connections available")
                return pd.DataFrame(), 'NEUTRAL'

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not bars:
                self.logger.warning("No price data received for RSI calculation")
                return pd.DataFrame(), 'NEUTRAL'

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate RSI using pandas
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))

            # Generate signal based on RSI levels
            latest_rsi = df['rsi'].iloc[-1]
            signal = 'NEUTRAL'

            if latest_rsi > self.config['rsi_overbought']:
                signal = 'SELL'
                df.loc[df.index[-1], 'signal'] = 'SELL'
            elif latest_rsi < self.config['rsi_oversold']:
                signal = 'BUY'
                df.loc[df.index[-1], 'signal'] = 'BUY'

            self.logger.info(f"RSI({rsi_period}) for {symbol}: {latest_rsi:.2f}, Signal: {signal}")

            return df, signal

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return pd.DataFrame(), 'NEUTRAL'

    def _calculate_confidence(self, rsi_value: Optional[float]) -> float:
        """
        Calculate signal confidence based on RSI value.

        Args:
            rsi_value: Current RSI value

        Returns:
            Confidence score (0-1)
        """
        if rsi_value is None:
            return 0.5

        # Higher confidence the further from neutral zone (50)
        if rsi_value > self.config['rsi_overbought']:
            # Overbought - confidence increases with higher RSI
            confidence = min(0.5 + (rsi_value - self.config['rsi_overbought']) / 60, 1.0)
        elif rsi_value < self.config['rsi_oversold']:
            # Oversold - confidence increases with lower RSI
            confidence = min(0.5 + (self.config['rsi_oversold'] - rsi_value) / 60, 1.0)
        else:
            # Neutral zone
            confidence = 0.3

        return confidence

    async def on_market_update(self, event: Dict):
        """
        Handle market update event.

        Args:
            event: Market update event
        """
        # Check if we need to update position based on RSI changes
        if event.get('symbol') == self.config['symbol']:
            try:
                df, signal = await self.calculate_rsi()
                if signal != 'NEUTRAL':
                    self.logger.debug(f"Market update: RSI signal changed to {signal}")
            except Exception as e:
                self.logger.error(f"Error handling market update: {e}")
