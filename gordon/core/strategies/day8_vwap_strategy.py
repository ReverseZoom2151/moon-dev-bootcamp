"""
Day 8 VWAP Strategy
===================
Volume Weighted Average Price (VWAP) strategy with combined signals.

Features:
- VWAP calculation and analysis
- Combined signals from SMA, RSI, and VWAP
- Configurable signal threshold
- Interactive trading mode
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime
import pandas as pd

from .base import BaseStrategy


class VWAPStrategy(BaseStrategy):
    """
    VWAP (Volume Weighted Average Price) strategy.

    Combines VWAP, SMA, and RSI signals for robust trading decisions.
    Generates signals based on price position relative to VWAP and confirmation
    from other indicators.
    """

    def __init__(self, name: str = "VWAP_Strategy"):
        """Initialize VWAP strategy."""
        super().__init__(name)

        # Strategy-specific configuration
        self.default_config = {
            'symbol': 'BTCUSDT',
            'size': 1,
            'timeframe': '15m',
            'limit': 100,
            'sma_period': 20,
            'rsi_period': 14,
            'signal_threshold': 2,  # Minimum number of agreeing signals
            'target_profit_percentage': 9,
            'max_loss_percentage': -8,
            'max_risk_amount': 1000,
            'auto_trade': True
        }

    async def initialize(self, config: Dict):
        """Initialize strategy with configuration."""
        await super().initialize(config)

        # Merge with default config
        for key, value in self.default_config.items():
            if key not in self.config:
                self.config[key] = value

        self.logger.info(f"VWAP Strategy initialized with signal_threshold={self.config['signal_threshold']}")

    async def execute(self) -> Optional[Dict]:
        """
        Execute VWAP strategy logic with combined signals.

        Returns:
            Trading signal or None
        """
        try:
            # Get combined signal from multiple indicators
            combined_signal, signals_detail = await self.get_combined_signal()

            if combined_signal and combined_signal != 'NEUTRAL':
                return {
                    'action': combined_signal,
                    'symbol': self.config['symbol'],
                    'size': self.config['size'],
                    'confidence': self._calculate_confidence(signals_detail),
                    'metadata': {
                        'strategy': 'VWAP_Combined',
                        'signals': signals_detail,
                        'threshold': self.config['signal_threshold']
                    }
                }

            return None

        except Exception as e:
            self.logger.error(f"Error in VWAP strategy execution: {e}")
            return None

    async def calculate_vwap(self, exchange: str = None) -> Tuple[pd.DataFrame, str]:
        """
        Calculate VWAP and generate trading signal.

        VWAP (Volume Weighted Average Price) provides insight into:
        - Average price weighted by volume
        - Price above VWAP: Bullish (BUY signal)
        - Price below VWAP: Bearish (SELL signal)

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (dataframe with VWAP, signal)
        """
        symbol = self.config['symbol']
        timeframe = self.config['timeframe']
        limit = self.config['limit']

        try:
            # Get exchange connection
            if not self.exchange_connections:
                return pd.DataFrame(), 'NEUTRAL'

            exchange = exchange or list(self.exchange_connections.keys())[0]
            conn = self.exchange_connections[exchange]

            # Fetch OHLCV data
            bars = await conn.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not bars:
                return pd.DataFrame(), 'NEUTRAL'

            df = pd.DataFrame(bars, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate VWAP
            df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            df['volume_x_typical'] = df['volume'] * df['typical_price']
            df['cum_volume'] = df['volume'].cumsum()
            df['cum_vol_x_price'] = df['volume_x_typical'].cumsum()
            df['vwap'] = df['cum_vol_x_price'] / df['cum_volume']

            # Get current price for signal generation
            ticker = await conn.fetch_ticker(symbol)
            current = ticker['last']
            last_vwap = df['vwap'].iloc[-1]

            # Generate signal
            signal = 'NEUTRAL'
            if current > last_vwap:
                signal = 'BUY'
                df.loc[df.index[-1], 'vwap_signal'] = 'BUY'
            else:
                signal = 'SELL'
                df.loc[df.index[-1], 'vwap_signal'] = 'SELL'

            self.logger.info(f"VWAP for {symbol}: {last_vwap:.2f}, Price: {current:.2f}, Signal: {signal}")

            return df, signal

        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return pd.DataFrame(), 'NEUTRAL'

    async def get_combined_signal(self, exchange: str = None) -> Tuple[str, Dict]:
        """
        Get combined signal from multiple indicators.

        Combines signals from SMA, RSI, and VWAP:
        - 2+ BUY signals = BUY
        - 2+ SELL signals = SELL
        - Otherwise = NEUTRAL

        Args:
            exchange: Exchange name (uses first available if None)

        Returns:
            Tuple of (combined signal, signals detail dict)
        """
        symbol = self.config['symbol']

        try:
            # Import other strategies for their indicator calculations
            from .day6_sma_strategy import SMAStrategy
            from .day7_rsi_strategy import RSIStrategy

            # Create temporary strategy instances for calculations
            sma_strat = SMAStrategy()
            sma_strat.exchange_connections = self.exchange_connections
            sma_strat.config = self.config.copy()

            rsi_strat = RSIStrategy()
            rsi_strat.exchange_connections = self.exchange_connections
            rsi_strat.config = self.config.copy()

            # Get signals from all indicators
            sma_df, sma_signal = await sma_strat.calculate_sma(exchange)
            rsi_df, rsi_signal = await rsi_strat.calculate_rsi(exchange)
            vwap_df, vwap_signal = await self.calculate_vwap(exchange)

            # Handle None/NEUTRAL signals
            sma_signal = sma_signal or 'NEUTRAL'
            rsi_signal = rsi_signal if rsi_signal in ['BUY', 'SELL'] else 'NEUTRAL'
            vwap_signal = vwap_signal if vwap_signal in ['BUY', 'SELL'] else 'NEUTRAL'

            signals_detail = {
                'sma': sma_signal,
                'rsi': rsi_signal,
                'vwap': vwap_signal
            }

            # Count signals
            buy_count = sum(1 for signal in [sma_signal, rsi_signal, vwap_signal] if signal == 'BUY')
            sell_count = sum(1 for signal in [sma_signal, rsi_signal, vwap_signal] if signal == 'SELL')

            # Determine combined signal based on threshold
            threshold = self.config['signal_threshold']
            if buy_count >= threshold:
                combined = 'BUY'
            elif sell_count >= threshold:
                combined = 'SELL'
            else:
                combined = 'NEUTRAL'

            self.logger.info(f"Combined signal for {symbol}: {combined} "
                           f"(SMA: {sma_signal}, RSI: {rsi_signal}, VWAP: {vwap_signal})")

            return combined, signals_detail

        except Exception as e:
            self.logger.error(f"Error getting combined signal: {e}")
            return 'NEUTRAL', {}

    def _calculate_confidence(self, signals_detail: Dict) -> float:
        """
        Calculate signal confidence based on indicator agreement.

        Args:
            signals_detail: Dictionary of individual signals

        Returns:
            Confidence score (0-1)
        """
        if not signals_detail:
            return 0.5

        signals = [signals_detail.get('sma'), signals_detail.get('rsi'), signals_detail.get('vwap')]
        non_neutral = [s for s in signals if s in ['BUY', 'SELL']]

        if not non_neutral:
            return 0.3

        # Count agreement
        buy_count = sum(1 for s in non_neutral if s == 'BUY')
        sell_count = sum(1 for s in non_neutral if s == 'SELL')
        max_agreement = max(buy_count, sell_count)

        # Confidence increases with more agreeing signals
        if max_agreement == 3:
            return 0.95
        elif max_agreement == 2:
            return 0.75
        else:
            return 0.55
