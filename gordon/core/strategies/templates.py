"""
Strategy Templates
==================
Base templates to reduce boilerplate in strategy implementations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import abstractmethod
import pandas as pd
from datetime import datetime

from .base import BaseStrategy
from .indicators import SMA, RSI, VWAP, BollingerBands


class IndicatorBasedStrategy(BaseStrategy):
    """
    Template for strategies based on technical indicators.
    Handles common indicator calculation and signal combination logic.
    """

    def __init__(self, name: str, indicators: List[str], timeframe: str = '15m'):
        """
        Initialize indicator-based strategy.

        Args:
            name: Strategy name
            indicators: List of indicator names to use
            timeframe: Default timeframe for data
        """
        super().__init__(name)
        self.indicators = indicators
        self.timeframe = timeframe
        self.indicator_instances = self._load_indicators()
        self.data_cache = {}

    def _load_indicators(self) -> Dict[str, Any]:
        """Load indicator instances based on names."""
        available_indicators = {
            'sma': SMA,
            'rsi': RSI,
            'vwap': VWAP,
            'bollinger': BollingerBands
        }

        instances = {}
        for indicator_name in self.indicators:
            if indicator_name.lower() in available_indicators:
                instances[indicator_name] = available_indicators[indicator_name.lower()]()
                self.logger.debug(f"Loaded indicator: {indicator_name}")

        return instances

    async def fetch_data(self, exchange: str, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Fetch market data for indicators.

        Args:
            exchange: Exchange name
            symbol: Trading symbol
            limit: Number of candles

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{exchange}:{symbol}:{self.timeframe}:{limit}"

        # Check cache (5-minute TTL)
        if cache_key in self.data_cache:
            cached_time, cached_data = self.data_cache[cache_key]
            if (datetime.now() - cached_time).seconds < 300:
                return cached_data

        # Fetch from exchange
        if exchange not in self.exchange_connections:
            self.logger.error(f"No connection for exchange: {exchange}")
            return pd.DataFrame()

        conn = self.exchange_connections[exchange]

        try:
            ohlcv = await conn.fetch_ohlcv(symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Cache the data
            self.data_cache[cache_key] = (datetime.now(), df)

            return df

        except Exception as e:
            self.logger.error(f"Error fetching data: {e}")
            return pd.DataFrame()

    async def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all configured indicators.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Dictionary of indicator values
        """
        results = {}

        for name, indicator in self.indicator_instances.items():
            try:
                if name == 'sma':
                    df = indicator.calculate_df(df, period=self.config.get('sma_period', 20))
                    results['sma'] = df[f'sma_{self.config.get("sma_period", 20)}'].iloc[-1]

                elif name == 'rsi':
                    df = indicator.calculate_df(df, period=self.config.get('rsi_period', 14))
                    results['rsi'] = df[f'rsi_{self.config.get("rsi_period", 14)}'].iloc[-1]

                elif name == 'vwap':
                    df = indicator.calculate_df(df)
                    results['vwap'] = df['vwap'].iloc[-1]

                elif name == 'bollinger':
                    period = self.config.get('bb_period', 20)
                    std_dev = self.config.get('bb_std', 2.0)
                    upper, middle, lower = indicator.calculate(df['close'], period, std_dev)
                    results['bb_upper'] = upper.iloc[-1]
                    results['bb_middle'] = middle.iloc[-1]
                    results['bb_lower'] = lower.iloc[-1]

            except Exception as e:
                self.logger.error(f"Error calculating {name}: {e}")

        return results

    @abstractmethod
    def combine_signals(self, indicators: Dict[str, Any], price: float) -> Optional[str]:
        """
        Combine indicator values to generate signal.
        Must be implemented by concrete strategies.

        Args:
            indicators: Dictionary of indicator values
            price: Current price

        Returns:
            Signal: 'BUY', 'SELL', or None
        """
        pass

    async def execute(self) -> Optional[Dict]:
        """
        Execute strategy logic using indicators.

        Returns:
            Trading signal or None
        """
        if self.is_paused:
            return None

        # Get default symbol from config
        symbol = self.config.get('symbol', 'BTCUSDT')
        exchange = self.config.get('exchange', 'binance')

        # Fetch data
        df = await self.fetch_data(exchange, symbol)
        if df.empty:
            return None

        # Calculate indicators
        indicators = await self.calculate_indicators(df)

        # Get current price
        current_price = df['close'].iloc[-1]

        # Generate signal
        signal_type = self.combine_signals(indicators, current_price)

        if signal_type:
            return {
                'action': signal_type,
                'symbol': symbol,
                'size': self.config.get('position_size', 0.01),
                'confidence': self.calculate_confidence(indicators),
                'indicators': indicators,
                'metadata': {
                    'strategy': self.name,
                    'timeframe': self.timeframe,
                    'price': current_price
                }
            }

        return None

    def calculate_confidence(self, indicators: Dict[str, Any]) -> float:
        """
        Calculate signal confidence based on indicators.
        Can be overridden by concrete strategies.

        Args:
            indicators: Dictionary of indicator values

        Returns:
            Confidence score (0-1)
        """
        # Default implementation: higher confidence if more indicators agree
        return min(len(indicators) / len(self.indicators), 1.0)


class MultiTimeframeStrategy(BaseStrategy):
    """
    Template for strategies using multiple timeframes.
    Analyzes different timeframes to generate signals.
    """

    def __init__(self, name: str, timeframes: List[str]):
        """
        Initialize multi-timeframe strategy.

        Args:
            name: Strategy name
            timeframes: List of timeframes to analyze
        """
        super().__init__(name)
        self.timeframes = sorted(timeframes, key=self._timeframe_to_minutes)
        self.timeframe_data = {}

    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        multipliers = {'m': 1, 'h': 60, 'd': 1440, 'w': 10080}

        for suffix, multiplier in multipliers.items():
            if timeframe.endswith(suffix):
                return int(timeframe[:-1]) * multiplier

        return 60  # Default to 1 hour

    async def fetch_multi_timeframe_data(self, exchange: str, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all configured timeframes.

        Args:
            exchange: Exchange name
            symbol: Trading symbol

        Returns:
            Dictionary of DataFrames by timeframe
        """
        data = {}

        if exchange not in self.exchange_connections:
            return data

        conn = self.exchange_connections[exchange]

        for timeframe in self.timeframes:
            try:
                limit = self.config.get(f'limit_{timeframe}', 100)
                ohlcv = await conn.fetch_ohlcv(symbol, timeframe, limit=limit)

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                data[timeframe] = df

            except Exception as e:
                self.logger.error(f"Error fetching {timeframe} data: {e}")
                data[timeframe] = pd.DataFrame()

        return data

    @abstractmethod
    def analyze_timeframe(self, timeframe: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a single timeframe.
        Must be implemented by concrete strategies.

        Args:
            timeframe: Timeframe string
            df: DataFrame for this timeframe

        Returns:
            Analysis results
        """
        pass

    @abstractmethod
    def combine_timeframes(self, analyses: Dict[str, Dict]) -> Optional[str]:
        """
        Combine analyses from multiple timeframes.
        Must be implemented by concrete strategies.

        Args:
            analyses: Dictionary of analyses by timeframe

        Returns:
            Signal: 'BUY', 'SELL', or None
        """
        pass

    async def execute(self) -> Optional[Dict]:
        """
        Execute multi-timeframe strategy.

        Returns:
            Trading signal or None
        """
        if self.is_paused:
            return None

        symbol = self.config.get('symbol', 'BTCUSDT')
        exchange = self.config.get('exchange', 'binance')

        # Fetch data for all timeframes
        mtf_data = await self.fetch_multi_timeframe_data(exchange, symbol)

        # Analyze each timeframe
        analyses = {}
        for timeframe, df in mtf_data.items():
            if not df.empty:
                analyses[timeframe] = self.analyze_timeframe(timeframe, df)

        # Combine timeframe analyses
        signal_type = self.combine_timeframes(analyses)

        if signal_type:
            return {
                'action': signal_type,
                'symbol': symbol,
                'size': self.config.get('position_size', 0.01),
                'timeframes': list(analyses.keys()),
                'metadata': {
                    'strategy': self.name,
                    'analyses': analyses
                }
            }

        return None


class MomentumStrategy(IndicatorBasedStrategy):
    """
    Template for momentum-based strategies.
    Uses momentum indicators to generate signals.
    """

    def __init__(self, name: str = "Momentum"):
        """Initialize momentum strategy with RSI and volume."""
        super().__init__(name, indicators=['rsi'], timeframe='15m')
        self.overbought = 70
        self.oversold = 30

    def combine_signals(self, indicators: Dict[str, Any], price: float) -> Optional[str]:
        """
        Generate signal based on momentum indicators.

        Args:
            indicators: Indicator values
            price: Current price

        Returns:
            Signal or None
        """
        rsi = indicators.get('rsi')

        if rsi is None:
            return None

        # Oversold - potential buy
        if rsi < self.oversold:
            return 'BUY'

        # Overbought - potential sell
        if rsi > self.overbought:
            return 'SELL'

        return None


class TrendFollowingStrategy(IndicatorBasedStrategy):
    """
    Template for trend-following strategies.
    Uses moving averages and trend indicators.
    """

    def __init__(self, name: str = "TrendFollowing"):
        """Initialize trend strategy with SMA."""
        super().__init__(name, indicators=['sma'], timeframe='1h')
        self.last_signal = None

    def combine_signals(self, indicators: Dict[str, Any], price: float) -> Optional[str]:
        """
        Generate signal based on trend indicators.

        Args:
            indicators: Indicator values
            price: Current price

        Returns:
            Signal or None
        """
        sma = indicators.get('sma')

        if sma is None:
            return None

        # Price crosses above SMA - buy signal
        if price > sma * 1.01 and self.last_signal != 'BUY':
            self.last_signal = 'BUY'
            return 'BUY'

        # Price crosses below SMA - sell signal
        if price < sma * 0.99 and self.last_signal != 'SELL':
            self.last_signal = 'SELL'
            return 'SELL'

        return None