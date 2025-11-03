"""
Enhanced Trading Utilities
===========================
Day 44: Enhanced utility functions for trading operations.
Technical indicators, data processing, and helper functions.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Optional pandas_ta import - gracefully handle if not installed
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    ta = None

logger = logging.getLogger(__name__)


class EnhancedTradingUtils:
    """
    Enhanced trading utility functions.
    
    Provides technical indicators, data processing,
    and helper functions for trading operations.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize enhanced trading utilities.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
    
    def calculate_vwap(
        self,
        df: pd.DataFrame,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close',
        volume_col: str = 'volume'
    ) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            df: DataFrame with OHLCV data
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            volume_col: Column name for volume
            
        Returns:
            Series with VWAP values
        """
        try:
            typical_price = (df[high_col] + df[low_col] + df[close_col]) / 3
            vwap = (typical_price * df[volume_col]).cumsum() / df[volume_col].cumsum()
            return vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {e}")
            return pd.Series()
    
    def calculate_atr(
        self,
        df: pd.DataFrame,
        window: int = 14,
        high_col: str = 'high',
        low_col: str = 'low',
        close_col: str = 'close'
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with OHLCV data
            window: ATR period
            high_col: Column name for high prices
            low_col: Column name for low prices
            close_col: Column name for close prices
            
        Returns:
            Series with ATR values
        """
        try:
            if not PANDAS_TA_AVAILABLE:
                logger.warning("pandas_ta not available. Using fallback ATR calculation.")
                # Fallback ATR calculation
                high_low = df[high_col] - df[low_col]
                high_close = np.abs(df[high_col] - df[close_col].shift())
                low_close = np.abs(df[low_col] - df[close_col].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = tr.rolling(window=window).mean()
                return atr
            atr = ta.atr(df[high_col], df[low_col], df[close_col], length=window)
            return atr
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return pd.Series()
    
    def calculate_bollinger_bands(
        self,
        df: pd.DataFrame,
        length: int = 20,
        std_dev: float = 2.0,
        close_col: str = 'close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series, bool, bool]:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with OHLCV data
            length: Period for moving average
            std_dev: Standard deviation multiplier
            close_col: Column name for close prices
            
        Returns:
            Tuple of (upper_band, middle_band, lower_band, is_tight, is_wide)
        """
        try:
            if len(df) < length:
                return pd.Series(), pd.Series(), pd.Series(), False, False
            
            if not PANDAS_TA_AVAILABLE:
                logger.warning("pandas_ta not available. Using fallback Bollinger Bands calculation.")
                # Fallback Bollinger Bands calculation
                sma = df[close_col].rolling(window=length).mean()
                std = df[close_col].rolling(window=length).std()
                upper = sma + (std * std_dev)
                lower = sma - (std * std_dev)
                return upper, sma, lower, False, False
            
            bbands = ta.bbands(df[close_col], length=length, std=std_dev)
            
            upper = bbands[f'BBU_{length}_{std_dev:.1f}']
            middle = bbands[f'BBM_{length}_{std_dev:.1f}']
            lower = bbands[f'BBL_{length}_{std_dev:.1f}']
            
            # Determine if bands are tight or wide
            current_width = upper.iloc[-1] - lower.iloc[-1]
            avg_width = (upper - lower).rolling(50).mean().iloc[-1]
            
            is_tight = current_width < (avg_width * 0.8) if not pd.isna(avg_width) else False
            is_wide = current_width > (avg_width * 1.2) if not pd.isna(avg_width) else False
            
            return upper, middle, lower, is_tight, is_wide
            
        except Exception as e:
            logger.error(f"Error calculating Bollinger Bands: {e}")
            return pd.Series(), pd.Series(), pd.Series(), False, False
    
    def detect_volume_spike(
        self,
        df: pd.DataFrame,
        multiplier: float = 2.0,
        window: int = 20,
        volume_col: str = 'volume'
    ) -> bool:
        """
        Detect volume spikes.
        
        Args:
            df: DataFrame with volume data
            multiplier: Multiplier for spike detection
            window: Rolling window for average
            volume_col: Column name for volume
            
        Returns:
            True if volume spike detected
        """
        try:
            if len(df) < window:
                return False
            
            current_volume = df[volume_col].iloc[-1]
            avg_volume = df[volume_col].rolling(window).mean().iloc[-1]
            
            return current_volume > (avg_volume * multiplier) if not pd.isna(avg_volume) else False
            
        except Exception as e:
            logger.error(f"Error detecting volume spike: {e}")
            return False
    
    def process_ohlcv_data(
        self,
        data: List,
        time_period: int = 20,
        add_indicators: bool = True
    ) -> pd.DataFrame:
        """
        Process raw OHLCV data into DataFrame with indicators.
        
        Args:
            data: Raw OHLCV data list
            time_period: Period for indicators
            add_indicators: Whether to add technical indicators
            
        Returns:
            Processed DataFrame
        """
        try:
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count', 'taker_buy_volume',
                'taker_buy_quote_volume', 'ignore'
            ])
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Convert price columns to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add technical indicators
            if add_indicators and len(df) >= time_period:
                if PANDAS_TA_AVAILABLE:
                    df['sma'] = ta.sma(df['close'], length=time_period)
                    df['ema'] = ta.ema(df['close'], length=time_period)
                    df['rsi'] = ta.rsi(df['close'], length=14)
                    df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
                else:
                    # Fallback calculations
                    df['sma'] = df['close'].rolling(window=time_period).mean()
                    df['ema'] = df['close'].ewm(span=time_period, adjust=False).mean()
                    # Simple RSI calculation
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    df['rsi'] = 100 - (100 / (1 + rs))
                    # Simple ATR calculation
                    high_low = df['high'] - df['low']
                    high_close = np.abs(df['high'] - df['close'].shift())
                    low_close = np.abs(df['low'] - df['close'].shift())
                    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                    df['atr'] = tr.rolling(window=14).mean()
                
                # Add VWAP
                vwap = self.calculate_vwap(df)
                if not vwap.empty:
                    df['vwap'] = vwap
            
            return df
            
        except Exception as e:
            logger.error(f"Error processing OHLCV data: {e}")
            return pd.DataFrame()
    
    def calculate_sma(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Price series
            window: Moving average window
            
        Returns:
            Series with SMA values
        """
        try:
            if PANDAS_TA_AVAILABLE:
                return ta.sma(prices, length=window)
            else:
                return prices.rolling(window=window).mean()
        except Exception as e:
            logger.error(f"Error calculating SMA: {e}")
            return pd.Series()
    
    def calculate_ema(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Price series
            window: Moving average window
            
        Returns:
            Series with EMA values
        """
        try:
            if PANDAS_TA_AVAILABLE:
                return ta.ema(prices, length=window)
            else:
                return prices.ewm(span=window, adjust=False).mean()
        except Exception as e:
            logger.error(f"Error calculating EMA: {e}")
            return pd.Series()
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        window: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            window: RSI period
            
        Returns:
            Series with RSI values
        """
        try:
            if PANDAS_TA_AVAILABLE:
                return ta.rsi(prices, length=window)
            else:
                # Simple RSI calculation
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.error(f"Error calculating RSI: {e}")
            return pd.Series()
    
    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.
        
        Args:
            prices: Price series
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Tuple of (macd, signal, histogram)
        """
        try:
            macd_data = ta.macd(prices, fast=fast, slow=slow, signal=signal)
            macd = macd_data[f'MACD_{fast}_{slow}_{signal}']
            signal_line = macd_data[f'MACDs_{fast}_{slow}_{signal}']
            histogram = macd_data[f'MACDh_{fast}_{slow}_{signal}']
            return macd, signal_line, histogram
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")
            return pd.Series(), pd.Series(), pd.Series()
    
    def get_support_resistance(
        self,
        df: pd.DataFrame,
        window: int = 20,
        close_col: str = 'close'
    ) -> Tuple[float, float]:
        """
        Calculate support and resistance levels.
        
        Args:
            df: DataFrame with price data
            window: Window for calculating levels
            close_col: Column name for close prices
            
        Returns:
            Tuple of (support, resistance)
        """
        try:
            if len(df) < window:
                return 0.0, 0.0
            
            recent_data = df[close_col].tail(window)
            support = recent_data.min()
            resistance = recent_data.max()
            
            return float(support), float(resistance)
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {e}")
            return 0.0, 0.0
    
    def normalize_data(self, series: pd.Series) -> pd.Series:
        """
        Normalize data series to 0-1 range.
        
        Args:
            series: Data series to normalize
            
        Returns:
            Normalized series
        """
        try:
            min_val = series.min()
            max_val = series.max()
            
            if max_val == min_val:
                return pd.Series([0.5] * len(series), index=series.index)
            
            return (series - min_val) / (max_val - min_val)
            
        except Exception as e:
            logger.error(f"Error normalizing data: {e}")
            return series

