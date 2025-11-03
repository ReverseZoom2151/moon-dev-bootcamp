"""
RRS Data Processor
==================
Processes OHLCV data for RRS analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RRSDataProcessor:
    """
    Processes OHLCV data for RRS analysis.
    
    Calculates returns, volatility, volume metrics, and technical indicators.
    """
    
    def __init__(
        self,
        volatility_lookback: int = 20,
        volume_lookback: int = 10
    ):
        """
        Initialize data processor.
        
        Args:
            volatility_lookback: Periods for volatility calculation
            volume_lookback: Periods for volume analysis
        """
        self.volatility_lookback = volatility_lookback
        self.volume_lookback = volume_lookback
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data for RRS analysis.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Processed DataFrame with all required metrics
        """
        if df.empty:
            logger.warning("Empty DataFrame provided")
            return df
        
        required_columns = ['close', 'timestamp']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"Missing required columns: {required_columns}")
            return df
        
        logger.debug(f"Processing data for {len(df)} data points")
        
        # Create copy
        result_df = df.copy()
        
        # Ensure sorted by timestamp
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate returns and volatility
        result_df = self._calculate_returns_and_volatility(result_df)
        
        # Calculate volume metrics
        if 'volume' in result_df.columns:
            result_df = self._calculate_volume_metrics(result_df)
        else:
            logger.warning("Volume column not found, skipping volume metrics")
            result_df['volume_ratio'] = 1.0
        
        # Calculate technical indicators
        result_df = self._calculate_technical_indicators(result_df)
        
        logger.info(f"Successfully processed data with {len(result_df.columns)} columns")
        
        return result_df
    
    def _calculate_returns_and_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and volatility metrics."""
        result_df = df.copy()
        
        # Basic returns
        result_df['price_change'] = result_df['close'].diff()
        result_df['simple_return'] = result_df['close'].pct_change()
        result_df['log_return'] = np.log(result_df['close'] / result_df['close'].shift(1))
        
        # Cumulative returns
        result_df['cumulative_return'] = (1 + result_df['simple_return']).cumprod() - 1
        result_df['cumulative_log_return'] = result_df['log_return'].cumsum()
        
        # Volatility (rolling standard deviation)
        result_df['volatility'] = result_df['log_return'].rolling(
            window=self.volatility_lookback,
            min_periods=max(1, self.volatility_lookback // 2)
        ).std()
        
        # Annualized volatility
        result_df['annualized_volatility'] = result_df['volatility'] * np.sqrt(252)
        
        # Realized volatility (high-low based)
        if 'high' in result_df.columns and 'low' in result_df.columns:
            result_df['hl_volatility'] = np.log(result_df['high'] / result_df['low'])
            result_df['rolling_hl_volatility'] = result_df['hl_volatility'].rolling(
                window=self.volatility_lookback,
                min_periods=max(1, self.volatility_lookback // 2)
            ).mean()
        
        # Price momentum
        result_df['momentum_5'] = result_df['close'] / result_df['close'].shift(5) - 1
        result_df['momentum_10'] = result_df['close'] / result_df['close'].shift(10) - 1
        result_df['momentum_20'] = result_df['close'] / result_df['close'].shift(20) - 1
        
        # Moving averages
        result_df['ma_5'] = result_df['close'].rolling(window=5, min_periods=1).mean()
        result_df['ma_20'] = result_df['close'].rolling(window=20, min_periods=1).mean()
        result_df['ma_50'] = result_df['close'].rolling(window=50, min_periods=1).mean()
        
        # Handle infinite values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        return result_df
    
    def _calculate_volume_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based metrics."""
        result_df = df.copy()
        
        # Volume moving average
        result_df['volume_ma'] = result_df['volume'].rolling(
            window=self.volume_lookback,
            min_periods=max(1, self.volume_lookback // 2)
        ).mean()
        
        # Volume ratio
        result_df['volume_ratio'] = result_df['volume'] / result_df['volume_ma']
        
        # VWAP approximation
        if 'high' in result_df.columns and 'low' in result_df.columns:
            result_df['typical_price'] = (
                (result_df['high'] + result_df['low'] + result_df['close']) / 3
            )
            result_df['volume_price'] = result_df['typical_price'] * result_df['volume']
            
            result_df['vwap'] = (
                result_df['volume_price'].rolling(window=self.volume_lookback, min_periods=1).sum() /
                result_df['volume'].rolling(window=self.volume_lookback, min_periods=1).sum()
            )
        
        # Volume trend
        result_df['volume_trend'] = result_df['volume'].rolling(
            window=self.volume_lookback,
            min_periods=max(1, self.volume_lookback // 2)
        ).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
            raw=True
        )
        
        # Volume spikes
        volume_std = result_df['volume'].rolling(
            window=self.volume_lookback,
            min_periods=max(1, self.volume_lookback // 2)
        ).std()
        
        result_df['volume_spike'] = (
            (result_df['volume'] - result_df['volume_ma']) / volume_std
        )
        
        # Handle infinite values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Drop intermediate columns
        cols_to_drop = ['typical_price', 'volume_price', 'volume_ma']
        result_df = result_df.drop(
            columns=[col for col in cols_to_drop if col in result_df.columns]
        )
        
        return result_df
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional technical indicators."""
        result_df = df.copy()
        
        # RSI (Relative Strength Index)
        delta = result_df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        
        rs = avg_gain / (avg_loss + 1e-10)
        result_df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        if 'close' in result_df.columns:
            bb_period = 20
            bb_std = 2
            result_df['bb_middle'] = result_df['close'].rolling(window=bb_period, min_periods=1).mean()
            bb_std_dev = result_df['close'].rolling(window=bb_period, min_periods=1).std()
            result_df['bb_upper'] = result_df['bb_middle'] + (bb_std_dev * bb_std)
            result_df['bb_lower'] = result_df['bb_middle'] - (bb_std_dev * bb_std)
            result_df['bb_position'] = (
                (result_df['close'] - result_df['bb_lower']) /
                (result_df['bb_upper'] - result_df['bb_lower'] + 1e-10)
            )
        
        # MACD
        exp12 = result_df['close'].ewm(span=12, min_periods=1).mean()
        exp26 = result_df['close'].ewm(span=26, min_periods=1).mean()
        result_df['macd_line'] = exp12 - exp26
        result_df['macd_signal'] = result_df['macd_line'].ewm(span=9, min_periods=1).mean()
        result_df['macd_histogram'] = result_df['macd_line'] - result_df['macd_signal']
        
        # ATR (Average True Range)
        if 'high' in result_df.columns and 'low' in result_df.columns:
            high_low = result_df['high'] - result_df['low']
            high_close = np.abs(result_df['high'] - result_df['close'].shift(1))
            low_close = np.abs(result_df['low'] - result_df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            result_df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        
        # Handle infinite values
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        return result_df

